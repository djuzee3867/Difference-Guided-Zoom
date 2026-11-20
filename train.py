import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_jpeg, read_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgz import (cam_inputs, combine, guide_map, make_jitter, render, to_canvas,
                 two_pass, view_theta, zoom_theta)
from metrics import compute_metrics, load_rings

DELTAS = [0.1, 0.2, 0.3, 0.5, 1.0]


def read_split(splits_dir, split, keep):
    with open(os.path.join(splits_dir, f"{split}.csv"), encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if r["label"] in keep]


class JpegFiles(Dataset):
    def __init__(self, paths, labels, decode):
        self.paths, self.labels, self.decode = paths, labels, decode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        data = read_file(self.paths[i])
        if self.decode == "cpu":
            data = decode_jpeg(data, mode=ImageReadMode.RGB)
        return data, self.labels[i], i


def collate(batch):
    data, ys, idx = zip(*batch)
    shapes = [tuple(x.shape) for x in data]
    flat = torch.cat([x.reshape(-1) for x in data])
    return (flat, shapes), torch.tensor(ys), torch.tensor(idx)


def unpack(packed):
    flat, shapes = packed
    out, pos = [], 0
    for s in shapes:
        n = math.prod(s)
        out.append(flat[pos:pos + n].view(s))
        pos += n
    return out


class PKSampler:
    def __init__(self, labels, p, k, seed):
        self.by_class = defaultdict(list)
        for i, y in enumerate(labels):
            self.by_class[y].append(i)
        self.classes = np.array(sorted(self.by_class))
        w = np.sqrt([len(self.by_class[c]) for c in self.classes])
        self.prob = w / w.sum()
        self.p, self.k, self.seed, self.epoch = p, k, seed, 0
        self.n_batches = len(labels) // (p * k)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        pools = {}
        for _ in range(self.n_batches):
            batch = []
            for c in rng.choice(self.classes, size=self.p, replace=False, p=self.prob):
                for _ in range(self.k):
                    if not pools.get(c):
                        pools[c] = list(rng.permutation(self.by_class[c]))
                    batch.append(int(pools[c].pop()))
            yield batch


def load_batch(data, device, decode):
    if decode == "gpu":
        imgs = decode_jpeg(unpack(data), mode=ImageReadMode.RGB, device=device)
    else:
        flat, shapes = data
        imgs = unpack((flat.to(device, non_blocking=True), shapes))
    return to_canvas(imgs, device)


@torch.no_grad()
def predict(model, loader, method, device, decode):
    model.eval()
    p_full, p_zoom, sides = [], [], []
    t0 = time.time()
    for data, _, _ in loader:
        canvas, hw = load_batch(data, device, decode)
        theta = view_theta(hw, False)
        if method == "base":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p_full.append(model(render(canvas, theta)).float().softmax(1).cpu())
            continue
        p, pz, _, _, side = two_pass(model, canvas, theta, method)
        p_full.append(p.cpu())
        p_zoom.append(pz.cpu())
        sides.append(side.cpu())
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    p_full = torch.cat(p_full).numpy()
    if method == "base":
        return p_full, None, None, elapsed
    return p_full, torch.cat(p_zoom).numpy(), torch.cat(sides).numpy(), elapsed


def evaluate(p_full, p_zoom, labels, obs, rings, method, delta=None):
    if method == "base":
        return compute_metrics(p_full, labels, obs, rings), None, None
    best = None
    for dl in ([delta] if delta is not None else DELTAS):
        p, unsure = combine(p_full, p_zoom, dl)
        m = compute_metrics(p, labels, obs, rings)
        m["delta"], m["zoomed_frac"] = dl, float(unsure.mean())
        if best is None or m["macro_f1"] > best[0]["macro_f1"]:
            best = (m, dl, p)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits_dir", required=True,
                    help="folder with classes.json, train.csv, val.csv, test.csv (columns label, photo_id, observation_id)")
    ap.add_argument("--images", required=True, help="images written by preprocess_ip.py: <images>/<Genus_species>/<photo_id>.jpg")
    ap.add_argument("--rings", default="", help="JSON file of mimicry rings, used for MCR (omit to skip MCR)")
    ap.add_argument("--model", default="efficientnet_b0", help="timm model name, e.g. efficientnet_b0, resnet50")
    ap.add_argument("--method", default="base", choices=["base", "dgz", "camzoom"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--zoom_start", type=int, default=4, help="epoch (from 1) at which the zoom loss starts")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--p", type=int, default=32, help="species per batch")
    ap.add_argument("--k", type=int, default=4, help="images per species in a batch")
    ap.add_argument("--subset", default="", help="CSV of species to keep (column species), for quick tests")
    ap.add_argument("--decode", default="cpu", choices=["gpu", "cpu"],
                    help="decode JPEGs in CPU workers (default) or on the GPU with nvjpeg")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--eval_batch", type=int, default=256)
    ap.add_argument("--drop_path", type=float, default=0.0, help="stochastic depth rate (0 = model default)")
    ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing: less GPU memory, slower")
    ap.add_argument("--name", default="")
    ap.add_argument("--out_root", default="runs")
    args = ap.parse_args()

    short = {"efficientnet_b0": "B0", "resnet50": "R50", "convnext_base.fb_in22k_ft_in1k": "CNB"}.get(args.model, args.model)
    name = args.name or f"{short}-{args.method}-lr{args.lr:g}-s{args.seed}"
    out = os.path.join(args.out_root, name)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "test_metrics.json")):
        print(f"{name} already finished, skipping")
        return
    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    with open(os.path.join(args.splits_dir, "classes.json"), encoding="utf-8") as f:
        classes = json.load(f)
    if args.subset:
        with open(args.subset, encoding="utf-8-sig", newline="") as f:
            keep = {r["species"].replace("_", " ") for r in csv.DictReader(f)}
        classes = [c for c in classes if c in keep]
    cls_idx = {c: i for i, c in enumerate(classes)}
    keep = set(classes)
    rows = {s: read_split(args.splits_dir, s, keep) for s in ("train", "val", "test")}
    if not os.path.isdir(args.images):
        sys.exit(f"image folder {args.images} not found, run preprocess_ip.py first")

    def files(rs):
        return [os.path.join(args.images, r["label"].replace(" ", "_"), f"{r['photo_id']}.jpg") for r in rs]

    labels = {s: np.array([cls_idx[r["label"]] for r in rs]) for s, rs in rows.items()}
    obs = {s: np.array([r["observation_id"] for r in rs]) for s, rs in rows.items()}
    rings = load_rings(args.rings, classes) if args.rings else {}
    print(f"{name}: {len(classes)} species | train {len(rows['train']):,} val {len(rows['val']):,} test {len(rows['test']):,}",
          flush=True)

    sampler = PKSampler(labels["train"].tolist(), args.p, args.k, args.seed)
    common = dict(num_workers=args.workers, collate_fn=collate, pin_memory=True, persistent_workers=True,
                  prefetch_factor=4)
    train_loader = DataLoader(JpegFiles(files(rows["train"]), labels["train"].tolist(), args.decode),
                              batch_sampler=sampler, **common)

    def eval_loader(s):
        return DataLoader(JpegFiles(files(rows[s]), labels[s].tolist(), args.decode), batch_size=args.eval_batch,
                          shuffle=False, **common)

    val_loader = eval_loader("val")

    model = timm.create_model(args.model, pretrained=True, num_classes=len(classes),
                              **({"drop_path_rate": args.drop_path} if args.drop_path > 0 else {}))
    if args.grad_ckpt:
        model.set_grad_checkpointing(True)
    model = model.to(device).to(memory_format=torch.channels_last)
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0}],
                            lr=args.lr)
    steps, warm = args.epochs * len(sampler), args.warmup * len(sampler)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: (t + 1) / warm if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, steps - warm))))
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    start, best, log = 1, None, []
    last_ckpt = os.path.join(out, "last.pt")
    if os.path.exists(last_ckpt):
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"].cpu() if hasattr(ck["gen"], "cpu") else ck["gen"])
        start, best, log = ck["epoch"] + 1, ck["best"], ck["log"]
        print(f"resuming from epoch {ck['epoch']}", flush=True)
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args) | {"name": name, "n_classes": len(classes), "steps_per_epoch": len(sampler)}, f, indent=1)

    for epoch in range(start, args.epochs + 1):
        model.train()
        sampler.set_epoch(epoch)
        zoom_on = args.method != "base" and epoch >= args.zoom_start
        t0, n_img, sum_loss, sum_lz, sum_side = time.time(), 0, 0.0, 0.0, 0.0
        for data, y, _ in train_loader:
            y = y.to(device, non_blocking=True)
            canvas, hw = load_batch(data, device, args.decode)
            theta = view_theta(hw, True, gen)
            jitter = make_jitter(len(y), device, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats = model.forward_features(render(canvas, theta, jitter))
                logits = model.forward_head(feats)
            loss = F.cross_entropy(logits.float(), y, label_smoothing=0.1)
            loss.backward()
            sum_loss += loss.item() * len(y)
            if zoom_on:
                with torch.no_grad():
                    wrong = logits.detach().float().clone()
                    wrong[torch.arange(len(y), device=device), y] = float("-inf")
                    weight, f = cam_inputs(model, feats)
                    d = guide_map(f, weight, y, wrong.argmax(1), args.method)
                    theta_z, side = zoom_theta(d, theta)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lz = F.cross_entropy(model(render(canvas, theta_z, jitter)).float(), y, label_smoothing=0.1)
                lz.backward()
                sum_lz += lz.item() * len(y)
                sum_side += side.sum().item()
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            n_img += len(y)
        train_time = time.time() - t0

        p_full, p_zoom, _, val_time = predict(model, val_loader, args.method, device, args.decode)
        m, delta, _ = evaluate(p_full, p_zoom, labels["val"], obs["val"], rings, args.method)
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": sum_loss / n_img,
               "train_loss_zoom": sum_lz / n_img if zoom_on else None,
               "zoom_side": sum_side / n_img if zoom_on else None,
               "train_img_s": n_img / train_time, "train_min": train_time / 60, "val_s": val_time,
               "val_top1": m["top1"], "val_macro_f1": m["macro_f1"], "val_mcr": m["mcr"], "val_delta": delta,
               "val_zoomed": m.get("zoomed_frac"), "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
        log.append(row)
        improved = best is None or m["macro_f1"] > best["val_macro_f1"]
        if improved:
            best = {"epoch": epoch, "val_macro_f1": m["macro_f1"], "delta": delta, "val": m}
            torch.save({"model": model.state_dict(), "classes": classes, "delta": delta, "epoch": epoch,
                        "arch": args.model}, os.path.join(out, "best.pt"))
        mcr = f"{m['mcr']:.4f}" if m["mcr"] is not None else "-"
        print(f"epoch {epoch:2d} | loss {row['train_loss']:.3f}"
              + (f" zoom {row['train_loss_zoom']:.3f} (box {row['zoom_side']:.2f})" if zoom_on else "")
              + f" | val top1 {m['top1']:.4f} F1 {m['macro_f1']:.4f} MCR {mcr}"
              + (f" δ={delta} zoomed {m['zoomed_frac']:.0%}" if delta is not None else "")
              + f" | {row['train_img_s']:.0f} img/s {row['train_min']:.1f} min" + (" *" if improved else ""), flush=True)
        with open(os.path.join(out, "log.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
            w.writeheader()
            w.writerows(log)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "gen": gen.get_state(), "epoch": epoch, "best": best, "log": log}, last_ckpt)

    ck = torch.load(os.path.join(out, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    test_loader = eval_loader("test")
    p_full, p_zoom, sides, test_time = predict(model, test_loader, args.method, device, args.decode)
    m, delta, p = evaluate(p_full, p_zoom, labels["test"], obs["test"], rings, args.method, ck["delta"])
    if p is None:
        p = p_full
    m.update(best_epoch=ck["epoch"], val=best["val"], test_s=test_time,
             ms_per_image=1000 * test_time / len(labels["test"]),
             params_m=sum(x.numel() for x in model.parameters()) / 1e6)
    if args.method != "base":
        m["by_delta"] = {str(dl): {k: v for k, v in evaluate(p_full, p_zoom, labels["test"], obs["test"], rings,
                                                                 args.method, dl)[0].items() if k != "per_ring"}
                         for dl in DELTAS}
        m["no_zoom"] = {k: v for k, v in compute_metrics(p_full, labels["test"], obs["test"], rings).items()
                        if k != "per_ring"}
    unsure = combine(p_full, p_zoom, delta)[1] if args.method != "base" else np.zeros(len(p), bool)
    top5 = np.argsort(-p, axis=1)[:, :5]
    with open(os.path.join(out, "test_preds.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["photo_id", "observation_id", "label", "pred", "p_pred", "correct", "zoomed", "top5"])
        for r, y, t5, pr, z in zip(rows["test"], labels["test"], top5, p, unsure):
            w.writerow([r["photo_id"], r["observation_id"], classes[y], classes[t5[0]], f"{pr[t5[0]]:.4f}",
                        int(t5[0] == y), int(z), "|".join(classes[c] for c in t5)])
    np.save(os.path.join(out, "test_probs.npy"), p.astype(np.float16))
    with open(os.path.join(out, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1, ensure_ascii=False)
    os.remove(last_ckpt)
    mcr = f"{m['mcr']:.4f}" if m["mcr"] is not None else "-"
    print(f"TEST {name}: top1 {m['top1']:.4f} top5 {m.get('top5', 0):.4f} macroF1 {m['macro_f1']:.4f} "
          f"MCR {mcr} obs {m['obs_top1']:.4f}"
          + (f" | δ={delta} zoomed {m['zoomed_frac']:.0%}" if delta is not None else ""), flush=True)


if __name__ == "__main__":
    main()
