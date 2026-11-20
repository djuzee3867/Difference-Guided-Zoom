import argparse
import json
import os
import sys

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgz import CANVAS, OUT, combine, render, to_canvas, two_pass, view_theta, zoom_box
from preprocess_ip import apply_ip


def prepare(path):
    with Image.open(path) as im:
        a = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    h, w = a.shape[:2]
    if max(h, w) > CANVAS:
        s = CANVAS / max(h, w)
        a = cv2.resize(a, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    bgr = apply_ip(cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def save_vis(path, rgb, d, box, zoom_rgb):
    h, w = rgb.shape[:2]
    heat = F.interpolate(d[None, None].float(), size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    heat = (heat / heat.max().clamp_min(1e-9)).cpu().numpy()[..., None] * 0.75
    gray = rgb.astype(np.float32).mean(2, keepdims=True).repeat(3, 2)
    over = (gray * (1 - heat) + np.array([245, 130, 32], np.float32) * heat).clip(0, 255).astype(np.uint8)
    photo = Image.fromarray(rgb)
    draw = ImageDraw.Draw(photo)
    x0, y0, x1, y1 = box[0] * w, box[1] * h, (box[0] + box[2]) * w, (box[1] + box[3]) * h
    draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 255), width=6)
    draw.rectangle((x0, y0, x1, y1), outline=(20, 48, 110), width=3)
    zoom = Image.fromarray(zoom_rgb).resize((h, h), Image.BICUBIC)
    sheet = Image.new("RGB", (w * 2 + h + 16, h), (255, 255, 255))
    for im, x in ((photo, 0), (Image.fromarray(over), w + 8), (zoom, 2 * w + 16)):
        sheet.paste(im, (x, 0))
    sheet.save(path, quality=92)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--ckpt", required=True, help="best.pt from train.py")
    ap.add_argument("--arch", default="", help="timm model name (not needed if the checkpoint stores it)")
    ap.add_argument("--delta", type=float, default=None, help="default: read from the checkpoint; 1.0 = zoom on every image")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--vis", default="", help="folder to save explanation images")
    ap.add_argument("--json", action="store_true", help="print one JSON line per image")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    arch = args.arch or ck.get("arch") or sys.exit("checkpoint has no model name, pass --arch, e.g. efficientnet_b0")
    classes = ck["classes"]
    delta = args.delta if args.delta is not None else (ck.get("delta") or 1.0)
    model = timm.create_model(arch, pretrained=False, num_classes=len(classes))
    model.load_state_dict(ck["model"])
    model = model.to(device).to(memory_format=torch.channels_last).eval()
    if args.vis:
        os.makedirs(args.vis, exist_ok=True)

    def top(p):
        return [{"species": classes[i], "p": round(float(p[i]), 4)} for i in np.argsort(-p)[:args.top]]

    for path in args.images:
        rgb = prepare(path)
        canvas, hw = to_canvas([torch.from_numpy(rgb.copy()).permute(2, 0, 1)], device)
        theta = view_theta(hw, False)
        p, pz, d, theta_z, _ = two_pass(model, canvas, theta)
        cx, cy, side = (v.item() / OUT for v in zoom_box(d))
        box = [cx - side / 2, cy - side / 2, side, side]
        pf, pz = p.cpu().numpy(), pz.cpu().numpy()
        final, zoomed = combine(pf, pz, delta)
        a, b = np.argsort(-pf[0])[:2]
        res = {"image": path, "pair": [classes[a], classes[b]], "box": [round(v, 4) for v in box],
               "zoomed": bool(zoomed[0]), "delta": delta,
               "full": top(pf[0]), "zoom": top(pz[0]), "final": top(final[0])}
        if args.vis:
            zoom_rgb = render(canvas, theta_z, normalize=False)[0].clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            res["vis"] = os.path.join(args.vis, os.path.splitext(os.path.basename(path))[0] + "_dgz.jpg")
            save_vis(res["vis"], rgb, d[0], box, zoom_rgb)
        if args.json:
            print(json.dumps(res, ensure_ascii=False), flush=True)
            continue
        print(f"\n{path}")
        print(f"  first pass: {res['full'][0]['species']} {res['full'][0]['p']:.3f} | confused between {res['pair'][0]} and {res['pair'][1]}")
        print(f"  zoom box (x, y, w, h): {', '.join(f'{v:.2f}' for v in box)} | second pass used: {'yes' if res['zoomed'] else 'no'} (δ={delta})")
        for i, t in enumerate(res["final"], 1):
            print(f"  {i}. {t['species']:<40} {t['p']:.3f}")
        if args.vis:
            print(f"  explanation image: {res['vis']}")


if __name__ == "__main__":
    main()
