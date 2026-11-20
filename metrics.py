import json

import numpy as np
from sklearn.metrics import f1_score


def load_rings(path, classes):
    with open(path, encoding="utf-8") as f:
        rings = json.load(f)["rings"]
    idx = {c: i for i, c in enumerate(classes)}
    out = {}
    for name, r in rings.items():
        key = name.split("_")[0]
        key = "C+D" if key in ("C", "D") else key
        members = {idx[m["species"].replace("_", " ")] for m in r["members"] if m["species"].replace("_", " ") in idx}
        out.setdefault(key, set()).update(members)
    return dict(sorted(out.items()))


def compute_metrics(probs, labels, obs_ids, rings, min_per_class=5):
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels)
    n, n_cls = probs.shape
    pred = probs.argmax(1)
    correct = pred == labels
    out = {"n_images": int(n), "top1": float(correct.mean())}
    if n_cls >= 5:
        top5 = np.argpartition(-probs, 5, axis=1)[:, :5]
        out["top5"] = float((top5 == labels[:, None]).any(1).mean())

    counts = np.bincount(labels, minlength=n_cls)
    f1_cls = np.flatnonzero(counts >= min_per_class)
    out["macro_f1"] = float(f1_score(labels, pred, labels=f1_cls, average="macro", zero_division=0))
    out["macro_f1_n_classes"] = int(len(f1_cls))
    out["macro_f1_n_images"] = int(counts[f1_cls].sum())

    ring_of = [set() for _ in range(n_cls)]
    for k, members in rings.items():
        for c in members:
            ring_of[c].add(k)
    in_ring = np.array([bool(ring_of[y]) for y in labels])
    within = np.array([p != y and bool(ring_of[p] & ring_of[y]) for p, y in zip(pred, labels)])
    out["mcr"] = float(within[in_ring].mean()) if in_ring.any() else None
    out["ring_n_images"] = int(in_ring.sum())
    out["top1_ring"] = float(correct[in_ring].mean()) if in_ring.any() else None
    out["top1_normal"] = float(correct[~in_ring].mean()) if (~in_ring).any() else None
    out["normal_n_images"] = int((~in_ring).sum())
    per_ring = {}
    for k, members in rings.items():
        m = np.isin(labels, list(members))
        if m.any():
            wrong_in = np.isin(pred[m], list(members)) & ~correct[m]
            per_ring[k] = {"n": int(m.sum()), "mcr": float(wrong_in.mean()), "top1": float(correct[m].mean())}
    out["per_ring"] = per_ring

    _, inv = np.unique(np.asarray(obs_ids), return_inverse=True)
    n_obs = inv.max() + 1
    summed = np.zeros((n_obs, n_cls), dtype=np.float64)
    np.add.at(summed, inv, probs)
    obs_label = np.zeros(n_obs, dtype=labels.dtype)
    obs_label[inv] = labels
    out["obs_top1"] = float((summed.argmax(1) == obs_label).mean())
    out["n_observations"] = int(n_obs)
    return out


def mcr_bootstrap(pred, labels, obs_ids, rings, n_boot=1000, seed=0):
    pred, labels, obs_ids = np.asarray(pred), np.asarray(labels), np.asarray(obs_ids)
    n_cls = int(max(pred.max(), labels.max())) + 1
    ring_of = [set() for _ in range(n_cls)]
    for k, members in rings.items():
        for c in members:
            ring_of[c].add(k)
    in_ring = np.array([bool(ring_of[y]) for y in labels])
    within = np.array([p != y and bool(ring_of[p] & ring_of[y]) for p, y in zip(pred, labels)])
    obs, inv = np.unique(obs_ids[in_ring], return_inverse=True)
    num = np.bincount(inv, weights=within[in_ring].astype(float), minlength=len(obs))
    den = np.bincount(inv, minlength=len(obs)).astype(float)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        s = rng.integers(0, len(obs), len(obs))
        stats.append(num[s].sum() / den[s].sum())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
