import math

import numpy as np
import torch
import torch.nn.functional as F

CANVAS = 500
OUT = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def to_canvas(imgs, device):
    canvas = torch.zeros(len(imgs), 3, CANVAS, CANVAS, device=device)
    hw = torch.zeros(len(imgs), 2, device=device)
    for i, x in enumerate(imgs):
        x = x.to(device)
        h, w = x.shape[1:]
        if max(h, w) > CANVAS:
            s = CANVAS / max(h, w)
            h, w = max(1, round(h * s)), max(1, round(w * s))
            x = F.interpolate(x[None].float(), size=(h, w), mode="bilinear", antialias=True, align_corners=False)[0]
        canvas[i, :, :h, :w] = x
        hw[i, 0], hw[i, 1] = h, w
    return canvas, hw


def view_theta(hw, train, gen=None):
    b, dev = hw.shape[0], hw.device
    h, w = hw[:, 0], hw[:, 1]

    def u(lo, hi):
        return torch.empty(b, device=dev).uniform_(lo, hi, generator=gen)

    if train:
        area, ratio = h * w * u(0.5, 1.0), u(math.log(3 / 4), math.log(4 / 3)).exp()
        bw, bh = (area * ratio).sqrt().minimum(w), (area / ratio).sqrt().minimum(h)
        x0, y0 = u(0, 1) * (w - bw), u(0, 1) * (h - bh)
        fx = torch.where(u(0, 1) < 0.5, -1.0, 1.0)
        ang = u(-15, 15) * math.pi / 180
    else:
        bw, bh, x0, y0 = w, h, torch.zeros_like(w), torch.zeros_like(h)
        fx, ang = torch.ones_like(w), torch.zeros_like(w)
    cx, cy = (x0 + bw / 2) / CANVAS * 2 - 1, (y0 + bh / 2) / CANVAS * 2 - 1
    sx, sy = bw / CANVAS, bh / CANVAS
    cos, sin = ang.cos(), ang.sin()
    theta = torch.stack([
        torch.stack([cos * sx * fx, -sin * sy, cx], 1),
        torch.stack([sin * sx * fx, cos * sy, cy], 1),
    ], 1)
    return theta


def render(canvas, theta, jitter=None, normalize=True):
    grid = F.affine_grid(theta, (theta.shape[0], 3, OUT * 2, OUT * 2), align_corners=False)
    x = F.avg_pool2d(F.grid_sample(canvas, grid, mode="bilinear", padding_mode="zeros", align_corners=False), 2)
    if jitter is not None:
        bright, contrast = jitter[:, 0].view(-1, 1, 1, 1), jitter[:, 1].view(-1, 1, 1, 1)
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (((x - m) * contrast + m) * bright).clamp(0, 255)
    if not normalize:
        return x
    x = (x - MEAN.to(x.device)) / STD.to(x.device)
    return x.contiguous(memory_format=torch.channels_last)


def make_jitter(b, device, gen=None):
    return torch.empty(b, 2, device=device).uniform_(0.8, 1.2, generator=gen)


def cam_inputs(model, feats):
    weight = model.get_classifier().weight
    norm = getattr(getattr(model, "head", None), "norm", None)
    if norm is None or isinstance(norm, torch.nn.Identity):
        return weight, feats
    return weight * norm.weight, feats - feats.mean(1, keepdim=True)


def guide_map(feats, weight, a, b, kind="dgz"):
    f = feats.detach().float()
    if kind == "dgz":
        return torch.einsum("bc,bchw->bhw", (weight[a] - weight[b]).detach().float(), f).abs()
    return F.relu(torch.einsum("bc,bchw->bhw", weight[a].detach().float(), f))


def zoom_box(d, thr=0.5, expand=0.15, min_side=0.35):
    dev = d.device
    du = F.interpolate(d[:, None], size=(OUT, OUT), mode="bilinear", align_corners=False)[:, 0]
    mask = du >= thr * du.amax(dim=(1, 2), keepdim=True)
    ar = torch.arange(OUT, device=dev)
    rows, cols = mask.any(2), mask.any(1)
    y0 = torch.where(rows, ar, OUT).amin(1).float()
    y1 = torch.where(rows, ar, -1).amax(1).float() + 1
    x0 = torch.where(cols, ar, OUT).amin(1).float()
    x1 = torch.where(cols, ar, -1).amax(1).float() + 1
    side = (torch.maximum(x1 - x0, y1 - y0) * (1 + expand)).clamp(min_side * OUT, OUT)
    cx = ((x0 + x1) / 2).clamp(side / 2, OUT - side / 2)
    cy = ((y0 + y1) / 2).clamp(side / 2, OUT - side / 2)
    return cx, cy, side


def zoom_theta(d, theta_view, **box_kw):
    cx, cy, side = zoom_box(d, **box_kw)
    b = d.shape[0]
    cz = torch.stack([cx / OUT * 2 - 1, cy / OUT * 2 - 1], 1)
    sz = (side / OUT).view(b, 1, 1)
    m = theta_view[:, :, :2]
    t = theta_view[:, :, 2] + torch.einsum("bij,bj->bi", m, cz)
    return torch.cat([m * sz, t[:, :, None]], 2), side / OUT


@torch.no_grad()
def two_pass(model, canvas, theta, kind="dgz"):
    amp = canvas.device.type == "cuda"
    with torch.autocast(canvas.device.type, dtype=torch.bfloat16, enabled=amp):
        feats = model.forward_features(render(canvas, theta))
        logits = model.forward_head(feats)
    p = logits.float().softmax(1)
    top2 = p.topk(2, dim=1).indices
    weight, f = cam_inputs(model, feats)
    d = guide_map(f, weight, top2[:, 0], top2[:, 1], kind)
    theta_z, side = zoom_theta(d, theta)
    with torch.autocast(canvas.device.type, dtype=torch.bfloat16, enabled=amp):
        pz = model(render(canvas, theta_z)).float().softmax(1)
    return p, pz, d, theta_z, side


def combine(p_full, p_zoom, delta):
    top2 = -np.sort(-p_full, axis=1)[:, :2]
    unsure = (top2[:, 0] - top2[:, 1]) < delta
    p = p_full.copy()
    p[unsure] = (p_full[unsure] + p_zoom[unsure]) / 2
    return p, unsure
