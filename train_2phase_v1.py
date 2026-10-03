#!/usr/bin/env python3
"""Прогон fno_2phase_v1.ipynb на старых абсолютных полях data_2phase.

Лосс и отбор чекпоинта по metrics.md — в proxy_metrics.py и train_v2.py.
Здесь R^2 только подпись диагностики, чекпоинт по-прежнему relative L2 полей.
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_2phase"
OUT_DIR = ROOT / "runs" / "2phase_v1"
MODEL_PATH = OUT_DIR / "best_model_2phase_v1.pt"

SEED = 42
WELL_SIGMA = 1.5
IN_CH = 4
MODES = 18
WIDTH = 32
LAYERS = 3
EPOCHS = 300
LR = 1e-3
STEP = 50
W_PRES = 1.0
W_SAT = 1.0
BATCH = 10

_ref = np.zeros((15, 15))
_ref[7, 7] = 1.0
_PEAK = gaussian_filter(_ref, sigma=WELL_SIGMA, mode="constant").max()


def smooth_mask(points, nx, ny):
    m = np.zeros((nx, ny))
    for i, j in points:
        m[i, j] += 1.0
    m = gaussian_filter(m, sigma=WELL_SIGMA, mode="constant")
    return m / _PEAK


def mm(a, lo, hi):
    return (a - lo) / (hi - lo) if hi > lo else a * 0


class LpLoss:
    def __init__(self, p=2):
        self.p = p

    def __call__(self, x, y):
        n = x.size(0)
        d = torch.norm(x.reshape(n, -1) - y.reshape(n, -1), self.p, 1)
        b = torch.norm(y.reshape(n, -1), self.p, 1)
        return torch.mean(d / b)


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, m1, m2):
        super().__init__()
        self.m1, self.m2 = m1, m2
        s = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(s * torch.rand(in_ch, out_ch, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(s * torch.rand(in_ch, out_ch, m1, m2, dtype=torch.cfloat))

    def forward(self, x):
        b, _, nx, ny = x.shape
        xf = torch.fft.rfft2(x)
        o = torch.zeros(b, self.w1.shape[1], nx, ny // 2 + 1, dtype=torch.cfloat, device=x.device)
        o[:, :, : self.m1, : self.m2] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, : self.m1, : self.m2], self.w1
        )
        o[:, :, -self.m1 :, : self.m2] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, -self.m1 :, : self.m2], self.w2
        )
        return torch.fft.irfft2(o, s=(nx, ny))


class TwoPhaseFNO(nn.Module):
    def __init__(self, in_ch=IN_CH, modes=MODES, width=WIDTH, n_layers=LAYERS):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(in_ch, width)
        self.specs = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)]
        )
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.head_p = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_s = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        b, nx, ny, _ = x.shape
        x = self.fc0(x).permute(0, 3, 1, 2)
        for spec, w in zip(self.specs, self.ws):
            x1 = spec(x)
            x2 = w(x.reshape(b, self.width, -1)).reshape(b, self.width, nx, ny)
            x = F.relu(x1 + x2)
        x = x.permute(0, 2, 3, 1)
        p = self.head_p(x).squeeze(-1)
        s = self.head_s(x).squeeze(-1)
        return p, s


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_samples(files):
    bad = []
    good = []
    raw_perm, raw_pres, raw_sat = [], [], []
    raw_coords, raw_wtype = [], []
    t0 = time.time()
    for idx, fpath in enumerate(files):
        if idx % 100 == 0:
            print(f"проверено {idx}/{len(files)}", flush=True)
        try:
            d = sio.loadmat(fpath)
        except Exception as e:
            bad.append((fpath.name, f"не читается: {e}"))
            continue
        pres = d.get("pressure_traj")
        sat = d.get("saturation_traj")
        perm = d.get("perm_field")
        if pres is None or sat is None or perm is None:
            bad.append((fpath.name, "отсутствует поле"))
            continue
        if np.isnan(pres).any() or np.isinf(pres).any():
            bad.append((fpath.name, "NaN/Inf в давлении"))
            continue
        if np.isnan(sat).any() or np.isinf(sat).any():
            bad.append((fpath.name, "NaN/Inf в насыщенности"))
            continue
        if sat.min() < -1e-6 or sat.max() > 1 + 1e-6:
            bad.append((fpath.name, f"насыщенность вне [0,1]: [{sat.min():.3f}, {sat.max():.3f}]"))
            continue
        sat_change = np.abs(sat[:, :, -1] - sat[:, :, 0]).max()
        if sat_change < 0.01:
            bad.append((fpath.name, f"насыщенность не меняется ({sat_change:.4f})"))
            continue
        good.append(fpath)
        raw_perm.append(np.asarray(perm, dtype=np.float32))
        raw_pres.append(np.asarray(pres, dtype=np.float32))
        raw_sat.append(np.asarray(sat, dtype=np.float32))
        raw_coords.append(np.asarray(d["well_coords"]).astype(int))
        raw_wtype.append(np.asarray(d["well_type"]).astype(int).ravel())
    print(f"чтение {time.time() - t0:.1f} c | годных {len(good)} | битых {len(bad)}", flush=True)
    for name, reason in bad[:30]:
        print(f"  {name}: {reason}")
    if len(bad) > 30:
        print(f"  ... ещё {len(bad) - 30}")
    return good, raw_perm, raw_pres, raw_sat, raw_coords, raw_wtype


def build_pairs(raw_perm, raw_pres, raw_sat, raw_coords, raw_wtype):
    perm_all = np.stack(raw_perm)
    pres_all = np.concatenate([p.reshape(-1) for p in raw_pres])
    sat_all = np.concatenate([s.reshape(-1) for s in raw_sat])
    log_perm = np.log10(perm_all)
    norm = {
        "KMIN": float(log_perm.min()),
        "KMAX": float(log_perm.max()),
        "PMIN": float(pres_all.min()),
        "PMAX": float(pres_all.max()),
        "SMIN": float(sat_all.min()),
        "SMAX": float(sat_all.max()),
        "WELL_SIGMA": WELL_SIGMA,
    }
    print(
        f"log10(k): [{norm['KMIN']:.3f}, {norm['KMAX']:.3f}] | "
        f"P [{norm['PMIN']:.2f}, {norm['PMAX']:.2f}] | "
        f"Sw [{norm['SMIN']:.4f}, {norm['SMAX']:.4f}]",
        flush=True,
    )
    xs, yp, ys = [], [], []
    for perm, pres, sat, coords, wtype in zip(raw_perm, raw_pres, raw_sat, raw_coords, raw_wtype):
        nx, ny = perm.shape
        n_steps = pres.shape[2]
        inj = [(i - 1, j - 1) for (i, j), t in zip(coords, wtype) if t == 1]
        prod = [(i - 1, j - 1) for (i, j), t in zip(coords, wtype) if t != 1]
        perm_n = mm(np.log10(perm), norm["KMIN"], norm["KMAX"]).astype(np.float32)
        inj_mask = smooth_mask(inj, nx, ny).astype(np.float32)
        prod_mask = smooth_mask(prod, nx, ny).astype(np.float32)
        for t in range(n_steps):
            t_norm = t / max(n_steps - 1, 1)
            time_ch = np.full((nx, ny), t_norm, dtype=np.float32)
            xs.append(np.stack([perm_n, inj_mask, prod_mask, time_ch], axis=-1))
            yp.append(mm(pres[:, :, t], norm["PMIN"], norm["PMAX"]).astype(np.float32))
            ys.append(mm(sat[:, :, t], norm["SMIN"], norm["SMAX"]).astype(np.float32))
    x = np.stack(xs)
    return x, np.stack(yp), np.stack(ys), norm


def flips4(t):
    """Те же четыре отражения, что flip3 в ноутбуке: исходник, по x, по y, оба."""
    return torch.cat(
        [
            t,
            torch.flip(t, [1]),
            torch.flip(t, [2]),
            torch.flip(torch.flip(t, [1]), [2]),
        ],
        0,
    )


def save_curves(history, path):
    ep = np.arange(len(history["tr"]))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(ep, history["tr"], label="train")
    axes[0].plot(ep, history["va"], label="val")
    axes[0].set_title("relative L2")
    axes[0].legend()
    axes[1].plot(ep, history["mae_p"], label="pressure, bar")
    axes[1].plot(ep, history["mae_s"], label="saturation")
    axes[1].set_title("MAE")
    axes[1].legend()
    axes[2].plot(ep, history["sk_p"], label="pressure")
    axes[2].plot(ep, history["sk_s"], label="saturation")
    axes[2].set_title("R² диагностика, не отбор")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_preview(model, device, norm, sample_path):
    d = sio.loadmat(sample_path)
    perm = np.asarray(d["perm_field"], dtype=np.float64)
    pres = np.asarray(d["pressure_traj"], dtype=np.float64)
    sat = np.asarray(d["saturation_traj"], dtype=np.float64)
    coords = np.asarray(d["well_coords"]).astype(int)
    wtype = np.asarray(d["well_type"]).astype(int).ravel()
    nx, ny = perm.shape
    n_steps = pres.shape[2]
    inj = [(i - 1, j - 1) for (i, j), t in zip(coords, wtype) if t == 1]
    prod = [(i - 1, j - 1) for (i, j), t in zip(coords, wtype) if t != 1]
    perm_n = mm(np.log10(perm), norm["KMIN"], norm["KMAX"])
    inj_mask = smooth_mask(inj, nx, ny)
    prod_mask = smooth_mask(prod, nx, ny)
    show = np.linspace(0, n_steps - 1, min(5, n_steps)).astype(int)
    fig, axes = plt.subplots(4, len(show), figsize=(4 * len(show), 14))

    def denorm_p(a):
        return a * (norm["PMAX"] - norm["PMIN"]) + norm["PMIN"]

    def denorm_s(a):
        return a * (norm["SMAX"] - norm["SMIN"]) + norm["SMIN"]

    model.eval()
    for col, t in enumerate(show):
        t_norm = t / max(n_steps - 1, 1)
        time_ch = np.full((nx, ny), t_norm)
        x = np.stack([perm_n, inj_mask, prod_mask, time_ch], axis=-1)[None, ...].astype(np.float32)
        with torch.no_grad():
            pp, ps = model(torch.from_numpy(x).to(device))
        pred_p = denorm_p(pp.cpu().numpy()[0])
        pred_s = denorm_s(ps.cpu().numpy()[0])
        true_s = sat[:, :, t]
        true_p = pres[:, :, t]
        panels = [
            (true_s, "Sw истина", "Blues", 0, 1),
            (pred_s, "Sw прогноз", "Blues", 0, 1),
            (true_p, "P истина", "viridis", None, None),
            (pred_p, "P прогноз", "viridis", None, None),
        ]
        for row, (field, title, cmap, vmin, vmax) in enumerate(panels):
            im = axes[row, col].imshow(field.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, col].set_title(f"{title}, t={t}", fontsize=9)
            plt.colorbar(im, ax=axes[row, col], fraction=0.046)
    fig.suptitle(sample_path.name, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "preview.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = pick_device()
    print("device:", device, flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob("sample_*.mat"))
    print("найдено файлов:", len(files), flush=True)
    if not files:
        raise SystemExit(f"нет sample_*.mat в {DATA_DIR}")

    good, raw_perm, raw_pres, raw_sat, raw_coords, raw_wtype = load_samples(files)
    all_x, all_yp, all_ys, norm = build_pairs(raw_perm, raw_pres, raw_sat, raw_coords, raw_wtype)
    del raw_perm, raw_pres, raw_sat, raw_coords, raw_wtype
    print(f"пар вход-выход: {all_x.shape[0]}  X {all_x.shape}", flush=True)

    n = all_x.shape[0]
    nval = max(1, int(round(0.2 * n)))
    ntrain = n - nval
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n)
    tr_idx, va_idx = idx[:ntrain], idx[ntrain:]
    x_tr = np.ascontiguousarray(all_x[tr_idx])
    yp_tr = np.ascontiguousarray(all_yp[tr_idx])
    ys_tr = np.ascontiguousarray(all_ys[tr_idx])
    x_va = np.ascontiguousarray(all_x[va_idx])
    yp_va = np.ascontiguousarray(all_yp[va_idx])
    ys_va = np.ascontiguousarray(all_ys[va_idx])
    del all_x, all_yp, all_ys
    x_tr_t = flips4(torch.from_numpy(x_tr))
    yp_tr_t = flips4(torch.from_numpy(yp_tr))
    ys_tr_t = flips4(torch.from_numpy(ys_tr))
    del x_tr, yp_tr, ys_tr
    print(f"train: {ntrain}  val: {len(x_va)}  train с отражениями: {len(x_tr_t)}", flush=True)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr_t, yp_tr_t, ys_tr_t),
        batch_size=BATCH,
        shuffle=True,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(x_va), torch.from_numpy(yp_va), torch.from_numpy(ys_va)
        ),
        batch_size=BATCH,
        shuffle=False,
    )

    model = TwoPhaseFNO().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"параметров: {n_params:,}", flush=True)

    lp = LpLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=0.5)
    pspan = norm["PMAX"] - norm["PMIN"]
    sspan = norm["SMAX"] - norm["SMIN"]
    history = {"tr": [], "va": [], "mae_p": [], "mae_s": [], "sk_p": [], "sk_s": []}
    best_val = float("inf")
    t0 = time.time()

    for ep in range(args.epochs):
        model.train()
        tr_loss = 0.0
        ep_t = time.time()
        for xb, yp, ys in train_loader:
            xb, yp, ys = xb.to(device), yp.to(device), ys.to(device)
            optimizer.zero_grad()
            pp, ps = model(xb)
            loss = W_PRES * lp(pp, yp) + W_SAT * lp(ps, ys)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = mae_p = mae_s = 0.0
        sse_p = ssm_p = sse_s = ssm_s = 0.0
        with torch.no_grad():
            for xb, yp, ys in val_loader:
                xb, yp, ys = xb.to(device), yp.to(device), ys.to(device)
                pp, ps = model(xb)
                va_loss += (W_PRES * lp(pp, yp) + W_SAT * lp(ps, ys)).item() * xb.size(0)
                mae_p += (torch.abs(pp - yp).mean() * pspan).item() * xb.size(0)
                mae_s += (torch.abs(ps - ys).mean() * sspan).item() * xb.size(0)
                bp = yp.mean(dim=(1, 2), keepdim=True)
                bs = ys.mean(dim=(1, 2), keepdim=True)
                sse_p += ((pp - yp) ** 2).sum().item()
                ssm_p += ((bp - yp) ** 2).sum().item()
                sse_s += ((ps - ys) ** 2).sum().item()
                ssm_s += ((bs - ys) ** 2).sum().item()
        n_va = len(val_loader.dataset)
        va_loss /= n_va
        mae_p /= n_va
        mae_s /= n_va
        sk_p = 1 - sse_p / max(ssm_p, 1e-12)
        sk_s = 1 - sse_s / max(ssm_s, 1e-12)
        scheduler.step()
        history["tr"].append(tr_loss)
        history["va"].append(va_loss)
        history["mae_p"].append(mae_p)
        history["mae_s"].append(mae_s)
        history["sk_p"].append(sk_p)
        history["sk_s"].append(sk_s)

        if va_loss < best_val:
            best_val = va_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "norm": norm,
                    "arch": {"in_ch": IN_CH, "modes": MODES, "width": WIDTH, "n_layers": LAYERS},
                    "epoch": ep,
                    "val_loss": float(va_loss),
                },
                MODEL_PATH,
            )
        if ep % 10 == 0 or ep == args.epochs - 1:
            elapsed = time.time() - t0
            ep_sec = time.time() - ep_t
            eta = ep_sec * (args.epochs - ep - 1)
            print(
                f"ep {ep:3d}/{args.epochs} | tr {tr_loss:.4f} | va {va_loss:.4f} "
                f"| MAE_p {mae_p:.2f} бар | MAE_s {mae_s:.4f} "
                f"| r2_p {sk_p:.3f} r2_s {sk_s:.3f} диагн. "
                f"| {ep_sec:.0f}s/ep | eta {eta / 3600:.1f} ч",
                flush=True,
            )
            (OUT_DIR / "history.json").write_text(json.dumps(history))
            save_curves(history, OUT_DIR / "curves.png")

    print(f"обучение: {(time.time() - t0) / 60:.1f} мин", flush=True)
    print(f"лучший val: {best_val:.4f} -> {MODEL_PATH}", flush=True)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    preview = good[len(good) // 2]
    save_preview(model, device, norm, preview)
    summary = {
        "epochs": args.epochs,
        "best_val": best_val,
        "best_epoch": int(ckpt["epoch"]),
        "n_files": len(good),
        "n_pairs": n,
        "final": {k: history[k][-1] for k in history},
        "device": str(device),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
