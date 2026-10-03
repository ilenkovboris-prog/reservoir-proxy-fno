#!/usr/bin/env python3
"""Один шаг U-FNO или FNO на data_v2.

Лосс и отчёт — proxy_metrics.py (metrics.md):
relative L2(Δp) + relative L2(ΔSw) + relative L2(q_o).
Чекпоинт — меньший relative L2(q_o) на val-прогонах, не R^2.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch

from proxy_metrics import (
    corey_split,
    mean_scores,
    proxy_loss,
    r2_diagnostic,
    rollout_horizons,
    score_forecast,
)
from train_2phase_v1 import TwoPhaseFNO, pick_device
from train_ufno_v1 import TwoPhaseUFNO

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data_v2"
OUT_DIR = ROOT / "runs" / "v2"

SEED = 0
MODES = 18
WIDTH = 32
LR = 1e-3
STEP = 50
BATCH = 10
IN_CH = 9


def mm(a, lo, hi):
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def denorm(z, lo, hi):
    if hi > lo:
        return z * (hi - lo) + lo
    return torch.full_like(z, lo)


def _scalar(d, key):
    return float(np.squeeze(d[key]))


def load_runs(data_dir):
    files = sorted(Path(data_dir).glob("run_*.mat"))
    if not files:
        raise SystemExit(f"нет run_*.mat в {data_dir}")
    runs = []
    for path in files:
        d = sio.loadmat(path)
        well_type = np.asarray(d["well_type"]).astype(int).ravel()
        runs.append({
            "id": int(np.squeeze(d["run_id"])),
            "perm": np.asarray(d["perm_field"], dtype=np.float64),
            "ntg": np.asarray(d["ntg_field"], dtype=np.float64),
            "pressure": np.asarray(d["pressure_traj"], dtype=np.float64),
            "sw": np.asarray(d["sw_traj"], dtype=np.float64),
            "dp": np.asarray(d["dp"], dtype=np.float64),
            "dsw": np.asarray(d["dsw"], dtype=np.float64),
            "q_map": np.asarray(d["q_map"], dtype=np.float64),
            "q_oil": np.asarray(d["q_oil"], dtype=np.float64),
            "q_water": np.asarray(d["q_water"], dtype=np.float64),
            "q_sched": np.asarray(d["q_sched"], dtype=np.float64),
            "well_coords": np.asarray(d["well_coords"]).astype(int),
            "well_type": well_type,
            "dt": np.asarray(d["dt"], dtype=np.float64).ravel(),
            "kro": _scalar(d, "corey_kro"),
            "krw": _scalar(d, "corey_krw"),
            "no": _scalar(d, "corey_no"),
            "nw": _scalar(d, "corey_nw"),
            "swc": _scalar(d, "corey_swc"),
            "sor": _scalar(d, "corey_sor"),
            "mu_o": _scalar(d, "mu_o"),
            "mu_w": _scalar(d, "mu_w"),
        })
    prod = np.flatnonzero(runs[0]["well_type"] == 0)
    coords = runs[0]["well_coords"]
    for run in runs[1:]:
        if not np.array_equal(run["well_coords"], coords) or not np.array_equal(run["well_type"], runs[0]["well_type"]):
            raise SystemExit(f"скважины прогона {run['id']} отличаются от первого")
    return runs, prod


def load_split(data_dir, run_ids):
    path = Path(data_dir) / "splits.mat"
    if not path.exists():
        raise SystemExit(f"нет {path}")
    d = sio.loadmat(path)
    known = set(run_ids)
    out = {}
    for key in ("train_ids", "val_ids", "test_ids"):
        ids = [int(v) for v in np.asarray(d[key]).ravel()]
        missing = [i for i in ids if i not in known]
        if missing:
            raise SystemExit(f"{key}: нет прогонов {missing}")
        out[key] = ids
    return out


def fit_norm(runs):
    def bounds(chunks):
        cat = np.concatenate([np.asarray(c, dtype=np.float64).ravel() for c in chunks])
        return float(cat.min()), float(cat.max())

    qabs = max(float(np.max(np.abs(r["q_map"]))) for r in runs)
    return {
        "k": bounds([np.log10(r["perm"]) for r in runs]),
        "ntg": bounds([r["ntg"] for r in runs]),
        "p": bounds([r["pressure"] for r in runs]),
        "sw": bounds([r["sw"] for r in runs]),
        "dp": bounds([r["dp"] for r in runs]),
        "ds": bounds([r["dsw"] for r in runs]),
        "kro": bounds([r["kro"] for r in runs]),
        "krw": bounds([r["krw"] for r in runs]),
        "no": bounds([r["no"] for r in runs]),
        "nw": bounds([r["nw"] for r in runs]),
        "qabs": max(qabs, 1.0),
    }


def field_x(run, p_prev, sw_prev, q_map, norm):
    nx, ny = run["perm"].shape
    planes = [
        mm(np.log10(run["perm"]), *norm["k"]),
        mm(run["ntg"], *norm["ntg"]),
        np.full((nx, ny), mm(run["kro"], *norm["kro"])),
        np.full((nx, ny), mm(run["krw"], *norm["krw"])),
        np.full((nx, ny), mm(run["no"], *norm["no"])),
        np.full((nx, ny), mm(run["nw"], *norm["nw"])),
        q_map / norm["qabs"],
        mm(p_prev, *norm["p"]),
        mm(sw_prev, *norm["sw"]),
    ]
    return np.stack(planes, axis=-1).astype(np.float32)


def build_steps(runs, norm, prod_idx):
    steps = []
    for run in runs:
        n_steps = run["dp"].shape[2]
        for t in range(n_steps):
            steps.append({
                "run_id": run["id"],
                "t": t,
                "x": field_x(run, run["pressure"][:, :, t], run["sw"][:, :, t], run["q_map"][:, :, t], norm),
                "dp": run["dp"][:, :, t].astype(np.float32),
                "ds": run["dsw"][:, :, t].astype(np.float32),
                "sw_prev": run["sw"][run["well_coords"][prod_idx, 0] - 1, run["well_coords"][prod_idx, 1] - 1, t].astype(np.float32),
                "q_liq": run["q_sched"][prod_idx, t].astype(np.float32),
                "qo": run["q_oil"][prod_idx, t].astype(np.float32),
                "qw": run["q_water"][prod_idx, t].astype(np.float32),
                "corey": np.array([run["kro"], run["krw"], run["no"], run["nw"], run["swc"], run["sor"]], dtype=np.float32),
                "mu_o": np.float32(run["mu_o"]),
                "mu_w": np.float32(run["mu_w"]),
            })
    return steps


class StepSet(torch.utils.data.Dataset):
    def __init__(self, steps):
        self.steps = steps

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        s = self.steps[idx]
        return (
            torch.from_numpy(s["x"]),
            torch.from_numpy(s["dp"]),
            torch.from_numpy(s["ds"]),
            torch.from_numpy(s["sw_prev"]),
            torch.from_numpy(s["q_liq"]),
            torch.from_numpy(s["qo"]),
            torch.from_numpy(s["corey"]),
            torch.tensor(s["mu_o"]),
            torch.tensor(s["mu_w"]),
        )


def rates_from_delta(sw_prev, ds_hat, q_liq, corey, mu_o, mu_w, prod_i, prod_j):
    sw = (sw_prev + ds_hat[:, prod_i, prod_j]).clamp(0.0, 1.0)
    kro, krw, no, nw, swc, sor = corey.unbind(dim=1)
    view = lambda v: v[:, None]
    return corey_split(
        sw, q_liq, view(kro), view(krw), view(no), view(nw), view(swc), view(sor),
        mu_o[:, None], mu_w[:, None],
    )


def predict_delta(model, x, norm, device):
    dp_n, ds_n = model(x)
    return denorm(dp_n, *norm["dp"]), denorm(ds_n, *norm["ds"])


@torch.no_grad()
def one_step_run(model, run, norm, prod_idx, device):
    model.eval()
    n_steps = run["dp"].shape[2]
    coords = run["well_coords"]
    ii = torch.tensor(coords[prod_idx, 0] - 1, device=device)
    jj = torch.tensor(coords[prod_idx, 1] - 1, device=device)
    dp_hat = np.zeros_like(run["dp"])
    ds_hat = np.zeros_like(run["dsw"])
    p_hat = np.zeros_like(run["dp"])
    sw_hat = np.zeros_like(run["dsw"])
    qo_hat = np.zeros((len(prod_idx), n_steps))
    qw_hat = np.zeros_like(qo_hat)
    corey = torch.tensor(
        [run["kro"], run["krw"], run["no"], run["nw"], run["swc"], run["sor"]],
        dtype=torch.float32, device=device,
    )[None, :]
    mu_o = torch.tensor([run["mu_o"]], dtype=torch.float32, device=device)
    mu_w = torch.tensor([run["mu_w"]], dtype=torch.float32, device=device)
    for t in range(n_steps):
        x = torch.from_numpy(
            field_x(run, run["pressure"][:, :, t], run["sw"][:, :, t], run["q_map"][:, :, t], norm)
        ).unsqueeze(0).to(device)
        dp, ds = predict_delta(model, x, norm, device)
        sw_prev = torch.tensor(
            run["sw"][coords[prod_idx, 0] - 1, coords[prod_idx, 1] - 1, t],
            dtype=torch.float32, device=device,
        )[None, :]
        q_liq = torch.tensor(run["q_sched"][prod_idx, t], dtype=torch.float32, device=device)[None, :]
        qo, qw = rates_from_delta(sw_prev, ds, q_liq, corey, mu_o, mu_w, ii, jj)
        dp_hat[:, :, t] = dp[0].cpu().numpy()
        ds_hat[:, :, t] = ds[0].cpu().numpy()
        p_hat[:, :, t] = run["pressure"][:, :, t] + dp_hat[:, :, t]
        sw_hat[:, :, t] = run["sw"][:, :, t] + ds_hat[:, :, t]
        qo_hat[:, t] = qo[0].cpu().numpy()
        qw_hat[:, t] = qw[0].cpu().numpy()
    true_p = run["pressure"][:, :, 1:]
    true_sw = run["sw"][:, :, 1:]
    return score_forecast(
        qo_hat, run["q_oil"][prod_idx], qw_hat, run["q_water"][prod_idx],
        run["q_sched"][prod_idx], run["dt"],
        np.transpose(p_hat, (2, 0, 1)), np.transpose(true_p, (2, 0, 1)),
        np.transpose(sw_hat, (2, 0, 1)), np.transpose(true_sw, (2, 0, 1)),
        np.transpose(dp_hat, (2, 0, 1)), np.transpose(run["dp"], (2, 0, 1)),
        np.transpose(ds_hat, (2, 0, 1)), np.transpose(run["dsw"], (2, 0, 1)),
    )


@torch.no_grad()
def rollout_run(model, run, norm, prod_idx, device, horizon):
    model.eval()
    coords = run["well_coords"]
    p = run["pressure"][:, :, 0].copy()
    sw = run["sw"][:, :, 0].copy()
    qo_hat = np.zeros((len(prod_idx), horizon))
    qw_hat = np.zeros_like(qo_hat)
    p_hat = np.zeros((horizon, *p.shape))
    sw_hat = np.zeros((horizon, *sw.shape))
    for t in range(horizon):
        x = torch.from_numpy(field_x(run, p, sw, run["q_map"][:, :, t], norm)).unsqueeze(0).to(device)
        dp_n, ds_n = model(x)
        dp = denorm(dp_n, *norm["dp"])[0].cpu().numpy()
        ds = denorm(ds_n, *norm["ds"])[0].cpu().numpy()
        p = p + dp
        sw = np.clip(sw + ds, 0.0, 1.0)
        p_hat[t] = p
        sw_hat[t] = sw
        for k, w in enumerate(prod_idx):
            i, j = coords[w, 0] - 1, coords[w, 1] - 1
            qo, qw = _corey_np(sw[i, j], run["q_sched"][w, t], run)
            qo_hat[k, t] = qo
            qw_hat[k, t] = qw
    sl = slice(0, horizon)
    return score_forecast(
        qo_hat, run["q_oil"][prod_idx, sl], qw_hat, run["q_water"][prod_idx, sl],
        run["q_sched"][prod_idx, sl], run["dt"][sl],
        p_hat, np.transpose(run["pressure"][:, :, 1:horizon + 1], (2, 0, 1)),
        sw_hat, np.transpose(run["sw"][:, :, 1:horizon + 1], (2, 0, 1)),
    )


def _corey_np(sw, q_liq, run):
    sn = min(1.0, max(0.0, (sw - run["swc"]) / (1.0 - run["swc"] - run["sor"])))
    kr_w = run["krw"] * sn ** run["nw"]
    kr_o = run["kro"] * (1.0 - sn) ** run["no"]
    lw = kr_w / run["mu_w"]
    lo = kr_o / run["mu_o"]
    water_cut = lw / (lw + lo + 1e-12)
    return (1.0 - water_cut) * q_liq, water_cut * q_liq


def evaluate_split(model, runs, norm, prod_idx, device, with_rollout=False):
    one = [one_step_run(model, run, norm, prod_idx, device) for run in runs]
    out = {"one_step": mean_scores(one)}
    if with_rollout:
        rolled = {}
        per_run = {}
        keep = ("run_id", "rel_l2_qo", "rel_l2_qw", "mae_qo", "mae_qw", "cum_oil_rel", "mae_p_bar", "mae_sw")
        for h in rollout_horizons(runs[0]["dp"].shape[2]):
            rows = []
            for run in runs:
                row = rollout_run(model, run, norm, prod_idx, device, h)
                row["run_id"] = run["id"]
                rows.append(row)
            rolled[str(h)] = mean_scores(rows)
            per_run[str(h)] = [{k: row[k] for k in keep} for row in rows]
        out["rollout"] = rolled
        out["rollout_runs"] = per_run
    return out


@torch.no_grad()
def collect_qo_scatter(model, runs, norm, prod_idx, device):
    model.eval()
    pred, true = [], []
    for run in runs:
        n_steps = run["dp"].shape[2]
        coords = run["well_coords"]
        ii = torch.tensor(coords[prod_idx, 0] - 1, device=device)
        jj = torch.tensor(coords[prod_idx, 1] - 1, device=device)
        corey = torch.tensor(
            [run["kro"], run["krw"], run["no"], run["nw"], run["swc"], run["sor"]],
            dtype=torch.float32, device=device,
        )[None, :]
        mu_o = torch.tensor([run["mu_o"]], dtype=torch.float32, device=device)
        mu_w = torch.tensor([run["mu_w"]], dtype=torch.float32, device=device)
        for t in range(n_steps):
            x = torch.from_numpy(
                field_x(run, run["pressure"][:, :, t], run["sw"][:, :, t], run["q_map"][:, :, t], norm)
            ).unsqueeze(0).to(device)
            _, ds = predict_delta(model, x, norm, device)
            sw_prev = torch.tensor(
                run["sw"][coords[prod_idx, 0] - 1, coords[prod_idx, 1] - 1, t],
                dtype=torch.float32, device=device,
            )[None, :]
            q_liq = torch.tensor(run["q_sched"][prod_idx, t], dtype=torch.float32, device=device)[None, :]
            qo, _ = rates_from_delta(sw_prev, ds, q_liq, corey, mu_o, mu_w, ii, jj)
            pred.append(qo[0].detach().cpu().numpy())
            true.append(run["q_oil"][prod_idx, t])
    return np.concatenate(true), np.concatenate(pred)


def save_curves(history, path):
    ep = np.arange(len(history["tr"]))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(ep, history["tr"], label="train")
    axes[0].set_title("train rel L2: Δp + ΔSw + qo")
    axes[0].legend()
    axes[1].plot(ep, history["val_rel_qo"], label="qo")
    axes[1].plot(ep, history["val_rel_qw"], label="qw")
    axes[1].set_title("val relative L2, один шаг")
    axes[1].legend()
    axes[2].plot(ep, history["val_mae_p"], label="p, бар")
    axes[2].plot(ep, history["val_mae_sw"], label="Sw")
    axes[2].plot(ep, history["val_mae_qo"], label="qo, м3/сут")
    axes[2].set_title("val MAE")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_scatter(qo_true, qo_hat, path):
    r2 = r2_diagnostic(qo_hat, qo_true)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(qo_true, qo_hat, s=8, alpha=0.5)
    lo = min(qo_true.min(), qo_hat.min())
    hi = max(qo_true.max(), qo_hat.max())
    ax.plot([lo, hi], [lo, hi], color="black", lw=1)
    ax.set_xlabel("qo симулятор, м3/сут")
    ax.set_ylabel("qo прокси, м3/сут")
    ax.set_title(f"R² = {r2:.3f}  (диагностика, не отбор)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--arch", choices=("ufno", "fno"), default="ufno")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = pick_device()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print("device:", device, "arch:", args.arch, flush=True)

    runs, prod_idx = load_runs(args.data_dir)
    split = load_split(args.data_dir, [r["id"] for r in runs])
    by_id = {r["id"]: r for r in runs}
    train_runs = [by_id[i] for i in split["train_ids"]]
    val_runs = [by_id[i] for i in split["val_ids"]]
    test_runs = [by_id[i] for i in split["test_ids"]]
    norm = fit_norm(train_runs)
    print(
        f"прогоны train/val/test {len(train_runs)}/{len(val_runs)}/{len(test_runs)} | "
        f"Δp [{norm['dp'][0]:.2f}, {norm['dp'][1]:.2f}] бар",
        flush=True,
    )

    steps = build_steps(train_runs, norm, prod_idx)
    loader = torch.utils.data.DataLoader(StepSet(steps), batch_size=BATCH, shuffle=True, num_workers=0)
    coords = train_runs[0]["well_coords"]
    prod_i = torch.tensor(coords[prod_idx, 0] - 1, device=device)
    prod_j = torch.tensor(coords[prod_idx, 1] - 1, device=device)

    if args.arch == "ufno":
        model = TwoPhaseUFNO(in_ch=IN_CH, modes=MODES, width=WIDTH).to(device)
    else:
        model = TwoPhaseFNO(in_ch=IN_CH, modes=MODES, width=WIDTH, n_layers=3).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"параметров: {n_params:,}  шагов train: {len(steps)}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=0.5)
    history = {
        "tr": [], "val_rel_qo": [], "val_rel_qw": [],
        "val_mae_qo": [], "val_mae_qw": [], "val_mae_p": [], "val_mae_sw": [],
        "val_cum_oil_rel": [], "val_rel_dp": [], "val_rel_ds": [],
    }
    best = float("inf")
    model_path = out_dir / f"best_model_{args.arch}_v2.pt"
    t0 = time.time()

    if args.eval_only:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        norm = ckpt["norm"]
        best = float(ckpt["val_rel_qo"])
    else:
        for ep in range(args.epochs):
            model.train()
            tr_loss = 0.0
            seen = 0
            ep_t = time.time()
            for xb, dp, ds, sw_prev, q_liq, qo, corey, mu_o, mu_w in loader:
                xb = xb.to(device)
                dp, ds = dp.to(device), ds.to(device)
                sw_prev, q_liq, qo = sw_prev.to(device), q_liq.to(device), qo.to(device)
                corey, mu_o, mu_w = corey.to(device), mu_o.to(device), mu_w.to(device)
                optimizer.zero_grad()
                dp_hat, ds_hat = predict_delta(model, xb, norm, device)
                qo_hat, _ = rates_from_delta(sw_prev, ds_hat, q_liq, corey, mu_o, mu_w, prod_i, prod_j)
                loss, _, _, _ = proxy_loss(dp_hat, dp, ds_hat, ds, qo_hat, qo)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item() * xb.size(0)
                seen += xb.size(0)
            tr_loss /= seen
            scheduler.step()

            val = evaluate_split(model, val_runs, norm, prod_idx, device)["one_step"]
            history["tr"].append(tr_loss)
            history["val_rel_qo"].append(val["rel_l2_qo"])
            history["val_rel_qw"].append(val["rel_l2_qw"])
            history["val_mae_qo"].append(val["mae_qo"])
            history["val_mae_qw"].append(val["mae_qw"])
            history["val_mae_p"].append(val["mae_p_bar"])
            history["val_mae_sw"].append(val["mae_sw"])
            history["val_cum_oil_rel"].append(val["cum_oil_rel"])
            history["val_rel_dp"].append(val["rel_l2_dp"])
            history["val_rel_ds"].append(val["rel_l2_ds"])

            if val["rel_l2_qo"] < best:
                best = val["rel_l2_qo"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "norm": norm,
                        "arch": args.arch,
                        "in_ch": IN_CH,
                        "prod_idx": prod_idx.tolist(),
                        "epoch": ep,
                        "val_rel_qo": best,
                    },
                    model_path,
                )
            if ep % 10 == 0 or ep == args.epochs - 1:
                rolled = evaluate_split(model, val_runs, norm, prod_idx, device, with_rollout=True)["rollout"]
                full_h = str(val_runs[0]["dp"].shape[2])
                roll = rolled[full_h]
                ep_sec = time.time() - ep_t
                print(
                    f"ep {ep:3d}/{args.epochs} | train {tr_loss:.4f} | "
                    f"val qo rel {val['rel_l2_qo']:.4f} MAE {val['mae_qo']:.2f} | "
                    f"qw rel {val['rel_l2_qw']:.4f} MAE {val['mae_qw']:.2f} | "
                    f"cum oil rel {val['cum_oil_rel']:.4f} | "
                    f"MAE_p {val['mae_p_bar']:.2f} бар MAE_Sw {val['mae_sw']:.4f} | "
                    f"rel Δp {val['rel_l2_dp']:.4f} rel ΔS {val['rel_l2_ds']:.4f} | "
                    f"rollout {full_h} qo rel {roll['rel_l2_qo']:.4f} cum {roll['cum_oil_rel']:.4f} | "
                    f"r2_qo {val['diagnostic_r2_qo']:.3f} диагн. | {ep_sec:.0f}s/ep",
                    flush=True,
                )
                (out_dir / "history.json").write_text(json.dumps(history))
                save_curves(history, out_dir / "curves.png")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        best = float(ckpt["val_rel_qo"])

    model.eval()
    report = {}
    for name, group in (("val", val_runs), ("test", test_runs)):
        report[name] = evaluate_split(model, group, norm, prod_idx, device, with_rollout=True)
    qo_true, qo_hat = collect_qo_scatter(model, test_runs, norm, prod_idx, device)
    report["test"]["scatter_r2_qo"] = save_scatter(qo_true, qo_hat, out_dir / "qo_scatter_test.png")
    summary = {
        "arch": args.arch,
        "epochs": args.epochs,
        "best_epoch": int(ckpt["epoch"]),
        "selection": "val one-step relative L2 of qo",
        "best_val_rel_qo": best,
        "holdout": report,
        "device": str(device),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "best_epoch": summary["best_epoch"],
        "best_val_rel_qo": best,
        "test_one_step": report["test"]["one_step"],
        "test_rollout": report["test"]["rollout"],
        "test_rollout_runs": report["test"]["rollout_runs"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
