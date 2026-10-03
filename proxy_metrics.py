"""Потери и метрики по metrics.md.

Учёба: relative L2 по Δp и ΔSw плюс relative L2 по q_o.
q_o не выход сети: дробный поток от Sw в добывающей ячейке и заданного q_liq.
Чекпоинт — по q_o на hold-out прогонах.
R^2 считается только как диагностика scatter, не как критерий отбора.
"""

import numpy as np
import torch

W_DP = 1.0
W_DS = 1.0
W_QO = 1.0
# Горизонты rollout из metrics.md. Короткий прогон получает свою полную длину.
ROLLOUT_HORIZONS = (12, 36, 60)


def rel_lp(pred, target, p=2, eps=1e-8):
    """Среднее по батчу ||pred-target||_p / ||target||_p. Первая ось — батч."""
    if pred.shape != target.shape:
        raise ValueError(f"формы {tuple(pred.shape)} и {tuple(target.shape)}")
    if pred.ndim < 2:
        raise ValueError("нужна ось батча")
    diff = (pred - target).flatten(1)
    base = target.flatten(1)
    num = torch.linalg.vector_norm(diff, ord=p, dim=1)
    den = torch.linalg.vector_norm(base, ord=p, dim=1).clamp_min(eps)
    return (num / den).mean()


def corey_split(sw, q_liq, kro, krw, no, nw, swc, sor, mu_o, mu_w):
    """Нефть и вода, м3/сут. Та же формула, что corey_split в generate_data_v2.m."""
    sn = (sw - swc) / (1.0 - swc - sor)
    sn = sn.clamp(0.0, 1.0)
    kr_w = krw * sn ** nw
    kr_o = kro * (1.0 - sn) ** no
    lw = kr_w / mu_w
    lo = kr_o / mu_o
    water_cut = lw / (lw + lo + 1e-12)
    return (1.0 - water_cut) * q_liq, water_cut * q_liq


def proxy_loss(dp_hat, dp, ds_hat, ds, qo_hat, qo, w_dp=W_DP, w_ds=W_DS, w_qo=W_QO):
    """Сумма трёх relative L2. Все тензоры в физических единицах, с осью батча."""
    rel_dp = rel_lp(dp_hat, dp, p=2)
    rel_ds = rel_lp(ds_hat, ds, p=2)
    rel_qo = rel_lp(qo_hat, qo, p=2)
    loss = w_dp * rel_dp + w_ds * rel_ds + w_qo * rel_qo
    return loss, rel_dp, rel_ds, rel_qo


def rel_lp_np(pred, target, p=2, eps=1e-8):
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    den = np.linalg.norm(target, ord=p)
    return float(np.linalg.norm(pred - target, ord=p) / max(den, eps))


def r2_diagnostic(pred, target):
    """R^2 по всем элементам. Не использовать для early stopping и чекпоинта."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    ss_res = np.sum((target - pred) ** 2)
    center = target - target.mean()
    ss_tot = np.sum(center ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def cumulative_volume(rate_m3day, dt_seconds):
    """Сумма по скважинам и шагам. rate (n_wells, n_steps), dt в секундах."""
    rate = np.asarray(rate_m3day, dtype=np.float64)
    dt = np.asarray(dt_seconds, dtype=np.float64).reshape(1, -1)
    return float(np.sum(rate * dt / 86400.0))


def score_forecast(qo_hat, qo, qw_hat, qw, q_liq, dt, p_hat, p, sw_hat, sw, dp_hat=None, dp=None, ds_hat=None, ds=None):
    """Метрики одного прогона.

    Дебиты: (n_prod, n_steps), м3/сут. Поля: (n_steps, nx, ny).
    dt: (n_steps,) секунд. Δp/ΔSw необязательны — их relative L2 есть у одношагового прогноза.
    """
    qo_hat = np.asarray(qo_hat, dtype=np.float64)
    qo = np.asarray(qo, dtype=np.float64)
    qw_hat = np.asarray(qw_hat, dtype=np.float64)
    qw = np.asarray(qw, dtype=np.float64)
    q_liq = np.asarray(q_liq, dtype=np.float64)
    oil_hat = cumulative_volume(qo_hat, dt)
    oil = cumulative_volume(qo, dt)
    wc_hat = qw_hat / np.maximum(q_liq, 1e-12)
    wc = qw / np.maximum(q_liq, 1e-12)
    out = {
        "mae_qo": float(np.mean(np.abs(qo_hat - qo))),
        "rel_l2_qo": rel_lp_np(qo_hat, qo, 2),
        "rel_l1_qo": rel_lp_np(qo_hat, qo, 1),
        "mae_qw": float(np.mean(np.abs(qw_hat - qw))),
        "rel_l2_qw": rel_lp_np(qw_hat, qw, 2),
        "rel_l1_qw": rel_lp_np(qw_hat, qw, 1),
        "mae_wc": float(np.mean(np.abs(wc_hat - wc))),
        "cum_oil_mae_m3": abs(oil_hat - oil),
        "cum_oil_rel": abs(oil_hat - oil) / max(abs(oil), 1e-12),
        "cum_oil_m3": oil,
        "cum_oil_hat_m3": oil_hat,
        "mae_p_bar": float(np.mean(np.abs(np.asarray(p_hat) - np.asarray(p)))),
        "mae_sw": float(np.mean(np.abs(np.asarray(sw_hat) - np.asarray(sw)))),
        "diagnostic_r2_qo": r2_diagnostic(qo_hat, qo),
        "diagnostic_r2_p": r2_diagnostic(p_hat, p),
        "diagnostic_r2_sw": r2_diagnostic(sw_hat, sw),
    }
    if dp_hat is not None:
        out["rel_l2_dp"] = rel_lp_np(dp_hat, dp, 2)
        out["rel_l1_dp"] = rel_lp_np(dp_hat, dp, 1)
        out["rel_l2_ds"] = rel_lp_np(ds_hat, ds, 2)
        out["rel_l1_ds"] = rel_lp_np(ds_hat, ds, 1)
    return out


def mean_scores(rows):
    keys = [k for k in rows[0] if k != "run_id" and np.ndim(rows[0][k]) == 0]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def rollout_horizons(n_steps):
    """12 / 36 / 60, если они короче траектории, и всегда полный горизонт."""
    found = [h for h in ROLLOUT_HORIZONS if h <= n_steps]
    if n_steps not in found:
        found.append(n_steps)
    return found
