"""Prediction-to-portfolio conversion."""
from __future__ import annotations

import pandas as pd


def build_rebalance_weights(pred: pd.DataFrame, panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    p = pred.copy()
    formula = str(cfg.get("score_formula", "q50_over_uncertainty"))
    if formula == "q50":
        p["score"] = p["pred_q50_excess_ret_fwd_5d"]
    elif formula == "q50_over_uncertainty":
        p["score"] = p["pred_q50_excess_ret_fwd_5d"] / ((p["pred_q90_excess_ret_fwd_5d"] - p["pred_q10_excess_ret_fwd_5d"]).abs() + 1e-6)
    else:
        raise ValueError(f"unknown score_formula: {formula!r}")
    latest = panel[["date","stock_code","amount_mean_20d","close"]].copy()
    latest["date"] = pd.to_datetime(latest["date"])
    p["date"] = pd.to_datetime(p["date"])
    p = p.merge(latest, on=["date","stock_code"], how="left")
    p = p[p["amount_mean_20d"] >= float(cfg.get("min_amount_mean_20d", 0))]
    p = p[p["pred_q10_excess_ret_fwd_5d"] > float(cfg.get("q10_floor", -999))]
    dates = sorted(p["date"].unique())
    rebalance_every = int(cfg.get("rebalance_every_n_days", 5))
    rebalance_dates = dates[::rebalance_every]
    rows=[]
    for d in rebalance_dates:
        block = p[p["date"] == d].nlargest(int(cfg.get("top_n", 80)), "score")
        if block.empty: continue
        w = 1.0 / len(block)
        out = block[["date","stock_code","score","close"]].copy(); out["target_weight"] = w
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
