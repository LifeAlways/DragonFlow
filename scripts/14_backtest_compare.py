#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第十四步：V1 回测的过滤器/打分公式消融实验。

复用同一份 TFT 预测，对 4 个变体跑同样的执行模拟，输出 NAV / metrics 表，
并画一张 NAV 叠加图。无需重训模型。

    uv run python scripts/14_backtest_compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd

from dragonflow.backtest.execution import run_simple_backtest
from dragonflow.backtest.portfolio import build_rebalance_weights
from dragonflow.modeling.config import load_model_config
from dragonflow.utils.io import resolve_path, save_json, save_parquet
from dragonflow.utils.logger import get_logger
from dragonflow.viz.v1.compare import plot_nav_comparison, plot_metrics_table

logger = get_logger(__name__)

OUT_DIR = resolve_path("data/processed/viz_v1")


def _make_variants(base_cfg: dict) -> dict[str, dict]:
    """4 变体：baseline / no_q10 / q50_only / both."""
    bt = dict(base_cfg["backtest"])  # 起点
    variants = {
        "baseline":  {**bt},
        "no_q10":    {**bt, "q10_floor": -1.0},
        "q50_only":  {**bt, "score_formula": "q50"},
        "both":      {**bt, "q10_floor": -1.0, "score_formula": "q50"},
    }
    return variants


def main() -> None:
    cfg = load_model_config("configs/model_v1.yaml")
    paths = cfg["paths"]
    panel = pd.read_parquet(resolve_path(paths["tft_panel"]))
    pred = pd.read_parquet(resolve_path(paths["predictions"]))

    variants = _make_variants(cfg)
    nav_dfs: dict[str, pd.DataFrame] = {}
    all_metrics: dict[str, dict] = {}
    for name, bt_cfg in variants.items():
        logger.info("跑变体: {} | q10_floor={} score_formula={}",
                    name, bt_cfg.get("q10_floor"), bt_cfg.get("score_formula", "q50_over_uncertainty"))
        weights = build_rebalance_weights(pred, panel, bt_cfg)
        nav, _pos, metrics = run_simple_backtest(weights, panel, bt_cfg)
        nav_dfs[name] = nav
        all_metrics[name] = metrics
        save_parquet(nav, OUT_DIR / f"compare_nav_{name}.parquet")
        logger.info("  -> total_return={:+.2%} sharpe={:+.2f} max_dd={:+.2%} N={}",
                    metrics.get("total_return", 0), metrics.get("sharpe", 0),
                    metrics.get("max_drawdown", 0), metrics.get("n_days", 0))

    save_json(all_metrics, OUT_DIR / "compare_metrics.json")

    plot_nav_comparison(nav_dfs, OUT_DIR / "13_compare_nav.svg")
    plot_metrics_table(all_metrics, OUT_DIR / "14_compare_metrics.svg")

    print("\n" + "=" * 60)
    print("4 变体回测完成")
    for name, m in all_metrics.items():
        print(f"  {name:<12s} total={m.get('total_return',0):+8.2%}  "
              f"sharpe={m.get('sharpe',0):+6.2f}  mdd={m.get('max_drawdown',0):+7.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
