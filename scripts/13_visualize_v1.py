#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第十三步：DragonFlow-KronosGraph V1 流水线产物可视化。

读 data/processed/ 下的 V1 输出，输出 PNG 到 data/processed/viz_v1/，
并生成一张 index.html 便于在浏览器集中查看。

    uv run python scripts/13_visualize_v1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dragonflow.utils.io import resolve_path
from dragonflow.utils.logger import get_logger
from dragonflow.viz.v1.backtest import plot_nav_drawdown, plot_position_count
from dragonflow.viz.v1.kline import plot_kline_auxiliary_predictions, plot_kline_emb_pca
from dragonflow.viz.v1.predictions import (
    plot_pred_dispersion,
    plot_pred_vs_actual,
    plot_quantile_calibration,
    plot_quintile_returns,
    plot_rank_ic,
)
from dragonflow.viz.v1.spectral import (
    plot_cluster_size_over_time,
    plot_cluster_transition_heat,
    plot_spectral_pca,
)

logger = get_logger(__name__)

PROCESSED = resolve_path("data/processed")
OUT_DIR = PROCESSED / "viz_v1"


def _safe(label: str, fn, *args, **kwargs):
    try:
        out = fn(*args, **kwargs)
        logger.info("[OK] {} -> {}", label, out)
        return out, None
    except Exception as exc:  # noqa: BLE001
        logger.exception("[FAIL] {}: {}", label, exc)
        return None, str(exc)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nav_path = PROCESSED / "backtest_nav.parquet"
    pred_path = PROCESSED / "tft_predictions.parquet"
    panel_path = PROCESSED / "model_panel_tft.parquet"
    spec_path = PROCESSED / "spectral_embeddings.parquet"
    kline_path = PROCESSED / "kline_embeddings.parquet"

    jobs = [
        ("01_nav_drawdown", plot_nav_drawdown, (nav_path, OUT_DIR / "01_nav_drawdown.svg")),
        ("02_position_turnover", plot_position_count, (nav_path, OUT_DIR / "02_position_turnover.svg")),
        ("03_rank_ic", plot_rank_ic, (pred_path, panel_path, OUT_DIR / "03_rank_ic.svg")),
        ("04_quintile_returns", plot_quintile_returns, (pred_path, panel_path, OUT_DIR / "04_quintile_returns.svg")),
        ("05_pred_vs_actual", plot_pred_vs_actual, (pred_path, panel_path, OUT_DIR / "05_pred_vs_actual.svg")),
        ("06_quantile_calibration", plot_quantile_calibration, (pred_path, panel_path, OUT_DIR / "06_quantile_calibration.svg")),
        ("07_pred_dispersion", plot_pred_dispersion, (pred_path, OUT_DIR / "07_pred_dispersion.svg")),
        ("08_spectral_pca", plot_spectral_pca, (spec_path, panel_path, OUT_DIR / "08_spectral_pca.svg")),
        ("09_cluster_size_over_time", plot_cluster_size_over_time, (spec_path, OUT_DIR / "09_cluster_size_over_time.svg")),
        ("10_cluster_transition", plot_cluster_transition_heat, (spec_path, OUT_DIR / "10_cluster_transition.svg")),
        ("11_kline_emb_pca", plot_kline_emb_pca, (kline_path, panel_path, OUT_DIR / "11_kline_emb_pca.svg")),
        ("12_kline_aux_pred", plot_kline_auxiliary_predictions, (kline_path, panel_path, OUT_DIR / "12_kline_aux_pred.svg")),
    ]

    results = []
    for label, fn, args in jobs:
        path, err = _safe(label, fn, *args)
        results.append((label, path, err))

    _write_index(results, OUT_DIR / "index.html")

    print("\n" + "=" * 60)
    print(f"V1 可视化完成：{sum(1 for _, p, _ in results if p is not None)}/{len(results)} 张")
    print(f"输出目录：{OUT_DIR}")
    print(f"汇总：{OUT_DIR / 'index.html'}")
    print("=" * 60)


def _write_index(results, out_path: Path) -> None:
    titles = {
        "01_nav_drawdown": "净值曲线 + 水下回撤",
        "02_position_turnover": "持仓数量 & 单边换手率",
        "03_rank_ic": "日度 Rank IC 与累计",
        "04_quintile_returns": "5 分位组合超额累积 + 多空价差",
        "05_pred_vs_actual": "q50 预测 vs 真实超额（hexbin）",
        "06_quantile_calibration": "分位数校准 + 80% 区间覆盖率",
        "07_pred_dispersion": "预测横截面分布 + 模型不确定带",
        "08_spectral_pca": "谱嵌入 PCA · cluster / industry 上色",
        "09_cluster_size_over_time": "簇规模随 refit 时间堆叠",
        "10_cluster_transition": "首尾 refit 簇成员变迁矩阵",
        "11_kline_emb_pca": "K 线嵌入 PCA · 行业上色",
        "12_kline_aux_pred": "K 线辅助预测诊断",
    }
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>DragonFlow-KronosGraph V1 可视化</title>",
        "<style>",
        "body{background:#1a1a2e;color:#e0e0e0;font-family:'Microsoft YaHei','PingFang SC',sans-serif;margin:24px;}",
        "h1{color:#ffd700;border-bottom:1px solid #2a2a4a;padding-bottom:8px;}",
        ".card{background:#16213e;border:1px solid #2a2a4a;border-radius:8px;padding:14px;margin:14px 0;}",
        ".card h2{margin:0 0 10px 0;color:#4fc3f7;font-size:16px;}",
        ".card img{width:100%;border-radius:4px;}",
        ".err{color:#ff4444;}",
        "</style></head><body>",
        "<h1>DragonFlow-KronosGraph V1 流水线产物可视化</h1>",
        "<p style='color:#8892b0;'>由 <code>scripts/13_visualize_v1.py</code> 生成。",
        "底层模块在 <code>src/dragonflow/viz/v1/</code>，与 <code>dragonflow/viz/</code> 的 features 图隔离。</p>",
    ]
    for label, path, err in results:
        title = titles.get(label, label)
        parts.append("<div class='card'>")
        parts.append(f"<h2>{label} · {title}</h2>")
        if path is not None:
            parts.append(f"<img src='{Path(path).name}'>")
        else:
            parts.append(f"<div class='err'>生成失败：{err}</div>")
        parts.append("</div>")
    parts.append("</body></html>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
