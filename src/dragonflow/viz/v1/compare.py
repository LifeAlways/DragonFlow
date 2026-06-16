"""消融实验对比图：4 变体 NAV 叠加 + 指标表。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from dragonflow.viz.theme import COLORS, apply_dark_theme


_VARIANT_COLOR = {
    "baseline":  COLORS["up"],       # 红：当前
    "no_q10":    COLORS["gold"],     # 金：去 q10
    "q50_only":  COLORS["blue"],     # 蓝：换 score
    "both":      COLORS["down"],     # 绿：两者都改
}
_VARIANT_LABEL = {
    "baseline":  "baseline · q10>-0.03 + q50/uncertainty",
    "no_q10":    "去掉 q10 过滤",
    "q50_only":  "score 改用 q50 直选",
    "both":      "两改同时",
}


def plot_nav_comparison(nav_dfs: dict[str, pd.DataFrame], out_path: Path) -> Path:
    apply_dark_theme()
    fig, (ax_nav, ax_dd) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]},
    )

    for name, nav in nav_dfs.items():
        if nav.empty:
            continue
        nav = nav.sort_values("date").copy()
        nav["date"] = pd.to_datetime(nav["date"])
        color = _VARIANT_COLOR.get(name, COLORS["text_sub"])
        ax_nav.plot(nav["date"], nav["nav"], color=color, lw=2.0,
                    label=f"{_VARIANT_LABEL.get(name, name)} (终值 {nav['nav'].iloc[-1]-1:+.2%})")
        dd = nav["nav"] / nav["nav"].cummax() - 1.0
        ax_dd.plot(nav["date"], dd, color=color, lw=1.4)
        ax_dd.fill_between(nav["date"], dd, 0, color=color, alpha=0.15)

    ax_nav.axhline(1.0, color=COLORS["text_sub"], lw=0.8, ls="--", alpha=0.6)
    ax_nav.set_ylabel("NAV", color=COLORS["text"])
    ax_nav.set_title("V1 回测消融对比 · 同一份预测，不同过滤/打分组合",
                     color=COLORS["text"], fontsize=13, pad=12)
    ax_nav.legend(loc="best", fontsize=9)

    ax_dd.set_ylabel("回撤", color=COLORS["text"])
    ax_dd.set_xlabel("日期", color=COLORS["text"])
    ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_dd.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=COLORS["bg"])
    plt.close(fig)
    return out_path


def plot_metrics_table(metrics: dict[str, dict], out_path: Path) -> Path:
    apply_dark_theme()
    keys = ["total_return", "annual_return", "annual_volatility", "sharpe", "max_drawdown"]
    labels = ["总收益", "年化收益", "年化波动", "Sharpe", "最大回撤"]
    fmt = ["{:+.2%}", "{:+.2%}", "{:.2%}", "{:+.2f}", "{:+.2%}"]

    variants = list(metrics.keys())
    cells = []
    for k, f in zip(keys, fmt):
        row = []
        for v in variants:
            val = metrics[v].get(k, 0.0) if metrics[v] else 0.0
            row.append(f.format(val))
        cells.append(row)

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.axis("off")
    table = ax.table(
        cellText=cells, rowLabels=labels,
        colLabels=[_VARIANT_LABEL.get(v, v).split(" · ")[0] for v in variants],
        loc="center", cellLoc="center", colWidths=[0.22] * len(variants),
    )
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.8)

    # 为表格着色（只迭代实际存在的 cell）
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["grid"])
        is_header = (i == 0) or (j == -1)
        if is_header:
            cell.set_facecolor(COLORS["panel"])
            cell.set_text_props(color=COLORS["gold"], weight="bold")
        else:
            cell.set_facecolor(COLORS["bg"])
            cell.set_text_props(color=COLORS["text"])

    ax.set_title("V1 回测消融对比 · 指标表", color=COLORS["text"], fontsize=13, pad=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=COLORS["bg"])
    plt.close(fig)
    return out_path
