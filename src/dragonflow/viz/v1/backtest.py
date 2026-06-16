"""回测：NAV 曲线 + 水下回撤图。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from dragonflow.viz.theme import COLORS, apply_dark_theme


def plot_nav_drawdown(nav_path: Path, out_path: Path) -> Path:
    apply_dark_theme()
    nav = pd.read_parquet(nav_path).sort_values("date").reset_index(drop=True)
    nav["date"] = pd.to_datetime(nav["date"])

    running_peak = nav["nav"].cummax()
    drawdown = nav["nav"] / running_peak - 1.0
    max_dd_idx = drawdown.idxmin()

    fig, (ax_nav, ax_dd) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]},
    )

    ax_nav.plot(nav["date"], nav["nav"], color=COLORS["blue"], lw=2.0, label="策略净值")
    ax_nav.axhline(1.0, color=COLORS["text_sub"], lw=0.8, ls="--", alpha=0.6)
    active = nav[nav["n_positions"] > 0]
    if not active.empty:
        ax_nav.axvspan(active["date"].min(), active["date"].max(),
                       color=COLORS["gold"], alpha=0.08, label="持仓区间")
    ax_nav.scatter([nav["date"].iloc[max_dd_idx]], [nav["nav"].iloc[max_dd_idx]],
                   color=COLORS["up"], s=60, zorder=5, label="最大回撤点")
    ax_nav.set_ylabel("NAV", color=COLORS["text"])
    ax_nav.set_title("DragonFlow-KronosGraph V1 回测净值与回撤", color=COLORS["text"],
                     fontsize=13, pad=12)
    final_ret = nav["nav"].iloc[-1] - 1.0
    ax_nav.text(
        0.02, 0.05,
        f"总收益 {final_ret:+.2%}\n最大回撤 {drawdown.min():+.2%}\n样本天数 {len(nav)}",
        transform=ax_nav.transAxes, va="bottom", ha="left",
        color=COLORS["text"], fontsize=10,
        bbox={"facecolor": COLORS["panel"], "edgecolor": COLORS["grid"], "alpha": 0.85},
    )
    ax_nav.legend(loc="upper right")

    ax_dd.fill_between(nav["date"], drawdown, 0, color=COLORS["up"], alpha=0.45)
    ax_dd.plot(nav["date"], drawdown, color=COLORS["up"], lw=1.2)
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


def plot_position_count(nav_path: Path, out_path: Path) -> Path:
    apply_dark_theme()
    nav = pd.read_parquet(nav_path).sort_values("date").reset_index(drop=True)
    nav["date"] = pd.to_datetime(nav["date"])

    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    ax1.bar(nav["date"], nav["n_positions"], color=COLORS["blue"], alpha=0.7,
            label="持仓股票数")
    ax1.set_ylabel("持仓数量", color=COLORS["blue"])
    ax1.tick_params(axis="y", colors=COLORS["blue"])

    ax2 = ax1.twinx()
    ax2.plot(nav["date"], nav["turnover"], color=COLORS["gold"], lw=1.6,
             marker="o", ms=3, label="单边换手率")
    ax2.set_ylabel("换手率", color=COLORS["gold"])
    ax2.tick_params(axis="y", colors=COLORS["gold"])
    ax2.grid(False)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    ax1.set_xlabel("日期", color=COLORS["text"])
    ax1.set_title("V1 持仓数量 & 单边换手率", color=COLORS["text"], fontsize=13, pad=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=COLORS["bg"])
    plt.close(fig)
    return out_path
