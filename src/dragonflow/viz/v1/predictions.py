"""预测质量诊断：Rank IC / 分位组合 / 散点 / 校准。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dragonflow.viz.theme import COLORS, apply_dark_theme


_PRED_COL = "pred_q50_excess_ret_fwd_5d"
_Q10_COL = "pred_q10_excess_ret_fwd_5d"
_Q90_COL = "pred_q90_excess_ret_fwd_5d"
_TARGET_COL = "excess_ret_fwd_5d"


def _join_predictions_with_actual(pred_path: Path, panel_path: Path) -> pd.DataFrame:
    pred = pd.read_parquet(pred_path)
    panel = pd.read_parquet(panel_path, columns=["date", "stock_code", _TARGET_COL])
    pred["date"] = pd.to_datetime(pred["date"])
    panel["date"] = pd.to_datetime(panel["date"])
    df = pred.merge(panel, on=["date", "stock_code"], how="inner")
    df = df.dropna(subset=[_PRED_COL, _TARGET_COL])
    return df


def plot_rank_ic(pred_path: Path, panel_path: Path, out_path: Path) -> Path:
    apply_dark_theme()
    df = _join_predictions_with_actual(pred_path, panel_path)

    def _ic(sub: pd.DataFrame) -> float:
        if len(sub) < 5:
            return np.nan
        return float(spearmanr(sub[_PRED_COL], sub[_TARGET_COL])[0])

    daily_ic = df.groupby("date").apply(_ic).dropna().sort_index()
    cum_ic = daily_ic.cumsum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    bar_colors = [COLORS["down"] if v >= 0 else COLORS["up"] for v in daily_ic.values]
    ax1.bar(daily_ic.index, daily_ic.values, color=bar_colors, alpha=0.85)
    ax1.axhline(0, color=COLORS["text_sub"], lw=0.8, ls="--", alpha=0.6)
    ax1.axhline(daily_ic.mean(), color=COLORS["gold"], lw=1.2, ls=":",
                label=f"均值 {daily_ic.mean():+.4f}")
    ax1.set_ylabel("Daily Rank IC", color=COLORS["text"])
    ic_ir = daily_ic.mean() / (daily_ic.std() + 1e-12)
    ax1.set_title(
        f"V1 日度 Rank IC（IC_mean={daily_ic.mean():+.4f}  ICIR={ic_ir:+.3f}  N={len(daily_ic)}）",
        color=COLORS["text"], fontsize=13, pad=10,
    )
    ax1.legend(loc="upper right")

    ax2.plot(cum_ic.index, cum_ic.values, color=COLORS["blue"], lw=2.0)
    ax2.axhline(0, color=COLORS["text_sub"], lw=0.8, ls="--", alpha=0.6)
    ax2.set_ylabel("累计 Rank IC", color=COLORS["text"])
    ax2.set_xlabel("日期", color=COLORS["text"])
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=COLORS["bg"])
    plt.close(fig)
    return out_path


def plot_quintile_returns(pred_path: Path, panel_path: Path, out_path: Path,
                          n_quantiles: int = 5) -> Path:
    apply_dark_theme()
    df = _join_predictions_with_actual(pred_path, panel_path)

    df = df.copy()
    df["q_group"] = df.groupby("date")[_PRED_COL].transform(
        lambda s: pd.qcut(s.rank(method="first"), q=n_quantiles,
                          labels=False, duplicates="drop")
    )
    df = df.dropna(subset=["q_group"])
    df["q_group"] = df["q_group"].astype(int)

    daily_ret = (
        df.groupby(["date", "q_group"])[_TARGET_COL].mean().unstack("q_group")
    ).sort_index()
    cum = (1.0 + daily_ret).cumprod() - 1.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    palette = [COLORS["up"], COLORS["orange"], COLORS["gold"], COLORS["cyan"], COLORS["down"]]
    for q in cum.columns:
        color = palette[q % len(palette)]
        label = f"Q{q+1}（{'最低' if q == 0 else '最高' if q == cum.columns.max() else '中间'}）"
        ax1.plot(cum.index, cum[q], color=color, lw=1.8, label=label)
    ax1.axhline(0, color=COLORS["text_sub"], lw=0.8, ls="--", alpha=0.6)
    ax1.set_ylabel("累计超额收益", color=COLORS["text"])
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax1.set_title(f"V1 预测分 {n_quantiles} 组超额累积收益（按 q50 排序）",
                  color=COLORS["text"], fontsize=13, pad=10)
    ax1.legend(loc="best", ncol=n_quantiles)

    top, bot = cum.columns.max(), cum.columns.min()
    spread = cum[top] - cum[bot]
    ax2.plot(spread.index, spread.values, color=COLORS["gold"], lw=2.0,
             label=f"Top - Bottom（终值 {spread.iloc[-1]:+.2%}）")
    ax2.fill_between(spread.index, spread.values, 0,
                     where=spread.values >= 0, color=COLORS["down"], alpha=0.25)
    ax2.fill_between(spread.index, spread.values, 0,
                     where=spread.values < 0, color=COLORS["up"], alpha=0.25)
    ax2.axhline(0, color=COLORS["text_sub"], lw=0.8, ls="--", alpha=0.6)
    ax2.set_ylabel("多空价差", color=COLORS["text"])
    ax2.set_xlabel("日期", color=COLORS["text"])
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.legend(loc="best")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=COLORS["bg"])
    plt.close(fig)
    return out_path


def plot_pred_vs_actual(pred_path: Path, panel_path: Path, out_path: Path) -> Path:
    apply_dark_theme()
    df = _join_predictions_with_actual(pred_path, panel_path)

    x = df[_PRED_COL].values
    y = df[_TARGET_COL].values
    p_lo, p_hi = np.quantile(x, [0.005, 0.995])
    a_lo, a_hi = np.quantile(y, [0.005, 0.995])
    mask = (x >= p_lo) & (x <= p_hi) & (y >= a_lo) & (y <= a_hi)
    x, y = x[mask], y[mask]

    fig, ax = plt.subplots(figsize=(8.5, 8))
    hb = ax.hexbin(x, y, gridsize=45, mincnt=1, cmap="cool")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("样本数", color=COLORS["text"])
    cb.ax.yaxis.set_tick_params(color=COLORS["text"])
    plt.setp(cb.ax.get_yticklabels(), color=COLORS["text"])

    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], color=COLORS["text_sub"], ls="--", lw=1.0,
            label="y = x")
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color=COLORS["gold"], lw=2.0,
            label=f"OLS: y = {slope:+.2f}x {intercept:+.4f}")

    pearson = float(np.corrcoef(x, y)[0, 1])
    ax.axhline(0, color=COLORS["text_sub"], lw=0.6, alpha=0.4)
    ax.axvline(0, color=COLORS["text_sub"], lw=0.6, alpha=0.4)
    ax.set_xlabel(f"预测 q50（{_PRED_COL}）", color=COLORS["text"])
    ax.set_ylabel(f"真实超额收益（{_TARGET_COL}）", color=COLORS["text"])
    ax.set_title(f"V1 预测 vs 真实超额收益（Pearson={pearson:+.4f}  N={len(x)}）",
                 color=COLORS["text"], fontsize=12, pad=10)
    ax.legend(loc="best")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=COLORS["bg"])
    plt.close(fig)
    return out_path


def plot_quantile_calibration(pred_path: Path, panel_path: Path, out_path: Path) -> Path:
    """检验 q10-q90 区间是否真的覆盖 80% 真实样本，分位数是否对齐。"""
    apply_dark_theme()
    pred = pd.read_parquet(pred_path)
    panel = pd.read_parquet(panel_path, columns=["date", "stock_code", _TARGET_COL])
    pred["date"] = pd.to_datetime(pred["date"])
    panel["date"] = pd.to_datetime(panel["date"])
    df = pred.merge(panel, on=["date", "stock_code"], how="inner").dropna(
        subset=[_Q10_COL, _PRED_COL, _Q90_COL, _TARGET_COL]
    )

    # 覆盖率：按日计算 actual 落在 [q10,q90] 的比例（理论值 0.80）
    df["covered"] = ((df[_TARGET_COL] >= df[_Q10_COL]) &
                     (df[_TARGET_COL] <= df[_Q90_COL])).astype(float)
    daily_cov = df.groupby("date")["covered"].mean().sort_index()

    # 三个名义分位数的实际频率
    nominal = [0.10, 0.50, 0.90]
    cols = [_Q10_COL, _PRED_COL, _Q90_COL]
    empirical = [float((df[_TARGET_COL] <= df[c]).mean()) for c in cols]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.plot([0, 1], [0, 1], color=COLORS["text_sub"], ls="--", lw=1.0,
             label="理想校准 y = x")
    ax1.scatter(nominal, empirical, s=180, c=[COLORS["blue"], COLORS["gold"], COLORS["down"]],
                edgecolors=COLORS["white"], zorder=5)
    for nom, emp, name in zip(nominal, empirical, ["q10", "q50", "q90"]):
        ax1.annotate(f"{name}\n{emp:.3f}", xy=(nom, emp), xytext=(8, 8),
                     textcoords="offset points", color=COLORS["text"], fontsize=10)
    ax1.set_xlim(-0.02, 1.02); ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlabel("名义分位 nominal", color=COLORS["text"])
    ax1.set_ylabel("经验频率 P(y ≤ q)", color=COLORS["text"])
    ax1.set_title("分位数校准点图", color=COLORS["text"], fontsize=12, pad=10)
    ax1.legend(loc="best")

    ax2.plot(daily_cov.index, daily_cov.values, color=COLORS["cyan"], lw=2.0,
             marker="o", ms=4, label="日度 80% 区间覆盖率")
    ax2.axhline(0.8, color=COLORS["gold"], lw=1.2, ls="--",
                label="理论值 0.80")
    ax2.axhline(daily_cov.mean(), color=COLORS["pink"], lw=1.2, ls=":",
                label=f"实测均值 {daily_cov.mean():.3f}")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("日期", color=COLORS["text"])
    ax2.set_ylabel("覆盖率", color=COLORS["text"])
    ax2.set_title("[q10, q90] 区间覆盖率随时间", color=COLORS["text"], fontsize=12, pad=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.legend(loc="lower right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=COLORS["bg"])
    plt.close(fig)
    return out_path


def plot_pred_dispersion(pred_path: Path, out_path: Path) -> Path:
    """每天预测的跨股票横截面分布带：25/50/75 + min/max + q10/q90 中位水平。"""
    apply_dark_theme()
    pred = pd.read_parquet(pred_path)
    pred["date"] = pd.to_datetime(pred["date"])

    summary = (
        pred.groupby("date")
        .agg(
            q50_p25=(_PRED_COL, lambda s: s.quantile(0.25)),
            q50_p50=(_PRED_COL, "median"),
            q50_p75=(_PRED_COL, lambda s: s.quantile(0.75)),
            q10_p50=(_Q10_COL, "median"),
            q90_p50=(_Q90_COL, "median"),
        )
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.fill_between(summary.index, summary["q90_p50"], summary["q10_p50"],
                    color=COLORS["blue"], alpha=0.15, label="模型不确定区间（中位 q10–q90）")
    ax.fill_between(summary.index, summary["q50_p75"], summary["q50_p25"],
                    color=COLORS["gold"], alpha=0.35,
                    label="股票截面 IQR（q50 的 25–75 分位）")
    ax.plot(summary.index, summary["q50_p50"], color=COLORS["white"], lw=2.0,
            label="截面 q50 中位数")
    ax.axhline(0, color=COLORS["text_sub"], lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("日期", color=COLORS["text"])
    ax.set_ylabel("预测超额收益", color=COLORS["text"])
    ax.set_title("V1 预测的横截面分布与模型不确定性带",
                 color=COLORS["text"], fontsize=13, pad=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.2%}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.legend(loc="best")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=COLORS["bg"])
    plt.close(fig)
    return out_path
