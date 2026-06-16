"""K 线编码可视化：嵌入 PCA + 辅助预测 vs 真实诊断。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from dragonflow.viz.theme import COLORS, apply_dark_theme


_EMB_COLS = ["kline_emb_1", "kline_emb_2", "kline_emb_3", "kline_emb_4"]


def plot_kline_emb_pca(kline_path: Path, panel_path: Path, out_path: Path,
                      max_points: int = 8000) -> Path:
    """最新日期下，K 线嵌入 PCA → 2D，按行业上色。"""
    apply_dark_theme()
    kemb = pd.read_parquet(kline_path)
    latest = kemb["date"].max()
    snap = kemb[kemb["date"] == latest].copy()

    panel = pd.read_parquet(panel_path, columns=["stock_code", "industry"]).drop_duplicates("stock_code")
    snap = snap.merge(panel, on="stock_code", how="left")
    snap["industry"] = snap["industry"].fillna("UNKNOWN")

    if len(snap) > max_points:
        snap = snap.sample(max_points, random_state=42)

    X = snap[_EMB_COLS].values
    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    snap["pca_x"], snap["pca_y"] = coords[:, 0], coords[:, 1]

    top_inds = snap["industry"].value_counts().head(8).index.tolist()
    snap["ind_disp"] = snap["industry"].where(snap["industry"].isin(top_inds), "其他")
    palette = [COLORS["blue"], COLORS["gold"], COLORS["down"], COLORS["purple"],
               COLORS["orange"], COLORS["pink"], COLORS["cyan"], COLORS["up"], COLORS["neutral"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, ind in enumerate(top_inds + ["其他"]):
        sub = snap[snap["ind_disp"] == ind]
        if sub.empty:
            continue
        ax.scatter(sub["pca_x"], sub["pca_y"], s=10, alpha=0.6,
                   color=palette[i % len(palette)],
                   label=f"{ind} (n={len(sub)})")
    ax.set_xlabel("PCA-1", color=COLORS["text"])
    ax.set_ylabel("PCA-2", color=COLORS["text"])
    ax.set_title(f"K 线编码器嵌入 PCA · 按行业 Top8 上色 · date={pd.Timestamp(latest).strftime('%Y-%m-%d')}",
                 color=COLORS["text"], fontsize=12, pad=10)
    ax.legend(loc="best", fontsize=8, ncol=2)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=COLORS["bg"])
    plt.close(fig)
    return out_path


def plot_kline_auxiliary_predictions(kline_path: Path, panel_path: Path,
                                     out_path: Path) -> Path:
    """K 线编码器的辅助预测 vs 真实诊断散点：ret_fwd_1d / vol_5d。"""
    apply_dark_theme()
    kemb = pd.read_parquet(kline_path)
    panel = pd.read_parquet(panel_path, columns=["date", "stock_code", "ret_fwd_1d", "vol_5d"])
    panel["date"] = pd.to_datetime(panel["date"])
    kemb["date"] = pd.to_datetime(kemb["date"])

    df = kemb.merge(panel, on=["date", "stock_code"], how="inner").dropna(
        subset=["kline_pred_ret_1d", "ret_fwd_1d", "kline_pred_vol_5d", "vol_5d"]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, xc, yc, title in [
        (ax1, "kline_pred_ret_1d", "ret_fwd_1d", "辅助预测下一日收益"),
        (ax2, "kline_pred_vol_5d", "vol_5d", "辅助预测 5 日波动"),
    ]:
        x = df[xc].values; y = df[yc].values
        p_lo, p_hi = np.quantile(x, [0.01, 0.99])
        a_lo, a_hi = np.quantile(y, [0.01, 0.99])
        mask = (x >= p_lo) & (x <= p_hi) & (y >= a_lo) & (y <= a_hi)
        x, y = x[mask], y[mask]
        hb = ax.hexbin(x, y, gridsize=40, mincnt=1, cmap="cool")
        cb = fig.colorbar(hb, ax=ax)
        cb.ax.yaxis.set_tick_params(color=COLORS["text"])
        plt.setp(cb.ax.get_yticklabels(), color=COLORS["text"])
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color=COLORS["text_sub"], ls="--", lw=1.0)
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, slope * xs + intercept, color=COLORS["gold"], lw=1.6,
                    label=f"slope={slope:+.3f}")
        pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else 0.0
        ax.set_xlabel(xc, color=COLORS["text"])
        ax.set_ylabel(yc, color=COLORS["text"])
        ax.set_title(f"{title}  Pearson={pearson:+.3f}  N={len(x)}",
                     color=COLORS["text"], fontsize=11, pad=8)
        ax.legend(loc="best", fontsize=9)
        ax.axhline(0, color=COLORS["text_sub"], lw=0.5, alpha=0.4)
        ax.axvline(0, color=COLORS["text_sub"], lw=0.5, alpha=0.4)

    fig.suptitle("K 线编码器：辅助任务预测 vs 真实", color=COLORS["text"], fontsize=13)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=COLORS["bg"])
    plt.close(fig)
    return out_path
