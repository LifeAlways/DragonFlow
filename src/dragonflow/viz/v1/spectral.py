"""谱聚类嵌入可视化：PCA 投影 + 簇规模时间堆叠。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from dragonflow.viz.theme import CLUSTER_PALETTE, COLORS, apply_dark_theme


def _cluster_color(idx: int) -> str:
    return CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)]


def plot_spectral_pca(emb_path: Path, panel_path: Path, out_path: Path,
                      refit_time_idx: int | None = None) -> Path:
    """对最新（或指定）refit 的 8 维谱嵌入做 PCA → 2D 散点，cluster 上色 + industry 上色。"""
    apply_dark_theme()
    emb = pd.read_parquet(emb_path)
    if refit_time_idx is None:
        refit_time_idx = int(emb["refit_time_idx"].max())
    snap = emb[emb["refit_time_idx"] == refit_time_idx].copy()

    feat_cols = [c for c in snap.columns if c.startswith("spectral_emb_")]
    X = snap[feat_cols].values
    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    snap["pca_x"], snap["pca_y"] = coords[:, 0], coords[:, 1]

    # join industry
    panel = pd.read_parquet(panel_path, columns=["stock_code", "industry"]).drop_duplicates("stock_code")
    snap = snap.merge(panel, on="stock_code", how="left")
    snap["industry"] = snap["industry"].fillna("UNKNOWN")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # ---- 左：按 cluster_id 上色
    for cid, grp in snap.groupby("cluster_id"):
        ax1.scatter(grp["pca_x"], grp["pca_y"], s=10, alpha=0.7,
                    color=_cluster_color(int(cid)),
                    label=f"C{int(cid)} (n={len(grp)})", rasterized=True)
    ax1.set_xlabel("PCA-1", color=COLORS["text"])
    ax1.set_ylabel("PCA-2", color=COLORS["text"])
    n_clusters = snap["cluster_id"].nunique()
    ax1.set_title(
        f"谱嵌入 PCA · 按 cluster 上色 · refit_time_idx={refit_time_idx} · K={n_clusters}",
        color=COLORS["text"], fontsize=12, pad=10,
    )
    ax1.legend(loc="best", fontsize=7, ncol=2)

    # ---- 右：按 top-N 行业上色
    top_inds = snap["industry"].value_counts().head(8).index.tolist()
    snap["ind_disp"] = snap["industry"].where(snap["industry"].isin(top_inds), "其他")
    palette = [COLORS["blue"], COLORS["gold"], COLORS["down"], COLORS["purple"],
               COLORS["orange"], COLORS["pink"], COLORS["cyan"], COLORS["up"], COLORS["neutral"]]
    for i, ind in enumerate(top_inds + ["其他"]):
        sub = snap[snap["ind_disp"] == ind]
        if sub.empty:
            continue
        ax2.scatter(sub["pca_x"], sub["pca_y"], s=10, alpha=0.7,
                    color=palette[i % len(palette)],
                    label=f"{ind} (n={len(sub)})", rasterized=True)
    ax2.set_xlabel("PCA-1", color=COLORS["text"])
    ax2.set_ylabel("PCA-2", color=COLORS["text"])
    ax2.set_title("同图 · 按行业 Top8 上色", color=COLORS["text"], fontsize=12, pad=10)
    ax2.legend(loc="best", fontsize=7, ncol=2)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=COLORS["bg"])
    plt.close(fig)
    return out_path


def plot_cluster_size_over_time(emb_path: Path, out_path: Path) -> Path:
    """每个 refit 日的簇规模堆叠面积图，体现簇结构随时间漂移。"""
    apply_dark_theme()
    emb = pd.read_parquet(emb_path)
    sizes = (
        emb.groupby(["refit_time_idx", "cluster_id"]).size().unstack("cluster_id").fillna(0).sort_index()
    )
    # 按规模总和排序，前 12 个独立显示，剩余合并
    top_cols = sizes.sum().sort_values(ascending=False).head(12).index.tolist()
    other_cols = [c for c in sizes.columns if c not in top_cols]
    if other_cols:
        sizes["其他"] = sizes[other_cols].sum(axis=1)
        sizes = sizes[top_cols + ["其他"]]

    colors = [_cluster_color(int(c) if isinstance(c, (int, np.integer)) else i)
              for i, c in enumerate(sizes.columns)]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.stackplot(sizes.index, sizes.T.values, labels=[str(c) for c in sizes.columns],
                 colors=colors, alpha=0.9)
    ax.set_xlabel("refit_time_idx", color=COLORS["text"])
    ax.set_ylabel("簇内股票数", color=COLORS["text"])
    ax.set_title("谱聚类簇规模随 refit 时间变化（堆叠面积）",
                 color=COLORS["text"], fontsize=13, pad=10)
    ax.legend(loc="upper left", fontsize=7, ncol=4, title="cluster_id")
    ax.margins(x=0)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=COLORS["bg"])
    plt.close(fig)
    return out_path


def plot_cluster_transition_heat(emb_path: Path, out_path: Path) -> Path:
    """相邻两次 refit 之间的簇映射矩阵，衡量聚类稳定性。"""
    apply_dark_theme()
    emb = pd.read_parquet(emb_path)
    refits = sorted(emb["refit_time_idx"].unique())
    if len(refits) < 2:
        return out_path
    first, last = refits[0], refits[-1]
    df0 = emb[emb["refit_time_idx"] == first].set_index("stock_code")["cluster_id"]
    df1 = emb[emb["refit_time_idx"] == last].set_index("stock_code")["cluster_id"]
    joint = pd.concat({"first": df0, "last": df1}, axis=1).dropna()
    cm = pd.crosstab(joint["first"].astype(int), joint["last"].astype(int))

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(cm.values, cmap="cool", aspect="auto")
    ax.set_xticks(range(len(cm.columns))); ax.set_xticklabels(cm.columns, color=COLORS["text"])
    ax.set_yticks(range(len(cm.index))); ax.set_yticklabels(cm.index, color=COLORS["text"])
    ax.set_xlabel(f"cluster_id @ refit={last}", color=COLORS["text"])
    ax.set_ylabel(f"cluster_id @ refit={first}", color=COLORS["text"])
    ax.set_title(f"簇成员变迁矩阵（refit {first} → {last}）",
                 color=COLORS["text"], fontsize=12, pad=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.values[i, j]
            if v:
                ax.text(j, i, int(v), ha="center", va="center",
                        color=COLORS["text"], fontsize=8)
    fig.colorbar(im, ax=ax, label="重叠股票数")
    ax.grid(False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=COLORS["bg"])
    plt.close(fig)
    return out_path
