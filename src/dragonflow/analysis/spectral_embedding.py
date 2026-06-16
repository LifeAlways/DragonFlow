"""Rolling spectral stock embeddings for DragonFlow-KronosGraph."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from dragonflow.utils.logger import get_logger

logger = get_logger(__name__)


def _topk_symmetric(sim: np.ndarray, n_neighbors: int) -> sparse.csr_matrix:
    n = sim.shape[0]
    rows, cols, vals = [], [], []
    for i in range(n):
        row = sim[i].copy()
        row[i] = 0.0
        if n_neighbors < n - 1:
            idx = np.argpartition(row, -n_neighbors)[-n_neighbors:]
        else:
            idx = np.arange(n)
        idx = idx[row[idx] > 0]
        rows.extend([i] * len(idx))
        cols.extend(idx.tolist())
        vals.extend(row[idx].tolist())
    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return mat.maximum(mat.T)


def _spectral_embedding_from_returns(ret: pd.DataFrame, dim: int, n_neighbors: int) -> np.ndarray:
    corr = ret.corr(min_periods=max(10, min(20, len(ret) // 2))).fillna(0.0).values
    sim = np.maximum(corr, 0.0)
    np.fill_diagonal(sim, 0.0)
    graph = _topk_symmetric(sim, n_neighbors=n_neighbors)
    degrees = np.asarray(graph.sum(axis=1)).ravel()
    degrees[degrees <= 1e-12] = 1e-12
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
    lap = sparse.eye(graph.shape[0], format="csr") - d_inv_sqrt @ graph @ d_inv_sqrt
    k = min(dim + 1, graph.shape[0] - 1)
    vals, vecs = eigsh(lap, k=k, which="SM")
    order = np.argsort(vals)
    emb = vecs[:, order[1:dim + 1]] if k > 1 else vecs[:, order[:dim]]
    if emb.shape[1] < dim:
        emb = np.pad(emb, ((0, 0), (0, dim - emb.shape[1])))
    return normalize(emb)


def _choose_k(emb: np.ndarray, k_min: int, k_max: int, random_state: int) -> tuple[int, np.ndarray]:
    best_k, best_score, best_labels = k_min, -np.inf, None
    max_allowed = min(k_max, len(emb) - 1)
    for k in range(k_min, max_allowed + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(emb)
        counts = np.bincount(labels)
        if counts.min() < 10:
            continue
        try:
            score = silhouette_score(emb, labels)
        except Exception:
            score = -np.inf
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    if best_labels is None:
        best_labels = KMeans(n_clusters=best_k, n_init=10, random_state=random_state).fit_predict(emb)
    return best_k, best_labels


def build_rolling_spectral_embeddings(
    panel: pd.DataFrame,
    graph_window: int = 40,
    refit_every_n_days: int = 5,
    embedding_dim: int = 8,
    n_neighbors: int = 30,
    k_min: int = 8,
    k_max: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    df = panel[["date", "stock_code", "time_idx", "ret_1d"]].copy()
    df = df.sort_values(["time_idx", "stock_code"])
    all_times = sorted(df["time_idx"].unique())
    refit_times = [t for t in all_times if t >= graph_window and (t - graph_window) % refit_every_n_days == 0]
    records: list[pd.DataFrame] = []
    logger.info("开始滚动谱嵌入：{} 个 refit 点", len(refit_times))
    for t in refit_times:
        hist = df[(df["time_idx"] > t - graph_window) & (df["time_idx"] <= t)]
        ret = hist.pivot(index="time_idx", columns="stock_code", values="ret_1d").fillna(0.0)
        stocks = ret.columns.astype(str).tolist()
        emb = _spectral_embedding_from_returns(ret, embedding_dim, n_neighbors)
        _, labels = _choose_k(emb, k_min, k_max, random_state)
        out = pd.DataFrame({"refit_time_idx": t, "stock_code": stocks, "cluster_id": labels.astype(int)})
        for j in range(embedding_dim):
            out[f"spectral_emb_{j+1}"] = emb[:, j]
        records.append(out)
        logger.info("谱嵌入完成：time_idx={} stocks={}", t, len(stocks))
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def attach_spectral_features(panel: pd.DataFrame, embeddings: pd.DataFrame, embedding_dim: int = 8) -> pd.DataFrame:
    placeholder_cols = ["cluster_id"] + [f"spectral_emb_{j+1}" for j in range(embedding_dim)]
    base = panel.drop(columns=[c for c in placeholder_cols if c in panel.columns], errors="ignore")
    if embeddings.empty:
        out = base.copy()
        out["cluster_id"] = 0
        for j in range(embedding_dim):
            out[f"spectral_emb_{j+1}"] = 0.0
        return out
    times = base[["time_idx"]].drop_duplicates().sort_values("time_idx")
    refits = embeddings[["refit_time_idx"]].drop_duplicates().sort_values("refit_time_idx")
    mapping = pd.merge_asof(times, refits, left_on="time_idx", right_on="refit_time_idx", direction="backward")
    out = base.merge(mapping, on="time_idx", how="left")
    out = out.merge(embeddings, on=["refit_time_idx", "stock_code"], how="left")
    out["cluster_id"] = out["cluster_id"].fillna(-1).astype(int)
    for j in range(embedding_dim):
        col = f"spectral_emb_{j+1}"
        out[col] = out[col].fillna(0.0)
    return out.drop(columns=["refit_time_idx"], errors="ignore")


def add_cluster_peer_features(panel: pd.DataFrame) -> pd.DataFrame:
    peer_cols = [
        "cluster_ret_mean_1d",
        "cluster_ret_mean_5d",
        "cluster_amount_mean",
        "cluster_turnover_mean",
        "stock_ret_minus_cluster_1d",
        "stock_ret_minus_cluster_5d",
    ]
    out = panel.drop(columns=[c for c in peer_cols if c in panel.columns], errors="ignore")
    grp = out.groupby(["date", "cluster_id"], dropna=False)
    peer = grp.agg(
        cluster_ret_mean_1d=("ret_1d", "mean"),
        cluster_ret_mean_5d=("ret_5d", "mean"),
        cluster_amount_mean=("amount", "mean"),
        cluster_turnover_mean=("turnover_rate", "mean"),
    ).reset_index()
    out = out.merge(peer, on=["date", "cluster_id"], how="left")
    out["stock_ret_minus_cluster_1d"] = out["ret_1d"] - out["cluster_ret_mean_1d"]
    out["stock_ret_minus_cluster_5d"] = out["ret_5d"] - out["cluster_ret_mean_5d"]
    return out
