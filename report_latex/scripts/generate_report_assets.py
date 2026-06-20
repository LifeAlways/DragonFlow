from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ASSETS = REPO_ROOT / "report_latex" / "assets"
TABLES = REPO_ROOT / "report_latex" / "tables"
ASSETS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

from dragonflow.viz.charts_matplotlib import (  # noqa: E402
    plot_cluster_boxplot,
    plot_elbow_silhouette,
    plot_limit_up_down_top,
    plot_max_drawdown_hist,
    plot_monthly_drawdown_top,
    plot_monthly_return_violin,
    plot_pca_scatter,
    plot_price_volume_corr_hist,
    plot_skew_kurtosis,
    plot_top_bottom_returns,
    plot_turnover_vs_return,
    plot_volatility_vs_return,
)
from dragonflow.viz.charts_pyecharts import (  # noqa: E402
    chart_cluster_radar,
    chart_daily_amount_area,
    chart_daily_up_down_bar,
    chart_index_line,
    chart_kline,
    chart_multi_stock_lines,
)


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / path, dtype={"stock_code": str}, encoding="utf-8-sig")


def save_fig(fig, filename: str) -> None:
    fig.savefig(ASSETS / filename, dpi=180, bbox_inches="tight", facecolor="white")


def render_chart(chart, filename: str) -> None:
    chart.render(str(ASSETS / filename))


def save_daily_amount_static(daily_df: pd.DataFrame) -> None:
    amount = daily_df.groupby("date")["amount"].sum().sort_index() / 1e8
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    ax.plot(amount.index, amount.values, color="#06b6d4", linewidth=2)
    ax.fill_between(amount.index, amount.values, color="#06b6d4", alpha=0.18)
    ax.set_title("中证2000成分股日成交额走势", fontsize=14, fontweight="bold")
    ax.set_ylabel("成交额（亿元）")
    ax.grid(axis="y", color="#e5e7eb")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate(rotation=25)
    save_fig(fig, "05_daily_amount.png")
    plt.close(fig)


def save_cluster_radar_static(features_df: pd.DataFrame) -> None:
    metrics = [
        ("cum_return", "收益"),
        ("annual_volatility", "波动"),
        ("max_drawdown", "回撤"),
        ("avg_turnover", "换手"),
        ("avg_amplitude", "振幅"),
        ("up_day_ratio", "上涨天数"),
    ]
    summary = features_df.groupby("cluster_name")[[m for m, _ in metrics]].mean()
    norm = summary.copy()
    for col in norm.columns:
        mn, mx = norm[col].min(), norm[col].max()
        if mx == mn:
            norm[col] = 0.5
        elif col == "max_drawdown":
            norm[col] = 1 - (norm[col] - mn) / (mx - mn)
        else:
            norm[col] = (norm[col] - mn) / (mx - mn)

    labels = [label for _, label in metrics]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 7), facecolor="white")
    ax = fig.add_subplot(111, polar=True)
    palette = ["#06b6d4", "#ec4899", "#10b981", "#f97316"]
    for idx, (cluster_name, row) in enumerate(norm.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, color=palette[idx % len(palette)], linewidth=2, label=cluster_name)
        ax.fill(angles, values, color=palette[idx % len(palette)], alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels([])
    ax.set_title("聚类画像雷达图", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.08), frameon=True)
    save_fig(fig, "12_cluster_radar.png")
    plt.close(fig)


def save_representative_kline_static(daily_df: pd.DataFrame, code: str, name: str, cluster_name: str) -> None:
    df = daily_df[daily_df["stock_code"].astype(str).str.zfill(6) == code].sort_values("date").copy()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}, facecolor="white"
    )
    ax1.plot(df["date"], df["close"], color="#2563eb", linewidth=2, label="收盘价")
    ax1.plot(df["date"], df["close"].rolling(5).mean(), color="#f97316", linewidth=1.4, label="MA5")
    ax1.plot(df["date"], df["close"].rolling(20).mean(), color="#64748b", linewidth=1.4, label="MA20")
    ax1.set_title(f"{name}（{cluster_name}）代表股价格走势", fontsize=14, fontweight="bold")
    ax1.set_ylabel("价格")
    ax1.grid(axis="y", color="#e5e7eb")
    ax1.legend(loc="upper left", frameon=False)
    ax2.bar(df["date"], df["volume"], color="#38bdf8", alpha=0.65)
    ax2.set_ylabel("成交量")
    ax2.grid(axis="y", color="#e5e7eb")
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.autofmt_xdate(rotation=25)
    save_fig(fig, "14_representative_kline.png")
    plt.close(fig)


def main() -> None:
    daily_df = read_csv("data/processed/stock_daily_csi2000_qfq_20260101_20260531_clean.csv")
    index_daily = read_csv("data/processed/index_daily_932000_20260101_20260531.csv")
    features_df = read_csv("data/processed/stock_clusters.csv")
    pca_df = read_csv("data/processed/pca_2d.csv")
    k_search_df = read_csv("data/processed/k_search.csv")

    for frame in (daily_df, index_daily):
        frame["date"] = pd.to_datetime(frame["date"])

    render_chart(chart_index_line(index_daily), "01_index_line.html")
    render_chart(chart_daily_up_down_bar(daily_df), "02_daily_up_down.html")
    render_chart(chart_daily_amount_area(daily_df), "05_daily_amount.html")
    render_chart(chart_cluster_radar(features_df), "12_cluster_radar.html")

    representatives: dict[str, tuple[str, str]] = {}
    for cluster_name, group in features_df.groupby("cluster_name"):
        mean_ret = group["cum_return"].mean()
        closest_idx = (group["cum_return"] - mean_ret).abs().idxmin()
        rep = group.loc[closest_idx]
        representatives[str(cluster_name)] = (
            str(rep["stock_code"]).zfill(6),
            str(rep.get("stock_name", "")),
        )

    rep_codes = [code for code, _ in representatives.values()]
    rep_names = {
        code: f"{name}({cluster_name})"
        for cluster_name, (code, name) in representatives.items()
    }
    render_chart(chart_multi_stock_lines(daily_df, rep_codes, rep_names), "13_representative_lines.html")

    first_cluster, (first_code, first_name) = next(iter(representatives.items()))
    render_chart(chart_kline(daily_df, first_code, f"{first_name}({first_cluster})"), "14_representative_kline.html")
    save_daily_amount_static(daily_df)
    save_cluster_radar_static(features_df)
    save_representative_kline_static(daily_df, first_code, first_name, first_cluster)

    static_jobs = [
        (plot_monthly_return_violin(daily_df), "03_monthly_return_violin.png"),
        (plot_top_bottom_returns(features_df), "04_top_bottom_returns.png"),
        (plot_volatility_vs_return(features_df, color_col="cluster_name"), "06_volatility_vs_return.png"),
        (plot_max_drawdown_hist(features_df), "07_max_drawdown_hist.png"),
        (plot_skew_kurtosis(features_df), "08_skew_kurtosis.png"),
        (plot_monthly_drawdown_top(daily_df), "09_monthly_drawdown_top.png"),
        (plot_turnover_vs_return(features_df), "10_turnover_vs_return.png"),
        (plot_limit_up_down_top(features_df), "11_limit_up_down_top.png"),
        (plot_price_volume_corr_hist(features_df), "15_price_volume_corr_hist.png"),
        (plot_pca_scatter(pca_df), "16_pca_cluster_scatter.png"),
        (plot_elbow_silhouette(k_search_df), "17_elbow_silhouette.png"),
        (plot_cluster_boxplot(features_df), "18_cluster_boxplot.png"),
    ]
    for fig, filename in static_jobs:
        save_fig(fig, filename)

    notebook_files = REPO_ROOT / "notebooks" / "dragonflow_analysis_files"
    extracted = []
    if notebook_files.exists():
        for src in sorted(notebook_files.glob("dragonflow_analysis_*_0.png")):
            dst = ASSETS / f"notebook_{src.name}"
            shutil.copy2(src, dst)
            extracted.append(dst.name)

    cluster_summary = (
        features_df.groupby(["cluster_id", "cluster_name"])
        .agg(
            stock_count=("stock_code", "count"),
            mean_return=("cum_return", "mean"),
            mean_volatility=("annual_volatility", "mean"),
            mean_drawdown=("max_drawdown", "mean"),
            mean_turnover=("avg_turnover", "mean"),
            mean_amplitude=("avg_amplitude", "mean"),
        )
        .round(3)
        .reset_index()
    )
    cluster_summary.to_csv(TABLES / "cluster_summary.csv", index=False, encoding="utf-8-sig")

    k_search_df.round(4).to_csv(TABLES / "k_search.csv", index=False, encoding="utf-8-sig")

    coverage = pd.read_csv(REPO_ROOT / "data/processed/data_coverage_report.csv", encoding="utf-8-sig")
    preprocess = pd.read_csv(REPO_ROOT / "data/processed/preprocess_report.csv", encoding="utf-8-sig")
    summary = {
        "n_stocks": int(features_df["stock_code"].nunique()),
        "n_daily_rows": int(len(daily_df)),
        "n_index_rows": int(len(index_daily)),
        "date_start": str(daily_df["date"].min().date()),
        "date_end": str(daily_df["date"].max().date()),
        "mean_daily_rows": round(float(coverage["n_daily_rows"].mean()), 3),
        "min_daily_rows": int(coverage["n_daily_rows"].min()),
        "max_daily_rows": int(coverage["n_daily_rows"].max()),
        "download_success": int(coverage["download_success"].sum()),
        "anomaly_count": int(preprocess["anomaly_count"].sum()),
        "remaining_na": int(preprocess["remaining_na"].sum()),
        "pca_components": 11,
        "pca_variance": 0.9087,
        "best_k": int(k_search_df.sort_values("silhouette", ascending=False).iloc[0]["k"]),
        "representatives": representatives,
        "extracted_notebook_images": extracted,
    }
    (TABLES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
