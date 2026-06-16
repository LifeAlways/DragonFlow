# DragonFlow

中证 2000 全成分股的端到端量化研究原型 —— 从 AkShare 取数、特征工程、谱聚类嵌入、TFT 分位数预测、组合回测，到 14 张矢量可视化报告。

> 本仓库同时作为西南财经大学
> **《数据可视化》** 与 **《金融大数据分析》** 两门课程的小组项目。

## 课程双线定位

| 课程 | 重点交付 | 入口 |
|---|---|---|
| **数据可视化** | Notebook 探索分析、暗色金融主题图表（matplotlib + pyecharts）、流水线产物的 12 张诊断图 | `notebooks/dragonflow_analysis.ipynb`、`data/processed/viz_v1/index.html` |
| **金融大数据分析** | "谱聚类嵌入 + Temporal Fusion Transformer + 5 日超额收益分位预测 + 组合回测"端到端流水线（DragonFlow-KronosGraph V1） | `scripts/07_*` ~ `scripts/14_*`、`docs/` 三份方案 |

**Demo Presentation 主题**：《从夯到拉锐评 2026 年 1 至 5 月热点龙头股》

---

## 端到端流水线（14 步）

数据下载、合成代理、清洗在 01–06；建模、训练、预测、回测、可视化、消融在 07–14。

```bash
# 第一阶段：数据
uv run python scripts/01_download_csi2000_data.py        # 中证 2000 成分股 + 行情 + 财报
uv run python scripts/02_finalize_partial.py             # 兜底/收尾
uv run python scripts/03_synthesize_index_proxy.py       # 等权代理指数（EM 掐流兜底）
uv run python scripts/04_synthesize_spot_snapshot.py     # 截面快照合成
uv run python scripts/05_feature_engineering.py          # 单股 + 截面特征
uv run python scripts/06_clustering.py                   # 基础聚类

# 第二阶段：建模（DragonFlow-KronosGraph V1）
uv run python scripts/07_build_model_dataset.py                                # 建模面板 189811 × 104
uv run python scripts/08_spectral_embedding.py   --config configs/model_v1.yaml # 滚动谱嵌入 + cluster_id
uv run python scripts/09_train_kline_encoder.py  --config configs/model_v1.yaml # Kronos 思路 K 线编码器
uv run python scripts/10_train_tft.py            --config configs/model_v1.yaml # 小型 TFT 分位回归
uv run python scripts/11_predict_tft.py          --config configs/model_v1.yaml --range test
uv run python scripts/12_backtest_strategy.py    --config configs/model_v1.yaml # Top-N 多头回测
uv run python scripts/13_visualize_v1.py                                       # 12 张 SVG + index.html
uv run python scripts/14_backtest_compare.py                                   # 4 变体消融实验
```

---

## 数据可视化交付

### Notebook（《数据可视化》课程主交付）

- `notebooks/dragonflow_analysis.ipynb` — 主分析 notebook
- `notebooks/dragonflow_analysis.html` / `.pdf` / `.md` — 导出版本（无需 Jupyter 即可阅读）
- `notebooks/_charts/` — 8 份 pyecharts 交互式 HTML
- `notebooks/dragonflow_analysis_files/` — 17 张 matplotlib PNG

Notebook 调用 `src/dragonflow/viz/` 下的可视化函数：

| 模块 | 用途 |
|---|---|
| `viz/theme.py` | 金融暗色主题（matplotlib + pyecharts 共用）、聚类色板 |
| `viz/charts_matplotlib.py` | 14 张静态图：月收益小提琴、行业风险收益气泡、波动 vs 收益、PCA 散点、肘点轮廓、聚类箱线 |
| `viz/charts_pyecharts.py` | 10 张交互图：指数线、涨跌柱、成交额面积、行业月度热力图、河流图、聚类雷达、桑基图、K 线、多股对比 |

### V1 流水线可视化（《金融大数据分析》课程支撑）

14 张 **SVG**（矢量可编辑，文字保留为 `<text>` 节点）输出到 `data/processed/viz_v1/`：

| # | 图 | 说明 |
|---|---|---|
| 01 | NAV + 水下回撤 | 策略净值、最大回撤区间 |
| 02 | 持仓 & 换手率 | 双轴时间序列 |
| 03 | Rank IC | 日度柱状 + 累计折线（IC=+0.082 / ICIR=+1.12） |
| 04 | 5 分位组合 | Q1–Q5 累计超额 + 多空价差（Top-Bottom +18.75%） |
| 05 | 预测 vs 真实 | q50 hexbin + OLS 回归 |
| 06 | 分位数校准 | q10/q50/q90 点图 + 80% 区间覆盖率 |
| 07 | 预测横截面分布 | 25/50/75 分位带 + 模型不确定区间 |
| 08 | 谱嵌入 PCA | 双面板：cluster 着色 / 行业着色 |
| 09 | 簇规模时间堆叠 | 12 个 cluster 随 refit 漂移 |
| 10 | 簇成员变迁矩阵 | 首尾 refit 重叠混淆矩阵 |
| 11 | K 线嵌入 PCA | 行业 Top8 上色 |
| 12 | K 线辅助预测诊断 | ret_1d / vol_5d 散点 |
| **13** | **回测消融 NAV 对比** | 4 变体叠加（baseline / no_q10 / q50_only / both） |
| **14** | **消融指标表** | total / sharpe / mdd 对照 |

入口：`data/processed/viz_v1/index.html`。SVG 在 Inkscape / Illustrator / Affinity Designer 中可直接编辑文字与配色。

---

## 主要结论（V1 实测）

| 指标 | 原配置 | 修复后 |
|---|---|---|
| **总收益** | -11.81% | **+1.50%** |
| **Sharpe** | -3.50 | **+0.26** |
| **最大回撤** | -11.81% | **-5.26%** |

模型本身有 alpha（Rank IC +0.082、Top–Bottom +18.75%），但原 `q10_floor=-0.03` 把候选池从 1999 砍到 2 只，导致回测亏 11.8%。详见 `docs/` 与 `data/processed/viz_v1/13_compare_nav.svg`。

---

## 项目结构

```
DragonFlow/
├── app/                              # 前端：FastAPI + React (Vite/TS) + Streamlit
│   ├── api/main.py                   # FastAPI 后端
│   ├── src/                          # React 前端（6 页 + 3 组件）
│   └── streamlit_app.py              # Streamlit 单文件 dashboard
├── configs/
│   └── model_v1.yaml                 # V1 训练 + 回测配置
├── data/
│   ├── raw/                          # 原始下载（gitignore）
│   └── processed/                    # 加工产物 + V1 流水线输出
│       ├── stock_daily_*.parquet     # 个股日线长表
│       ├── model_panel_*.parquet     # 建模面板
│       ├── tft_predictions.parquet   # TFT 预测
│       ├── backtest_*.parquet        # 回测净值/持仓/指标
│       └── viz_v1/                   # 14 张 SVG + index.html
├── docs/
│   ├── spectral_tft_quant_strategy.md      # 策略总蓝图
│   ├── tft_feature_architecture_v1.md      # 特征/IO schema/模型架构
│   └── training_guide_kronosgraph_v1.md    # 服务器训练手册
├── notebooks/                        # 《数据可视化》课程交付
│   ├── dragonflow_analysis.ipynb
│   └── _charts/, *_files/, *.html/.pdf/.md
├── scripts/01-14_*.py                # 14 步流水线
├── src/dragonflow/
│   ├── analysis/                     # 聚类、谱嵌入
│   ├── backtest/                     # portfolio / execution / metrics
│   ├── data/                         # 下载、schema、预处理
│   ├── features/                     # 特征工程
│   ├── modeling/                     # TFT、K 线编码器、targets、技术/市场特征
│   ├── utils/                        # io / logger
│   └── viz/
│       ├── charts_matplotlib.py      # 静态图
│       ├── charts_pyecharts.py       # 交互图
│       ├── theme.py                  # 暗色主题
│       └── v1/                       # V1 流水线专用可视化（与上面隔离）
│           ├── backtest.py
│           ├── predictions.py
│           ├── spectral.py
│           ├── kline.py
│           └── compare.py
└── pyproject.toml + uv.lock
```

---

## 快速开始

```bash
# 1. 安装基础依赖
uv sync

# 2. 装 PyTorch（按机器情况二选一）
# GPU（CUDA ≥ 13.0 驱动）：
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
# 或纯 CPU 冒烟测试：
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. 已包含 2026-01 ~ 05 中证2000 数据快照，无需再跑 01_download
# 直接从 07 开始：
uv run python scripts/07_build_model_dataset.py
```

启动 React 前端（可选）：
```bash
cd app && npm install && npm run dev
```

启动 Streamlit dashboard（可选）：
```bash
uv run streamlit run app/streamlit_app.py
```

---

## 技术栈

| 模块 | 技术 | 用途 |
|---|---|---|
| 开发语言 | Python 3.11 | 主要开发语言 |
| 数据获取 | AkShare | 中证 2000 成分股/指数/财报 |
| 数据处理 | Pandas / Numpy / PyArrow | 长表清洗、滚动窗口、parquet IO |
| 机器学习 | Scikit-learn / Scipy | PCA / KMeans / 谱嵌入 / 相关性 |
| 深度学习 | **PyTorch** | TFT 分位回归、K 线编码器 |
| 静态可视化 | Matplotlib | 14 张 V1 SVG + 笔记本统计图 |
| 交互可视化 | Pyecharts | K 线、桑基、热力图、河流 |
| Web 前端 | React + Vite + TS / Streamlit | 双前端方案 |
| Web 后端 | FastAPI + Uvicorn | 数据 API |
| 数据存储 | CSV / Parquet | 原始 + 加工 |
| 笔记本 | Jupyter | 探索式数据可视化 |
| 项目管理 | uv + Git | 依赖锁定 + 版本管理 |

---

## 仓库里已有什么数据（2026-06-08 快照）

下一位同学 clone 之后**无需再下载**就能直接开干。部分数据因 EM 代理偶发掐流，已用本地合成代理顶替，"来源"列里 `computed_local_*` / `derived_from_*` 即为本地合成产物，**不是官方接口数据**，做学术报告时请如实标注。

| 数据 | 状态 | 文件 | 行数 | 来源 |
|---|---|---|---|---|
| 中证2000成分股 | ✅ 100% | `data/raw/csi2000/constituents_932000_*.csv` | 2000 | 中证指数 `index_stock_cons_csindex` |
| **个股日线**（前复权）| ✅ **100%** | `data/raw/stock_daily/qfq/*.csv` + `data/processed/stock_daily_csi2000_qfq_*.parquet` | 2000 只 / 189,811 行 | EM `stock_zh_a_hist` 优先 + 新浪兜底 |
| **中证2000指数日线**（官方）| ✅ **100%** | `data/raw/csi2000/index_daily_932000_*.csv/.parquet` | 95 | EM `stock_zh_index_daily_em(csi932000)` |
| 中证2000指数日线（本地代理）| ✅ 备用 | `data/processed/index_daily_932000_proxy_equal_weight_*.parquet` | 94 | `computed_local_equal_weight_proxy` |
| 个股基础信息 | ⚠️ 68.6% | `data/processed/stock_info_csi2000_latest.parquet` | 1371 / 2000 | EM `stock_individual_info_em` |
| 截面快照（本地合成）| ✅ 替代 | `data/processed/stock_spot_snapshot_csi2000_latest.parquet` | 2000 | `derived_from_last_daily_row` |
| 利润表 (2026Q1) | ✅ | `data/processed/fundamental_csi2000_latest.parquet` | ~2000 | EM `stock_lrb_em` |
| 资产负债表 (2026Q1) | ✅ | 同上 | ~2000 | EM `stock_zcfz_em` |
| 现金流量表 (2026Q1) | ✅ | 同上 | ~2000 | EM `stock_xjll_em` |

V1 流水线产物（**已纳入 git，无需重训重跑**）：

```
data/processed/
  model_panel_base.parquet         # 189811 × 104 建模面板
  model_panel_tft.parquet          # 同上 + 谱嵌入列
  spectral_embeddings.parquet      # 22000 × (refit, code, cluster, 8 维)
  kline_embeddings.parquet         # 131811 × K 线编码 4 维 + 辅助预测 3 列
  tft_predictions.parquet          # 19998 × (date, code, q10/q50/q90)
  backtest_nav.parquet             # 95 天 NAV
  backtest_positions.parquet       # 持仓明细
  backtest_metrics.json            # total / sharpe / mdd
  viz_v1/                          # 14 张 SVG + 4 个对比 NAV parquet + index.html
```

---

## 常见问题

- **接口失败 / 网络慢**：脚本对成分股、指数行情、财报都配置了多接口 fallback。临时性失败可以直接重跑（默认会跳过已下载文件，断点续跑）。
- **AkShare 限流**：可加大 `--sleep`（如 0.5/1.0）减小请求频率。
- **首次冒烟测试**：可以加 `--limit 20 --skip-fundamental` 只下载 20 只成分股的日线，快速验证环境。
- **本地无 GPU**：装 CPU 版 torch，流水线照样能跑完，只是 09/10 训练慢一些。
- **想自己改图**：所有图都是矢量 SVG，文字是 `<text>` 节点。Inkscape / Illustrator 打开直接编辑；改色可用文本编辑器搜替换 hex（如 `#4fc3f7` → 你的主色）。

---

## 详细方案文档

策略与实现细节见 `docs/`：

- `spectral_tft_quant_strategy.md` — 策略总蓝图（13 节）：目标、特征体系、谱聚类嵌入、TFT 模型设计、信号生成、组合构建、回测、风控、落地路线
- `tft_feature_architecture_v1.md` — V1 特征/IO schema：数据输入、训练标签、80 列 schema、TFT 输入输出、训练切分
- `training_guide_kronosgraph_v1.md` — 服务器训练手册：环境、配置、一键脚本顺序、调参建议
