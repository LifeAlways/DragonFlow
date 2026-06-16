# DragonFlow-KronosGraph V1 训练文档

## 1. 模块目标

`DragonFlow-KronosGraph V1` 是当前项目的端到端量化预测原型：

```text
基础建模面板
-> 谱聚类股票关系嵌入
-> Kronos 启发的轻量 K 线编码器
-> 轻量 Temporal Fusion/Transformer 预测模型
-> 分位数收益预测
-> Top-K 多头回测
```

当前本地没有 GPU，所以本仓库只在本地完成代码和数据面板验证。真实训练建议放到服务器执行。

## 2. 服务器环境

建议：

```text
Python >= 3.11
CUDA GPU 可选但推荐
内存 >= 32GB
显存 >= 8GB
```

安装项目依赖：

```bash
uv sync
```

安装 PyTorch 请按服务器 CUDA 版本选择官方命令。例如 CUDA 12.1：

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

如果只做 CPU 冒烟测试：

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 3. 配置文件

主配置在：

```text
configs/model_v1.yaml
```

关键可调项：

```yaml
spectral:
  graph_window: 40
  refit_every_n_days: 5
  embedding_dim: 8
  n_neighbors: 30
  k_min: 8
  k_max: 20

kline_encoder:
  input_length: 30
  d_model: 24
  n_heads: 2
  n_layers: 1
  batch_size: 512
  max_epochs: 8
  device: auto

model:
  encoder_length: 30
  d_model: 48
  n_heads: 2
  n_layers: 1
  batch_size: 512
  max_epochs: 12
  device: auto
```

`device: auto` 会自动检测 CUDA。服务器上有 GPU 时使用 GPU，没有 GPU 时退回 CPU。

当前 1-5 月数据很短，不建议把 `n_layers`、`d_model` 和 `max_epochs` 调大。补足多年历史后再扩大模型。

## 4. 训练脚本顺序

### 4.1 构建基础面板

```bash
uv run python scripts/07_build_model_dataset.py
```

输出：

```text
data/processed/model_panel_base.parquet
data/processed/model_feature_schema_v1.json
data/processed/model_time_split_v1.json
```

### 4.2 生成谱聚类嵌入

```bash
uv run python scripts/08_spectral_embedding.py --config configs/model_v1.yaml
```

输出：

```text
data/processed/spectral_embeddings.parquet
data/processed/model_panel_tft.parquet
```

`model_panel_tft.parquet` 会在基础面板上增加：

```text
cluster_id
spectral_emb_1 ... spectral_emb_8
cluster_ret_mean_1d
cluster_ret_mean_5d
cluster_amount_mean
cluster_turnover_mean
stock_ret_minus_cluster_1d
stock_ret_minus_cluster_5d
```

### 4.3 训练 K 线编码器

```bash
uv run python scripts/09_train_kline_encoder.py --config configs/model_v1.yaml
```

输出：

```text
models/dragonflow_kronosgraph_v1/kline_encoder.ckpt
data/processed/kline_embeddings.parquet
```

K 线编码器学习：

```text
[t-29, ..., t] 的 K 线形态序列
-> ret_fwd_1d / range_fwd_1d / vol_fwd_5d
-> kline_emb_1 ... kline_emb_4
```

### 4.4 训练主预测模型

```bash
uv run python scripts/10_train_tft.py --config configs/model_v1.yaml
```

输出：

```text
models/dragonflow_kronosgraph_v1/model.ckpt
models/dragonflow_kronosgraph_v1/numeric_scaler.pkl
models/dragonflow_kronosgraph_v1/feature_schema.json
data/processed/model_panel_tft.parquet
```

主模型预测目标：

```text
excess_ret_fwd_5d
```

输出分位数：

```text
pred_q10_excess_ret_fwd_5d
pred_q50_excess_ret_fwd_5d
pred_q90_excess_ret_fwd_5d
```

### 4.5 预测

测试集预测：

```bash
uv run python scripts/11_predict_tft.py --config configs/model_v1.yaml --range test
```

也可以预测验证集、最后无标签展示区间或全部样本外区间：

```bash
uv run python scripts/11_predict_tft.py --range valid
uv run python scripts/11_predict_tft.py --range predict
uv run python scripts/11_predict_tft.py --range all
```

输出：

```text
data/processed/tft_predictions.parquet
```

### 4.6 回测

```bash
uv run python scripts/12_backtest_strategy.py --config configs/model_v1.yaml
```

输出：

```text
data/processed/backtest_nav.parquet
data/processed/backtest_positions.parquet
data/processed/backtest_metrics.json
```

V1 当前默认配置（已纳入消融实验结论）：

```yaml
backtest:
  rebalance_every_n_days: 5
  top_n: 80
  min_amount_mean_20d: 30000000
  q10_floor: -1.0          # 关闭下行过滤
  score_formula: q50       # alpha 直选
  buy_cost_bps: 10
  sell_cost_bps: 10
  slippage_bps: 5
```

实测：total_return +1.50%，Sharpe +0.26，MaxDD -5.26%。

### 4.7 可视化（V1 流水线产物）

```bash
uv run python scripts/13_visualize_v1.py
```

12 张 SVG（矢量可编辑）输出到 `data/processed/viz_v1/`，含 NAV/回撤、Rank IC、
5 分位组合、分位校准、谱嵌入 PCA、K 线编码诊断等。`index.html` 是浏览器入口。

### 4.8 回测消融实验（可选）

```bash
uv run python scripts/14_backtest_compare.py
```

跑 4 个变体（baseline / no_q10 / q50_only / both）对比过滤器和 score 公式的边际贡献，
输出 `13_compare_nav.svg` + `14_compare_metrics.svg`。配置改动应先在这里验证再写入
`configs/model_v1.yaml`。

## 5. 时间切分

当前 95 个交易日使用工程原型切分：

```text
0  - 39：预热
40 - 64：训练
65 - 79：验证
80 - 89：测试
90 - 94：无完整 5 日标签，只做预测展示
```

配置位置：

```yaml
split:
  warmup_time_idx: [0, 39]
  train_time_idx: [40, 64]
  valid_time_idx: [65, 79]
  test_time_idx: [80, 89]
  prediction_only_time_idx: [90, 94]
```

不要随机打散时间序列。

## 6. 一键运行顺序

服务器上完整执行：

```bash
uv run python scripts/07_build_model_dataset.py
uv run python scripts/08_spectral_embedding.py --config configs/model_v1.yaml
uv run python scripts/09_train_kline_encoder.py --config configs/model_v1.yaml
uv run python scripts/10_train_tft.py --config configs/model_v1.yaml
uv run python scripts/11_predict_tft.py --config configs/model_v1.yaml --range test
uv run python scripts/12_backtest_strategy.py --config configs/model_v1.yaml
uv run python scripts/13_visualize_v1.py
uv run python scripts/14_backtest_compare.py
```

## 7. 调参建议

本数据只有 95 个交易日，建议保持小模型：

```yaml
kline_encoder:
  d_model: 24
  n_layers: 1

model:
  d_model: 48
  n_layers: 1
```

如果显存不足：

```yaml
batch_size: 256
```

如果训练过慢：

```yaml
max_epochs: 5
```

补足多年历史后可以尝试：

```yaml
model:
  encoder_length: 60
  d_model: 96
  n_heads: 4
  n_layers: 2
```

## 8. 注意事项

- V1 是研究原型，不是生产策略。
- 当前财报数值没有进入核心模型输入，避免 point-in-time 泄漏。
- 训练时保存了 scaler 和 schema，推理时必须复用。
- `model_panel_tft.parquet` 是主模型训练输入，不要手工改列名。
- 如果服务器没有 GPU，脚本仍可 CPU 跑，但速度会慢。
- 如果要直接接入 Kronos 预训练模型，需要另做依赖、许可证、模型下载和输入适配。
