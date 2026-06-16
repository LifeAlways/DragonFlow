# TFT V1 特征、输入输出与整体架构设计

## 1. V1 范围

V1 目标是先用当前 `2026-01` 到 `2026-05` 的中证 2000 日线数据跑通端到端链路：

```text
日线行情
-> 特征面板
-> K 线形态编码
-> 谱聚类嵌入
-> 小型 TFT
-> 未来 5 日收益预测
-> 股票打分
-> 简化回测
```

当前阶段不追求生产级收益结论，只验证：

- 特征是否无未来函数。
- 输入输出 schema 是否稳定。
- 模型是否能训练、预测、回测。
- 后续补长历史后是否能平滑扩展。

V1 暂时不把财报数值作为核心输入。原因是当前只有 1-5 月数据，财报历史不连续，做严格 point-in-time 基本面特征样本太少。V1 可以保留行业、市值、上市年限这类相对稳定特征；财报增强放到 V2。

K 线建模参考 Kronos 的思想，但不在 V1 里训练完整金融基础模型。Kronos 的关键思路是把 OHLCV/OHLCVA K 线序列视作“市场语言”，先用 tokenizer 把连续 K 线离散化，再用自回归 Transformer 预测未来 K 线。我们的当前数据太短，不能复刻这种大规模预训练，因此 V1 只采用轻量版本：用连续 K 线形态特征 + 小型 K 线编码器生成 `kline_emb_*`，作为 TFT 的增强输入。

## 2. 数据输入

### 2.1 必需数据

```text
data/processed/stock_daily_csi2000_qfq_20260101_20260531.parquet
```

字段：

```text
date
stock_code
open
close
high
low
volume
amount
amplitude
pct_change
change_amount
turnover_rate
adjust
source
```

### 2.2 建议数据

```text
data/processed/index_daily_932000_proxy_equal_weight_20260101_20260531.parquet
data/processed/stock_info_csi2000_latest.parquet
```

指数用于市场环境和超额收益标签；个股信息用于行业、市值、上市年限。

### 2.3 暂缓核心使用的数据

```text
data/processed/fundamental_csi2000_latest.parquet
data/processed/stock_spot_snapshot_csi2000_latest.parquet
```

V1 不把财报数值作为核心模型输入。若使用快照中的 `total_market_value`，只能当作静态近似特征，不能在严谨回测中当作历史每日市值。

## 3. 训练标签

主标签使用 5 日前瞻超额收益：

```text
ret_fwd_5d_i_t = close_i_{t+5} / close_i_{t+1} - 1
index_ret_fwd_5d_t = index_close_{t+5} / index_close_{t+1} - 1
target_i_t = ret_fwd_5d_i_t - index_ret_fwd_5d_t
```

执行口径：

- `t` 日收盘后生成预测。
- 假设 `t+1` 日成交。
- 标签从 `t+1` 到 `t+5`，避免同收盘价成交泄漏。

辅助标签：

```text
ret_fwd_1d
ret_fwd_5d
index_ret_fwd_5d
direction_fwd_5d = target_i_t > 0
```

TFT 训练目标：

```text
target = excess_ret_fwd_5d
loss = quantile_loss([0.1, 0.5, 0.9])
```

模型输出：

```text
pred_q10_excess_ret_fwd_5d
pred_q50_excess_ret_fwd_5d
pred_q90_excess_ret_fwd_5d
```

## 4. 特征规划

### 4.1 标识字段

```text
stock_code
date
time_idx
```

其中 `time_idx` 是全市场统一交易日整数序号，按日期从 0 开始递增。

### 4.2 静态类别特征

```text
stock_code
industry
```

如果行业缺失：

```text
industry = "UNKNOWN"
```

V1 不建议把 `cluster_id` 作为真正静态类别，因为谱聚类会随时间变化。更合适的做法是：

- `cluster_id` 作为 time-varying known categorical。
- 谱嵌入作为 time-varying known real 或 observed real。

### 4.3 静态连续特征

```text
log_total_market_value
log_float_market_value
listing_years
```

缺失处理：

- 数值缺失先用训练集横截面中位数填充。
- 同时增加 missing flag。

```text
total_market_value_missing
float_market_value_missing
listing_years_missing
```

### 4.4 时间变化已知特征

这些特征在预测未来窗口时可以提前知道：

```text
time_idx
day_of_week
month
is_month_end
cluster_id
```

V1 中 `cluster_id` 采用最近一次谱聚类结果，并在下次重算前 forward fill。

### 4.5 时间变化观测特征：个股行情

原始或简单变换：

```text
open_to_close = close / open - 1
high_to_low = high / low - 1
close_to_high = close / high - 1
close_to_low = close / low - 1
log_volume = log1p(volume)
log_amount = log1p(amount)
turnover_rate
amplitude
pct_change
```

收益特征：

```text
ret_1d = close / close.shift(1) - 1
ret_3d = close / close.shift(3) - 1
ret_5d = close / close.shift(5) - 1
ret_10d = close / close.shift(10) - 1
ret_20d = close / close.shift(20) - 1
```

趋势特征：

```text
ma5_gap = close / rolling_mean(close, 5) - 1
ma10_gap = close / rolling_mean(close, 10) - 1
ma20_gap = close / rolling_mean(close, 20) - 1
ma5_ma20_gap = rolling_mean(close, 5) / rolling_mean(close, 20) - 1
```

波动特征：

```text
vol_5d = rolling_std(ret_1d, 5)
vol_10d = rolling_std(ret_1d, 10)
vol_20d = rolling_std(ret_1d, 20)
```

流动性特征：

```text
amount_mean_5d
amount_mean_20d
amount_zscore_20d
turnover_mean_5d
turnover_mean_20d
turnover_zscore_20d
```

价格位置：

```text
price_position_20d =
    (close - rolling_min(low, 20)) /
    (rolling_max(high, 20) - rolling_min(low, 20))
```

异动标记：

```text
is_big_up = pct_change >= 5
is_big_down = pct_change <= -5
is_limit_up_like = pct_change >= 9.5
is_limit_down_like = pct_change <= -9.5
```

### 4.6 时间变化观测特征：市场环境

由中证 2000 代理指数和全市场横截面计算：

```text
index_ret_1d
index_ret_5d
index_ret_20d
index_vol_10d
index_vol_20d
market_ret_mean_1d
market_ret_std_1d
market_breadth
market_amount_sum
market_turnover_mean
```

其中：

```text
market_breadth = 当日 pct_change > 0 的股票占比
market_ret_std_1d = 当日股票收益横截面标准差
```

### 4.7 谱聚类特征

V1 谱聚类只使用过去窗口数据，不使用未来。

输出字段：

```text
cluster_id
spectral_emb_1
spectral_emb_2
spectral_emb_3
spectral_emb_4
spectral_emb_5
spectral_emb_6
spectral_emb_7
spectral_emb_8
```

同簇聚合特征：

```text
cluster_ret_mean_1d
cluster_ret_mean_5d
cluster_amount_mean
cluster_turnover_mean
stock_ret_minus_cluster_1d
stock_ret_minus_cluster_5d
```

V1 参数：

```text
graph_window = 40
embedding_dim = 8
n_neighbors = 30
k_min = 8
k_max = 20
refit_every_n_days = 5
```

### 4.8 Kronos 启发的 K 线建模特征

Kronos 启发点：

- 不只看收益率，而是完整建模 `open/high/low/close/volume/amount` 的联合形态。
- 把连续 K 线标准化后交给序列模型，让模型学习“形态语言”。
- 输出不一定直接是交易信号，也可以是未来 K 线预测、波动预测或隐藏表示。

V1 不做离散 tokenizer，先做连续版 K 线编码，避免在 95 天数据上训练 tokenizer 过拟合。

每根 K 线转换为以下相对形态：

```text
k_body = close / open - 1
k_range = high / low - 1
k_upper_shadow = high / max(open, close) - 1
k_lower_shadow = min(open, close) / low - 1
k_close_pos = (close - low) / (high - low)
k_gap = open / close.shift(1) - 1
k_volume_chg = log1p(volume) - log1p(volume.shift(1))
k_amount_chg = log1p(amount) - log1p(amount.shift(1))
```

再增加窗口级 K 线形态统计：

```text
k_body_mean_5d
k_body_std_5d
k_range_mean_5d
k_range_std_5d
k_upper_shadow_mean_5d
k_lower_shadow_mean_5d
k_close_pos_mean_5d
k_gap_abs_mean_5d
```

K 线编码器输入：

```text
[t-29, ..., t] 的 K 线形态序列
```

K 线编码器输出：

```text
kline_emb_1
kline_emb_2
kline_emb_3
kline_emb_4
kline_pred_ret_1d
kline_pred_range_1d
kline_pred_vol_5d
```

其中：

- `kline_emb_*` 是 K 线序列隐藏表示。
- `kline_pred_ret_1d` 是 K 线编码器辅助预测的下一日收益。
- `kline_pred_range_1d` 是辅助预测的下一日振幅。
- `kline_pred_vol_5d` 是辅助预测的未来 5 日波动。

V1 小模型配置：

```yaml
kline_encoder:
  input_length: 30
  input_features:
    - k_body
    - k_range
    - k_upper_shadow
    - k_lower_shadow
    - k_close_pos
    - k_gap
    - k_volume_chg
    - k_amount_chg
  d_model: 24
  n_heads: 2
  n_layers: 1
  dim_feedforward: 48
  dropout: 0.20
  output_dim: 4
```

训练方式有两种：

1. 端到端联合训练：K 线编码器作为 TFT 前置模块，跟主目标 `excess_ret_fwd_5d` 一起训练。
2. 两阶段训练：先训练 K 线编码器预测 `ret_fwd_1d / range_fwd_1d / vol_fwd_5d`，再冻结或半冻结编码器，把 `kline_emb_*` 合入 TFT。

V1 建议先用两阶段训练。原因是数据短，联合训练不稳定；两阶段可以先确认 K 线编码器是否学到有效形态。


## 5. TFT 端到端输入输出

### 5.1 单个训练样本

一个样本定义为：

```text
股票 i，在日期 t，过去 encoder_length 天的特征序列 -> 预测 t 之后 5 日超额收益
```

V1：

```text
encoder_length = 30
prediction_length = 5
```

输入窗口：

```text
[t-29, ..., t]
```

标签：

```text
excess_ret_fwd_5d_i_t
```

### 5.2 输入 schema

静态类别：

```text
stock_code
industry
```

静态连续：

```text
log_total_market_value
log_float_market_value
listing_years
total_market_value_missing
float_market_value_missing
listing_years_missing
```

时间变化已知类别：

```text
cluster_id
```

时间变化已知连续：

```text
time_idx
day_of_week
month
is_month_end
```

时间变化观测连续：

```text
open_to_close
high_to_low
close_to_high
close_to_low
log_volume
log_amount
turnover_rate
amplitude
pct_change
ret_1d
ret_3d
ret_5d
ret_10d
ret_20d
ma5_gap
ma10_gap
ma20_gap
ma5_ma20_gap
vol_5d
vol_10d
vol_20d
amount_mean_5d
amount_mean_20d
amount_zscore_20d
turnover_mean_5d
turnover_mean_20d
turnover_zscore_20d
price_position_20d
is_big_up
is_big_down
is_limit_up_like
is_limit_down_like
index_ret_1d
index_ret_5d
index_ret_20d
index_vol_10d
index_vol_20d
market_ret_mean_1d
market_ret_std_1d
market_breadth
market_amount_sum
market_turnover_mean
spectral_emb_1
spectral_emb_2
spectral_emb_3
spectral_emb_4
spectral_emb_5
spectral_emb_6
spectral_emb_7
spectral_emb_8
cluster_ret_mean_1d
cluster_ret_mean_5d
cluster_amount_mean
cluster_turnover_mean
stock_ret_minus_cluster_1d
stock_ret_minus_cluster_5d
k_body
k_range
k_upper_shadow
k_lower_shadow
k_close_pos
k_gap
k_volume_chg
k_amount_chg
k_body_mean_5d
k_body_std_5d
k_range_mean_5d
k_range_std_5d
k_upper_shadow_mean_5d
k_lower_shadow_mean_5d
k_close_pos_mean_5d
k_gap_abs_mean_5d
kline_emb_1
kline_emb_2
kline_emb_3
kline_emb_4
kline_pred_ret_1d
kline_pred_range_1d
kline_pred_vol_5d
```

目标：

```text
excess_ret_fwd_5d
```

### 5.3 输出 schema

预测输出表：

```text
date
stock_code
pred_q10_excess_ret_fwd_5d
pred_q50_excess_ret_fwd_5d
pred_q90_excess_ret_fwd_5d
prediction_version
model_version
```

交易打分（V1 默认 `score_formula: q50`，直接用 alpha 排序）：

```text
alpha = pred_q50_excess_ret_fwd_5d
score = alpha
```

历史上也尝试过 `q50 / uncertainty`：

```text
uncertainty = pred_q90_excess_ret_fwd_5d - pred_q10_excess_ret_fwd_5d
score_alt = alpha / (uncertainty + 1e-6)
```

但这会偏好不确定性低（区间窄）的股票，把 alpha 信号埋掉；
在 V1 数据上验证：用 `q50` 直选比 `q50/uncertainty` 多 +1.4pp 收益。
代码里通过 `cfg["backtest"]["score_formula"]` 切换。

可选风险过滤：

```text
原始设计：只买 pred_q10_excess_ret_fwd_5d > -0.03 的股票
```

**V1 当前默认关闭**（`q10_floor: -1.0`）。原因：模型 predicted q10
全样本最大值仅 -0.028（系统性偏负，校准图证实），该过滤会把候选池
从 1963 直接砍到 2 只，导致回测亏 11.8%。详见 `scripts/14_backtest_compare.py`
消融实验和 `data/processed/viz_v1/13_compare_nav.svg`。

V2 在长历史下若 q10 校准变好，可重新启用。

## 6. 训练/验证/测试切分

当前只有 95 个交易日，V1 采用偏工程验证的切分。

按交易日序号切：

```text
0  - 39：特征预热，不参与监督训练评估
40 - 64：训练
65 - 79：验证
80 - 89：测试
90 - 94：没有完整 5 日标签，只用于未来预测展示
```

原因：

- `encoder_length = 30`，前 30 天本来就无法形成完整样本。
- 谱聚类 `graph_window = 40`，前 40 天用于建图预热。
- 5 日前瞻标签会吃掉最后 5 天。

正式数据补齐后改为：

```text
前 70%：训练
中间 15%：验证
最后 15%：测试
```

严禁随机打散日期。

## 7. 小型 TFT 参数

当前数据很短，模型必须小，否则过拟合和训练不稳定。

V1 参数：

```yaml
tft:
  encoder_length: 30
  prediction_length: 5
  hidden_size: 16
  lstm_layers: 1
  attention_head_size: 2
  hidden_continuous_size: 8
  dropout: 0.20
  batch_size: 256
  max_epochs: 10
  learning_rate: 0.001
  gradient_clip_val: 0.1
  quantiles: [0.1, 0.5, 0.9]
```

如果直接实现 Transformer 而不是 `pytorch-forecasting` 的 TFT，可用更小配置：

```yaml
tiny_transformer:
  d_model: 32
  n_heads: 2
  n_layers: 1
  dim_feedforward: 64
  dropout: 0.20
  pooling: "last"
```

当前阶段不要堆很多 transformer block。95 天数据支撑不了深模型。

如果启用 K 线编码器，TFT 本体仍然保持小模型。不要同时加深 K 线编码器和 TFT，否则样本量不够：

```yaml
model_stack_v1:
  kline_encoder_layers: 1
  tft_lstm_layers: 1
  tft_attention_heads: 2
  train_mode: "two_stage"
```

## 8. 整体工程架构

### 8.1 模块划分

```text
src/dragonflow/modeling/
  dataset.py              # 构建 TFT 面板和 schema
  targets.py              # 前瞻收益标签
  market_features.py      # 指数和全市场特征
  technical_features.py   # 个股 rolling 特征
  kline_encoder.py        # Kronos 启发的轻量 K 线编码器
  tft.py                  # TFT 模型训练/预测封装
  schema.py               # 特征 schema 定义

src/dragonflow/analysis/
  spectral_embedding.py   # 股票图、谱嵌入、cluster_id

src/dragonflow/backtest/
  portfolio.py            # 股票打分转权重
  execution.py            # 成本和成交假设
  metrics.py              # 回测指标
```

### 8.2 脚本顺序

```text
scripts/07_build_model_dataset.py
scripts/08_spectral_embedding.py
scripts/09_train_kline_encoder.py
scripts/10_train_tft.py
scripts/11_predict_tft.py
scripts/12_backtest_strategy.py
```

### 8.3 中间产物

```text
data/processed/model_panel_base.parquet
data/processed/spectral_embeddings.parquet
data/processed/kline_embeddings.parquet
data/processed/model_panel_tft.parquet
data/processed/tft_predictions.parquet
data/processed/backtest_nav.parquet
data/processed/backtest_metrics.json
```

### 8.4 模型产物

```text
models/tft_csi2000_v1/
  model.ckpt
  kline_encoder.ckpt
  feature_schema.json
  feature_config.yaml
  categorical_encoders.pkl
  numeric_scaler.pkl
  spectral_config.yaml
  train_valid_test_split.json
```

预训练模型后续推理必须复用同一套 schema、编码器、scaler 和特征构造逻辑。

## 9. V1 组合规则

先用简单多头组合：

```text
调仓频率：每 5 个交易日
选股数量：top 80
权重：等权
单票上限：2%
最低流动性：amount_mean_20d >= 30000000
交易成本：买入 10 bps，卖出 10 bps，滑点 5 bps
```

过滤（V1 当前默认）：

```text
score (= pred_q50_excess_ret_fwd_5d) 排名前 80
amount_mean_20d >= 30000000
q10_floor = -1.0  # 关闭，详见 5.3 节
```

V1 不做复杂优化器，先把信号和回测链路跑通。

## 10. V2 扩展

补足历史后再加入：

- point-in-time 财报特征。
- 日频估值特征。
- 更长 encoder，例如 60 或 90。
- 更大的 TFT，例如 `hidden_size = 32`、`attention_head_size = 4`。
- 更严谨的 walk-forward retrain。
- 行业/簇中性约束。
- 组合优化器。

V2 财报特征必须满足：

```text
announcement_date <= prediction_date
```

训练时用了什么财报字段，推理时就必须按同一口径生成同名字段。

## 11. 参考

- Kronos 论文：`Kronos: A Foundation Model for the Language of Financial Markets`。
- Kronos GitHub：`https://github.com/shiyu-coder/Kronos`。

V1 只参考其 K 线序列建模思想，不直接依赖 Kronos 权重或 tokenizer。后续如果要直接接入预训练 Kronos，需要单独处理依赖、模型下载、许可证和推理成本。
