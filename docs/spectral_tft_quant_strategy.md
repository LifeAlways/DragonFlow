# 谱聚类嵌入 + TFT 量化交易方案

## 1. 目标

基于中证 2000 股票池，构建一套“谱聚类嵌入 + Temporal Fusion Transformer（TFT）”的价格预测与交易系统。谱聚类负责刻画股票之间的横截面关系，TFT 负责建模多股票共享的时间序列预测问题，最终把预测结果转成可回测、可风控、可迭代的组合交易信号。

当前仓库已经具备以下数据：

- 个股日线：`data/processed/stock_daily_csi2000_qfq_20260101_20260531.parquet`
- 中证 2000 等权代理指数：`data/processed/index_daily_932000_proxy_equal_weight_20260101_20260531.parquet`
- 个股基础信息：`data/processed/stock_info_csi2000_latest.parquet`
- 由最后一根日线合成的截面快照：`data/processed/stock_spot_snapshot_csi2000_latest.parquet`
- 财务报表数据：`data/processed/fundamental_csi2000_latest.parquet`

关键约束：当前日线窗口只有 `2026-01-05` 到 `2026-05-29`，共 95 个交易日、2000 只股票。这个长度足够做端到端原型，但不足以训练生产级 TFT。实盘研究建议至少补 2-3 年日频历史，最好补 5 年。

## 2. 交易假设

策略建立在三个假设上：

1. 个股未来收益受自身时间序列状态影响，包括趋势、反转、波动、成交、流动性和近期跳空。
2. 股票之间存在可学习的横截面关系，同一行业、同一流动性状态、相似基本面或高相关收益的股票，容易出现联动和轮动。
3. 市场状态会影响信号胜率。中证 2000 的趋势、波动、成交和横截面离散度，决定模型信号应该进攻还是防守。

谱聚类把股票关系图转成低维嵌入和簇标签；TFT 再结合个股历史序列、静态特征、市场状态和谱嵌入，预测未来短周期收益。

## 3. 预测目标

主目标建议使用 5 日前瞻收益：

```text
y_{i,t}^{5d} = close_{i,t+5} / close_{i,t+1} - 1
```

用 `t+1` 作为入场价格参考，是为了避免“当天收盘后出信号、却按当天收盘成交”的执行泄漏。

建议同时构造这些标签：

- `ret_fwd_1d`：下一交易日收益，用于执行诊断。
- `ret_fwd_5d`：主 alpha 目标。
- `ret_fwd_10d`：更慢的辅助信号。
- `excess_ret_fwd_5d`：个股 5 日前瞻收益减去中证 2000 代理指数 5 日前瞻收益。
- `direction_fwd_5d`：方向标签，`ret_fwd_5d > 0`。

建模方式优先选择分位数回归：

- 预测 `0.1 / 0.5 / 0.9` 三个分位数。
- 用中位数作为预期收益。
- 用低分位数作为下行风险过滤。
- 用高低分位差衡量不确定性。

比起单点回归，分位数回归更适合组合构建和风控。

## 4. 特征体系

### 4.1 已知动态特征

这些特征在预测时已知：

- 交易日序号、星期、月份、月末标记。
- 股票代码 `stock_code`。
- 最近一次谱聚类得到的 `cluster_id`。
- 行业、总市值分桶、流动性分桶、上市年限分桶。

### 4.2 观测动态特征

这些特征必须只使用 `t` 日及以前的数据：

- 原始行情：`open`、`high`、`low`、`close`、`volume`、`amount`、`amplitude`、`pct_change`、`change_amount`、`turnover_rate`。
- 收益：1 日、3 日、5 日、10 日、20 日收益。
- 波动：5 日、10 日、20 日滚动标准差。
- 趋势：`close / MA5 - 1`、`close / MA10 - 1`、`close / MA20 - 1`、`MA5 / MA20 - 1`。
- 流动性：成交额对数、成交额滚动 z-score、换手率滚动 z-score。
- 价格位置：收盘价在 20 日最高最低区间中的位置。
- 日内形态：`(close - open) / open`、`(high - low) / close`、上下影线比例。
- 异动行为：涨停/跌停标记、大阳线/大阴线标记。
- 市场环境：中证 2000 代理指数收益、波动、市场宽度、平均换手、横截面收益离散度。
- 同簇环境：簇平均收益、簇平均换手、簇动量、个股收益减同簇平均收益。

### 4.3 静态特征

来自个股信息和财务报表：

- 行业。
- 上市年限。
- 总市值、流通市值。
- 总市值对数。
- 净利润、营收、营收同比、净利润同比。
- 资产负债率。
- 经营现金流、净现金流。
- 可进一步派生盈利质量和资产负债质量指标。

注意：当前 `fundamental_csi2000_latest.parquet` 是利润表、资产负债表、现金流量表的部分纵向合并结果，不是一股一行。进入模型前需要先按 `stock_code` 聚合成一股一行。静态字段可取 `first`，数值字段建议取非空值优先的 `last non-null` 或按报表类型拆开再合并。

## 5. 谱聚类嵌入

### 5.1 股票关系图

对每个建模日期 `t`，使用过去一段窗口构建股票图，例如过去 60 个交易日：

```text
W_corr = max(corr(return_i, return_j), 0)
W_industry = 1 if same industry else 0
W_liquidity = exp(-abs(log_amount_i - log_amount_j))
W_style = cosine_similarity(style_features_i, style_features_j)

W = 0.55 * W_corr + 0.20 * W_industry + 0.15 * W_liquidity + 0.10 * W_style
```

图需要稀疏化：

- 每只股票保留权重最高的 30-50 个邻居。
- 对称化：`W = max(W, W.T)`。
- 删除低于阈值的边，例如权重小于 `0.05` 的边。

### 5.2 谱嵌入

使用归一化图拉普拉斯矩阵：

```text
L_sym = I - D^{-1/2} W D^{-1/2}
```

取最小的非平凡特征向量作为股票嵌入：

```text
spectral_emb_1 ... spectral_emb_d
```

推荐维度：

- 当前原型：`d = 8`
- 正式研究：`d = 16`

之后在嵌入空间做聚类：

- 如果只要标签，可以直接用 `SpectralClustering`。
- 如果要同时保存嵌入和标签，建议先算特征向量，再对嵌入跑 `KMeans`。

聚类数选择：

- 搜索 `k = 8..30`。
- 结合 silhouette、eigengap 和最小簇规模选择。
- 每个簇建议至少 20 只股票，避免簇太碎导致组合约束不稳定。

### 5.3 防止未来函数

日期 `t` 的图只能使用 `t` 及以前的信息。不能用完整 2026-01 到 2026-05 的全样本图，再把嵌入喂给早期日期，否则会把未来相关性泄漏进模型。

在当前 95 天数据上：

- 可以在至少 40 个交易日后使用 expanding graph。
- 或者只做一次全样本图作为探索性可视化，并明确标注“不可交易、存在未来信息”。

生产研究中：

- 每周或每月重算一次图。
- 两次重算之间向前填充簇标签和谱嵌入。

## 6. TFT 模型设计

### 6.1 模型形态

使用一个跨股票的全局 TFT，而不是每只股票单独训练一个模型。全局模型能学习市场共同模式，再通过股票静态特征、行业、簇标签和谱嵌入区分不同股票。

推荐实现：

- 快速研究：`pytorch-forecasting`
- 后续精细控制：原生 PyTorch

### 6.2 序列设置

正式研究推荐：

```text
encoder_length = 40 或 60 个交易日
prediction_length = 5 个交易日
target = ret_fwd_5d 或 excess_ret_fwd_5d
group_id = stock_code
time_idx = 全市场统一交易日整数序号
```

当前 95 天原型建议：

```text
encoder_length = 30
prediction_length = 5
```

这只能验证流程是否跑通。补足 2-3 年历史后，再把 encoder 拉到 60 日以上。

### 6.3 TFT 输入

静态类别特征：

- `stock_code`
- `industry`
- `cluster_id`
- `market_cap_bucket`
- `liquidity_bucket`

静态连续特征：

- 总市值对数。
- 上市年限。
- 财务质量指标。

已知时间变化特征：

- `time_idx`
- 日历特征。
- 最近一次有效 `cluster_id`，如果将其视为未来预测期已知。

观测时间变化特征：

- OHLCV 派生特征。
- 市场环境特征。
- 谱聚类嵌入。
- 同簇聚合特征。

### 6.4 损失函数和评价指标

训练损失：

- 分位数损失，分位点为 `[0.1, 0.5, 0.9]`。

验证指标：

- IC：预测值和真实收益的 Spearman 相关。
- 按日期计算 Rank IC。
- 方向准确率。
- Top decile 减 bottom decile 的收益差。
- 分位数校准：真实收益落在预测 0.1 到 0.9 区间内的比例。
- 扣除换手成本后的多空价差。

模型优劣主要看 Rank IC 和组合回测，不应主要看 RMSE。

## 7. 信号生成

每个调仓日 `t`，对每只股票预测 5 日超额收益：

```text
alpha_i = pred_q50_i
downside_i = pred_q10_i
uncertainty_i = pred_q90_i - pred_q10_i
score_i = alpha_i                         # V1 当前默认
score_i_alt = alpha_i / (uncertainty_i + 1e-6)   # 原方案，已弃用
```

V1 在 95 天数据下验证：`q50` 直选比 `q50/uncertainty` 多 +1.4pp 收益。
后者会把 score 主导权交给 `uncertainty`（相关性 corr(score,unc) ≈ 0.74），
偏好"区间窄"的股票而非 alpha 高的股票。
长历史 + 校准更好的 V2 模型可重新评估。

过滤条件：

- 生产环境中剔除历史不足 60 个交易日的股票。
- 剔除停牌、缺少次日开盘或缺少可执行价格的股票。
- 剔除过去 20 日平均成交额最低的 20%。
- 剔除近期异常波动且执行质量差的股票。
- 如果后续补充 ST、新股等字段，应剔除 ST 和上市过短股票。

簇内相对强弱修正：

```text
final_score_i = 0.70 * zscore_global(score_i)
              + 0.30 * zscore_within_cluster(score_i)
```

这样既保留全市场强信号，也避免单一热门簇占满组合。

## 8. 组合构建

### 8.1 多头策略版本

优先实现多头版本，更符合 A 股实际约束：

- 每 5 个交易日调仓一次。
- 持有 `final_score` 最高的 50-100 只股票。
- 等权或按波动率倒数加权。
- 单票最大权重：2%。
- 单簇最大权重：20%。
- 单行业最大权重：25%。
- 过去 20 日平均成交额门槛：例如 3000 万到 5000 万人民币。
- 目标总仓位：80%-100%，弱市降到 50%-70%。

市场状态仓位控制：

```text
if index_20d_return < 0 and index_20d_volatility high:
    gross_exposure = 50%-70%
else:
    gross_exposure = 90%-100%
```

### 8.2 多空研究版本

多空版本只建议作为 alpha 诊断：

- 做多最高分位股票。
- 做空最低分位股票。
- 金额中性。
- 行业或簇中性。
- 明确写入融券可得性和做空成本假设。

A 股宽基做空约束较强，所以多空版本不是默认交易产品。

### 8.3 权重计算

初版用简单权重即可：

```text
raw_weight_i = max(final_score_i, 0)
weight_i = raw_weight_i / sum(raw_weight)
```

然后依次施加：

- 单票权重上限。
- 单簇权重上限。
- 单行业权重上限。
- 流动性容量限制：持仓金额不超过过去 20 日平均成交额的 5%-10%。
- 换手限制：每次调仓单边换手不超过 30%-50%。

如果约束较多，可以升级为二次优化：

```text
maximize alpha' w - lambda_risk * w' Sigma w - lambda_turnover * |w - w_prev|
subject to caps and exposure constraints
```

第一版不必复杂化，贪心裁剪加归一化足够。

## 9. 回测设计

### 9.1 走步切分

补足历史后的正式研究：

- 训练集：前 60%-70% 日期。
- 验证集：中间 15%-20% 日期。
- 测试集：最后 15%-20% 日期。
- 每月或每季度滚动重训。

当前 95 天原型：

- 第 1-50 个交易日：特征预热和初始训练。
- 第 51-70 个交易日：验证。
- 第 71-95 个交易日：测试演示。

这个切分只能验证工程链路，不足以证明策略有效。

### 9.2 执行假设

默认使用保守假设：

- `t` 日收盘后生成信号。
- `t+1` 日按开盘价、VWAP 代理价或下一可成交价交易。
- 如果只有日线，优先用下一日开盘；没有开盘价时才退化到下一日收盘，并在报告中说明。
- 买入成本：10 bps。
- 卖出成本：10 bps。
- 如适用，加入卖出印花税。
- 滑点：至少 5 bps，也可以按成交额参与率估算冲击成本。

### 9.3 回测指标

组合层面：

- 年化收益。
- 年化波动。
- Sharpe。
- 最大回撤。
- Calmar。
- 调仓胜率。
- 平均换手。
- 平均持股数量。
- 行业和簇暴露。
- 基于成交额参与率的容量估计。

Alpha 诊断：

- 按日或按调仓日计算 Rank IC。
- IC 均值、标准差、ICIR。
- 分组收益曲线。
- 多空价差。
- 分位数预测校准。
- 按簇、行业、流动性、市值、市场状态拆分表现。

## 10. 风控

交易前控制：

- 价格或成交量缺失则不交易。
- 过去 20 日平均成交额低于门槛则不买入。
- 预测下行分位数低于阈值则不买入。
- 控制行业和簇集中度。
- 控制单票权重和调仓换手。

持仓中控制：

- 组合回撤超过 5%-8% 时降低总仓位。
- 调仓时如果个股预测排名跌到中位数以下则退出。
- 指数波动显著抬升时按波动率缩仓。

模型风险控制：

- 最近滚动窗口验证 IC 转负时暂停模型。
- 扣成本后的分组价差连续多次为负时暂停模型。
- 监控特征漂移和谱嵌入漂移。

## 11. 仓库落地路线

### 阶段 1：构建建模面板

新增：

- `src/dragonflow/modeling/dataset.py`
- `src/dragonflow/modeling/targets.py`
- `src/dragonflow/modeling/market_features.py`
- `scripts/07_build_model_dataset.py`

输出：

- `data/processed/model_panel.parquet`
- `data/processed/static_stock_features.parquet`
- `data/processed/forward_returns.parquet`

主要任务：

- 把 `date` 转成 datetime，并创建全市场统一 `time_idx`。
- 构造无未来函数的滚动技术特征。
- 用 `groupby("stock_code").shift(-horizon)` 构造前瞻收益。
- 把基本面聚合成每只股票一行的静态特征。

### 阶段 2：谱聚类嵌入

新增：

- `src/dragonflow/analysis/spectral_embedding.py`
- `scripts/08_spectral_embedding.py`

输出：

- `data/processed/spectral_embeddings.parquet`
- `data/processed/spectral_clusters.parquet`
- `data/processed/cluster_peer_features.parquet`

主要任务：

- 构建滚动稀疏股票图。
- 计算谱嵌入和簇标签。
- 按日期生成同簇聚合特征。
- 按图重算日期缓存结果。

### 阶段 3：TFT 训练和预测

新增：

- `src/dragonflow/modeling/tft.py`
- `scripts/09_train_tft.py`
- `scripts/10_predict_tft.py`

候选依赖：

- `torch`
- `pytorch-lightning`
- `pytorch-forecasting`

输出：

- `models/tft_csi2000/`
- `data/processed/tft_predictions.parquet`

主要任务：

- 构建 `TimeSeriesDataSet`。
- 使用分位数损失训练全局 TFT。
- 保存模型 checkpoint、特征 scaler 和类别 encoder。
- 只输出样本外预测。

### 阶段 4：组合和回测

新增：

- `src/dragonflow/backtest/portfolio.py`
- `src/dragonflow/backtest/execution.py`
- `src/dragonflow/backtest/metrics.py`
- `scripts/11_backtest_strategy.py`

输出：

- `data/processed/backtest_orders.parquet`
- `data/processed/backtest_positions.parquet`
- `data/processed/backtest_nav.parquet`
- `data/processed/backtest_metrics.json`

主要任务：

- 把 TFT 预测转成股票分数。
- 应用过滤器和组合约束。
- 生成目标权重和调仓订单。
- 加入交易成本模拟净值。
- 输出指标和诊断报告。

## 12. 当前数据下的最小原型参数

```yaml
data:
  start_date: "2026-01-05"
  end_date: "2026-05-29"
  universe: "CSI2000"

target:
  horizon_days: 5
  target_col: "excess_ret_fwd_5d"

features:
  rolling_windows: [3, 5, 10, 20]
  encoder_length: 30
  prediction_length: 5

spectral:
  graph_window: 40
  refit_frequency: "5B"
  embedding_dim: 8
  n_neighbors: 30
  k_min: 8
  k_max: 20

tft:
  hidden_size: 32
  attention_head_size: 4
  dropout: 0.15
  batch_size: 256
  max_epochs: 20
  quantiles: [0.1, 0.5, 0.9]

portfolio:
  rebalance_days: 5
  top_n: 80
  max_stock_weight: 0.02
  max_cluster_weight: 0.20
  max_industry_weight: 0.25
  min_avg_amount_20d: 30000000
  q10_floor: -1.0          # V1 关闭下行过滤：模型 q10 系统性偏负，原 -0.03 会把池子打没
  score_formula: q50       # 用 alpha 直选；q50_over_uncertainty 偏好低不确定股票
  buy_cost_bps: 10
  sell_cost_bps: 10
  slippage_bps: 5
```

V1 实测（95 天 + 上述配置）：total_return +1.50%，Sharpe +0.26，MaxDD -5.26%。
原方案（q10>-0.03 + q50/uncertainty）在同数据上是 -11.81% / -3.50 / -11.81%。
两个改动里 `q10_floor` 贡献最大（单改即从 -11.81% → +0.11%），`score_formula`
在大池子下再加 +1.4pp。详见 `data/processed/viz_v1/13_compare_nav.svg`
和 `compare_metrics.json`。

## 13. 策略通过标准

不要只因为模型 loss 好看就认为策略可用。至少满足这些条件后，才考虑进入更严肃的模拟盘：

- 样本外 Rank IC 均值为正且稳定。
- 扣成本后的 ICIR 明显大于 0。
- 最高分组长期跑赢最低分组。
- 多头组合扣成本后跑赢中证 2000 代理指数。
- 最大回撤相对收益可接受。
- 收益不是集中来自单一簇、单一行业或极短日期片段。
- 特征重要性和注意力诊断具备经济含义。

## 14. 主要失败模式

- 历史太短，当前 95 天样本极易过拟合。
- 使用全样本图做谱嵌入，造成未来信息泄漏。
- 财报数据没有按公告日期处理，造成基本面泄漏。
- 换手过高，交易成本吃掉 alpha。
- 模型实际学到的是流动性和市值暴露，不是有效选股 alpha。
- 聚类标签变化太频繁，组合约束不稳定。
- 回测用 close-to-close 收益，却假设可以同收盘价成交。

## 15. 推荐下一步

先实现阶段 1：构建 `model_panel.parquet`。也就是把当前日线、指数、基本面和静态信息整理成一个无未来函数的监督学习面板，并生成 5 日前瞻收益标签。

这个面板建好后，再接谱聚类嵌入和 TFT 会比较干净；不要一开始就把数据清洗、图构建、模型训练、信号生成和回测全部写进一个大脚本。
