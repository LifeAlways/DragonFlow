"""DragonFlow-KronosGraph V1 流水线产物的可视化。

与现有 `dragonflow.viz` 中给 features/daily 用的函数隔离，专门处理：
    tft_predictions.parquet
    backtest_nav.parquet / backtest_positions.parquet
    spectral_embeddings.parquet
    kline_embeddings.parquet
    model_panel_tft.parquet
"""
