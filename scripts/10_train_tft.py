#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, json
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]; _SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path: sys.path.insert(0, str(_SRC))
import argparse, pandas as pd
from dragonflow.modeling.config import load_model_config
from dragonflow.modeling.tft import train_tft_model
from dragonflow.utils.io import resolve_path, save_parquet

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config', default='configs/model_v1.yaml'); args=ap.parse_args()
    cfg=load_model_config(args.config); paths=cfg['paths']
    panel_path = paths['tft_panel'] if resolve_path(paths['tft_panel']).exists() else paths['base_panel']
    panel=pd.read_parquet(resolve_path(panel_path))
    kpath=resolve_path(paths['kline_embeddings'])
    if kpath.exists():
        kemb=pd.read_parquet(kpath)
        kline_cols=[c for c in kemb.columns if c.startswith('kline_')]
        panel=panel.drop(columns=[c for c in kline_cols if c in panel.columns], errors='ignore')
        panel=panel.merge(kemb, on=['date','stock_code','time_idx'], how='left')
        for c in kline_cols:
            if c in panel.columns: panel[c]=panel[c].fillna(0.0)
    save_parquet(panel, resolve_path(paths['tft_panel']))
    with open(resolve_path(paths['feature_schema']), 'r', encoding='utf-8') as f: schema=json.load(f)
    train_tft_model(panel, schema, cfg['model'], cfg['split'], resolve_path(paths['model_dir']))
    print('TFT训练完成')
if __name__ == '__main__': main()
