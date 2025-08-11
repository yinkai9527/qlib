#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
台灣股市量化分析範例腳本

此腳本展示如何使用 Qlib 框架進行台灣股市的量化分析，
包括數據初始化、模型訓練、回測和結果分析。
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Any

import qlib
import pandas as pd
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
from qlib.constant import REG_TW

warnings.filterwarnings("ignore")


def init_qlib_tw():
    """初始化 Qlib 台灣股市環境"""
    try:
        provider_uri = Path.home() / ".qlib" / "qlib_data" / "tw_data"
        
        if not exists_qlib_data(provider_uri):
            print(f"❌ 未找到台灣股市數據: {provider_uri}")
            print("請先執行數據準備步驟:")
            print("cd scripts/data_collector/yahoo")
            print("python collector.py download_data --region TW ...")
            return False
            
        # 初始化 Qlib 台灣股市模式
        qlib.init(
            provider_uri=str(provider_uri),
            region=REG_TW,  # 台灣股市配置
            auto_mount=True,
        )
        print(f"✅ Qlib 台灣股市模式初始化成功: {provider_uri}")
        return True
        
    except Exception as e:
        print(f"❌ Qlib 初始化失敗: {e}")
        return False


def create_taiwan_dataset():
    """創建台灣股市數據集"""
    
    # 數據處理配置
    data_handler_config = {
        "start_time": "2008-01-01",
        "end_time": "2020-08-01",
        "fit_start_time": "2008-01-01", 
        "fit_end_time": "2014-12-31",
        "instruments": "twii",  # 台灣加權指數成分股
        
        # 數據前處理
        "infer_processors": [
            {
                "class": "RobustZScoreNorm",
                "kwargs": {
                    "fields_group": "feature",
                    "clip_outlier": True
                }
            },
            {
                "class": "Fillna", 
                "kwargs": {
                    "fields_group": "feature"
                }
            }
        ],
        
        # 標籤處理
        "learn_processors": [
            {"class": "DropnaLabel"},
            {
                "class": "CSRankNorm",
                "kwargs": {
                    "fields_group": "label"
                }
            }
        ],
        
        # 預測目標: 未來2日報酬率
        "label": ["Ref($close, -2) / Ref($close, -1) - 1"]
    }
    
    # 創建 Alpha158 特徵數據處理器
    handler = Alpha158(**data_handler_config)
    
    # 創建數據集
    from qlib.data.dataset import DatasetH
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": ("2008-01-01", "2014-12-31"),
            "valid": ("2015-01-01", "2016-12-31"), 
            "test": ("2017-01-01", "2020-08-01")
        }
    )
    
    print("✅ 台灣股市數據集創建成功")
    print(f"   訓練期: 2008-01-01 ~ 2014-12-31")
    print(f"   驗證期: 2015-01-01 ~ 2016-12-31")
    print(f"   測試期: 2017-01-01 ~ 2020-08-01")
    
    return dataset


def train_taiwan_model(dataset):
    """訓練台灣股市預測模型"""
    
    # LightGBM 模型配置 - 針對台灣股市優化
    model_config = {
        "loss": "mse",
        "colsample_bytree": 0.8879,
        "learning_rate": 0.0421,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 20
    }
    
    # 創建並訓練模型
    model = LGBModel(**model_config)
    
    print("🚀 開始訓練台灣股市預測模型...")
    model.fit(dataset)
    print("✅ 模型訓練完成")
    
    return model


def backtest_taiwan_strategy(model, dataset):
    """台灣股市回測分析"""
    
    # 預測
    pred = model.predict(dataset)
    
    # 策略配置 - 台灣股市交易規則
    from qlib.contrib.strategy import TopkDropoutStrategy
    strategy = TopkDropoutStrategy(
        signal=pred,
        topk=50,     # 買入前50檔股票
        n_drop=5     # 每次調整賣出5檔
    )
    
    # 回測配置
    from qlib.backtest import backtest
    from qlib.contrib.evaluate import portfolio_analysis
    
    backtest_config = {
        "start_time": "2017-01-01",
        "end_time": "2020-08-01", 
        "account": 1000000,  # 100萬台幣初始資金
        "benchmark": "SH000001",  # 基準指數
        "exchange_kwargs": {
            "limit_threshold": 0.1,      # 台灣股市10%漲跌停
            "deal_price": "close",       # 收盤價成交
            "open_cost": 0.001425,       # 開倉成本 0.1425%  
            "close_cost": 0.001425,      # 平倉成本 0.1425%
            "min_cost": 20               # 最低手續費20元台幣
        }
    }
    
    print("📊 執行台灣股市策略回測...")
    
    # 執行回測
    from qlib.backtest import backtest as bt
    portfolio_metric_dict, indicator_dict = bt(
        strategy=strategy,
        **backtest_config
    )
    
    print("✅ 回測完成")
    return portfolio_metric_dict, indicator_dict


def analyze_taiwan_results(portfolio_metric_dict, indicator_dict):
    """分析台灣股市策略結果"""
    
    print("\n" + "="*60)
    print("🇹🇼 台灣股市量化策略回測結果")
    print("="*60)
    
    # 提取關鍵指標
    returns = portfolio_metric_dict.get("return", {})
    
    if returns:
        annual_return = returns.get("annualized_return", 0)
        sharpe_ratio = returns.get("information_ratio", 0)  
        max_drawdown = returns.get("max_drawdown", 0)
        
        print(f"📈 年化報酬率: {annual_return:.2%}")
        print(f"📊 夏普比率: {sharpe_ratio:.4f}")
        print(f"📉 最大回撤: {max_drawdown:.2%}")
    
    # 月度報酬分析
    if 'return' in portfolio_metric_dict:
        monthly_returns = portfolio_metric_dict['return']
        if isinstance(monthly_returns, pd.Series):
            monthly_mean = monthly_returns.mean()
            monthly_std = monthly_returns.std()
            
            print(f"📅 月均報酬: {monthly_mean:.2%}")
            print(f"📏 波動率: {monthly_std:.2%}")
    
    # 勝率統計
    if indicator_dict and 'IC' in indicator_dict:
        ic_series = indicator_dict['IC']
        win_rate = (ic_series > 0).mean()
        print(f"🎯 IC勝率: {win_rate:.2%}")
    
    print("="*60)


def main():
    """主要執行函數"""
    
    print("🇹🇼 歡迎使用 Qlib 台灣股市量化分析系統")
    print("="*60)
    
    # 1. 初始化環境
    if not init_qlib_tw():
        return
    
    try:
        # 2. 創建數據集
        dataset = create_taiwan_dataset()
        
        # 3. 訓練模型  
        model = train_taiwan_model(dataset)
        
        # 4. 回測策略
        portfolio_results, indicators = backtest_taiwan_strategy(model, dataset)
        
        # 5. 分析結果
        analyze_taiwan_results(portfolio_results, indicators)
        
        print("\n✅ 台灣股市量化分析完成!")
        
    except Exception as e:
        print(f"❌ 執行過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
