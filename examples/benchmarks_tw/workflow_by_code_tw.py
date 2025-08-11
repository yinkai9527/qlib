#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
台灣股市工作流範例 - 使用 workflow_by_code 方式

此範例展示如何透過程式碼方式建立完整的台灣股市量化研究工作流，
而非使用 YAML 配置檔案。
"""

import sys
import warnings
from pathlib import Path

import qlib
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.constant import REG_TW

warnings.filterwarnings("ignore")

def run_taiwan_workflow():
    """執行台灣股市完整工作流"""
    
    print("🇹🇼 台灣股市量化研究工作流")
    print("="*50)
    
    # 1. 初始化 Qlib
    provider_uri = Path.home() / ".qlib" / "qlib_data" / "tw_data"
    qlib.init(provider_uri=str(provider_uri), region=REG_TW)
    print("✅ Qlib 台灣模式初始化完成")
    
    # 2. 模型配置
    model_config = {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
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
    }
    
    # 3. 數據集配置
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": "2008-01-01",
                    "end_time": "2020-08-01",
                    "fit_start_time": "2008-01-01",
                    "fit_end_time": "2014-12-31",
                    "instruments": "twii",
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
                    "learn_processors": [
                        {"class": "DropnaLabel"},
                        {
                            "class": "CSRankNorm", 
                            "kwargs": {
                                "fields_group": "label"
                            }
                        }
                    ],
                    "label": ["Ref($close, -2) / Ref($close, -1) - 1"]
                }
            },
            "segments": {
                "train": ("2008-01-01", "2014-12-31"),
                "valid": ("2015-01-01", "2016-12-31"),
                "test": ("2017-01-01", "2020-08-01")
            }
        }
    }
    
    # 4. 投資組合分析配置
    port_analysis_config = {
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {
                "signal": "<PRED>",
                "topk": 50,
                "n_drop": 5
            }
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01", 
            "account": 1000000,
            "benchmark": "SH000001",
            "exchange_kwargs": {
                "limit_threshold": 0.1,
                "deal_price": "close", 
                "open_cost": 0.001425,
                "close_cost": 0.001425,
                "min_cost": 20
            }
        }
    }
    
    # 5. 建立任務
    with R.start(experiment_name="Taiwan_Stock_LightGBM_Alpha158"):
        print("📊 建立模型...")
        model = init_instance_by_config(model_config)
        
        print("📈 建立數據集...")
        dataset = init_instance_by_config(dataset_config)
        
        print("🚀 訓練模型...")
        model.fit(dataset)
        
        print("📊 記錄訓練結果...")
        R.save_objects(model=model, dataset=dataset)
        
        # 信號記錄
        sr = SignalRecord(model, dataset, R.get_recorder())
        sr.generate()
        
        # 投資組合分析記錄
        par = PortAnaRecord(R.get_recorder(), port_analysis_config, "day")
        par.generate()
        
    print("✅ 台灣股市工作流執行完成!")
    print("📁 結果已保存到 MLflow")


if __name__ == "__main__":
    run_taiwan_workflow()
