#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
台灣股市範例執行器

使用方式:
python run_examples.py -e basic      # 基礎範例
python run_examples.py -e workflow   # 完整工作流程  
python run_examples.py -e yaml       # YAML配置範例
python run_examples.py -e all        # 所有範例
"""

import argparse
import sys
import subprocess
from pathlib import Path

def run_basic_example():
    """運行基礎範例"""
    print("🔸 執行台灣股市基礎範例...")
    
    basic_code = '''
import qlib
from qlib.data import D
from qlib.constant import REG_TW

# 初始化台灣股市模式
provider_uri = "~/.qlib/qlib_data/tw_data"
qlib.init(provider_uri=provider_uri, region=REG_TW)

print("✅ Qlib 台灣模式初始化成功")

# 測試數據載入
print("📊 測試交易日曆:")
calendar = D.calendar(start_time='2020-01-01', end_time='2020-01-10', freq='day')
print(f"   2020年1月前10個交易日: {len(calendar)} 天")

print("🏢 測試股票清單:")
instruments = D.instruments('twii')
stocks = D.list_instruments(
    instruments=instruments,
    start_time='2020-01-01', 
    end_time='2020-01-10',
    as_list=True
)[:5]
print(f"   台灣加權指數成分股(前5檔): {stocks}")

print("📈 測試股價數據:")
data = D.features(
    ['TW2330'], 
    ['$close', '$volume'], 
    start_time='2020-01-01', 
    end_time='2020-01-10'
)
print(data)

print("✅ 基礎範例執行成功!")
'''
    
    exec(basic_code)


def run_workflow_example():
    """運行工作流程範例"""
    print("🔸 執行台灣股市工作流程範例...")
    
    script_path = Path(__file__).parent / "workflow_by_code_tw.py"
    if script_path.exists():
        subprocess.run([sys.executable, str(script_path)])
    else:
        print("❌ 工作流程腳本不存在")


def run_yaml_example():
    """運行 YAML 配置範例"""
    print("🔸 執行 YAML 配置範例...")
    
    yaml_path = Path(__file__).parent / "workflow_config_lightgbm_Alpha158_TW.yaml"
    if yaml_path.exists():
        try:
            subprocess.run(["qrun", str(yaml_path)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ YAML 範例執行失敗: {e}")
        except FileNotFoundError:
            print("❌ 請確認已安裝 qlib 並且 qrun 指令可用")
    else:
        print("❌ YAML 配置檔案不存在")


def check_data_availability():
    """檢查數據可用性"""
    from pathlib import Path
    
    data_path = Path.home() / ".qlib" / "qlib_data" / "tw_data"
    
    if not data_path.exists():
        print("❌ 台灣股市數據不存在")
        print("請先執行以下步驟準備數據:")
        print("1. cd scripts/data_collector/yahoo")
        print("2. python collector.py download_data --region TW ...")
        print("3. python collector.py normalize_data --region TW ...")
        print("4. cd ../../ && python dump_bin.py dump_all ...")
        return False
    
    # 檢查基本文件結構
    required_dirs = ["calendars", "instruments", "features"]
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not (data_path / dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"❌ 缺少必要目錄: {missing_dirs}")
        return False
    
    print("✅ 台灣股市數據檢查通過")
    return True


def main():
    parser = argparse.ArgumentParser(description="台灣股市範例執行器")
    parser.add_argument(
        "-e", "--example", 
        choices=["basic", "workflow", "yaml", "all"],
        default="basic",
        help="選擇要執行的範例類型"
    )
    
    args = parser.parse_args()
    
    print("🇹🇼 歡迎使用台灣股市 Qlib 範例")
    print("="*50)
    
    # 檢查數據可用性
    if not check_data_availability():
        return
    
    if args.example == "basic":
        run_basic_example()
        
    elif args.example == "workflow":
        run_workflow_example()
        
    elif args.example == "yaml":
        run_yaml_example()
        
    elif args.example == "all":
        print("🚀 執行所有範例...")
        run_basic_example()
        print("\n" + "-"*50 + "\n")
        run_workflow_example()
        print("\n" + "-"*50 + "\n") 
        run_yaml_example()
        
    print("\n✅ 範例執行完成!")


if __name__ == "__main__":
    main()
