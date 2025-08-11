from loguru import logger
import os
from typing import Optional

import fire
import pandas as pd
import qlib
from tqdm import tqdm

from qlib.data import D


class DataHealthChecker:
    """Checks a dataset for data completeness and correctness. The data will be converted to a pd.DataFrame and checked for the following problems:
    - any of the columns ["open", "high", "low", "close", "volume"] are missing
    - any data is missing
    - any step change in the OHLCV columns is above a threshold (default: 0.5 for price, 3 for volume)
    - any factor is missing
    """

    def __init__(
        self,
        csv_path=None,
        qlib_dir=None,
        freq="day",
        large_step_threshold_price=0.5,
        large_step_threshold_volume=3,
        missing_data_num=0,
    ):
        assert csv_path or qlib_dir, "One of csv_path or qlib_dir should be provided."
        assert not (csv_path and qlib_dir), "Only one of csv_path or qlib_dir should be provided."

        self.data = {}
        self.problems = {}
        self.freq = freq
        self.large_step_threshold_price = large_step_threshold_price
        self.large_step_threshold_volume = large_step_threshold_volume
        self.missing_data_num = missing_data_num

        if csv_path:
            assert os.path.isdir(csv_path), f"{csv_path} should be a directory."
            files = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
            for filename in tqdm(files, desc="Loading data"):
                df = pd.read_csv(os.path.join(csv_path, filename))
                self.data[filename] = df

        elif qlib_dir:
            qlib.init(provider_uri=qlib_dir)
            self.load_qlib_data()

    def load_qlib_data(self):
        # 先嘗試使用 "all" 市場，如果失敗則嘗試台灣指數
        try:
            instruments = D.instruments(market="all")
            instrument_list = D.list_instruments(instruments=instruments, as_list=True, freq=self.freq)
        except ValueError as e:
            if "all.txt" in str(e):
                logger.info("all.txt not found, trying Taiwan market indices...")
                # 嘗試台灣指數
                try:
                    instruments = D.instruments(market="twii")  # 台灣加權指數
                    instrument_list = D.list_instruments(instruments=instruments, as_list=True, freq=self.freq)
                    logger.info(f"✅ Using TWII (Taiwan Weighted Index) with {len(instrument_list)} instruments")
                except ValueError:
                    # 如果 twii 也失敗，嘗試其他台灣指數
                    try:
                        instruments = D.instruments(market="tw50")  # 台灣50指數
                        instrument_list = D.list_instruments(instruments=instruments, as_list=True, freq=self.freq)
                        logger.info(f"✅ Using TW50 (Taiwan 50 Index) with {len(instrument_list)} instruments")
                    except ValueError:
                        try:
                            instruments = D.instruments(market="twmidcap")  # 台灣中型100指數
                            instrument_list = D.list_instruments(instruments=instruments, as_list=True, freq=self.freq)
                            logger.info(f"✅ Using TWMIDCAP (Taiwan Mid-Cap 100) with {len(instrument_list)} instruments")
                        except ValueError:
                            logger.error("❌ No valid market index found. Please ensure your data contains valid instrument files.")
                            raise e
            else:
                raise e
        
        required_fields = ["$open", "$close", "$low", "$high", "$volume", "$factor"]
        logger.info(f"📊 Loading data for {len(instrument_list)} instruments...")
        
        for instrument in tqdm(instrument_list, desc="Loading instrument data"):
            try:
                df = D.features([instrument], required_fields, freq=self.freq)
                df.rename(
                    columns={
                        "$open": "open",
                        "$close": "close",
                        "$low": "low",
                        "$high": "high",
                        "$volume": "volume",
                        "$factor": "factor",
                    },
                    inplace=True,
                )
                self.data[instrument] = df
            except Exception as e:
                logger.warning(f"⚠️ Failed to load data for {instrument}: {e}")
                continue
        
        logger.info(f"✅ Successfully loaded data for {len(self.data)} instruments")

    def check_missing_data(self) -> Optional[pd.DataFrame]:
        """Check if any data is missing in the DataFrame."""
        result_dict = {
            "instruments": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }
        for filename, df in self.data.items():
            missing_data_columns = df.isnull().sum()[df.isnull().sum() > self.missing_data_num].index.tolist()
            if len(missing_data_columns) > 0:
                result_dict["instruments"].append(filename)
                result_dict["open"].append(df.isnull().sum()["open"])
                result_dict["high"].append(df.isnull().sum()["high"])
                result_dict["low"].append(df.isnull().sum()["low"])
                result_dict["close"].append(df.isnull().sum()["close"])
                result_dict["volume"].append(df.isnull().sum()["volume"])

        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            return result_df
        else:
            logger.info(f"✅ There are no missing data.")
            return None

    def check_large_step_changes(self) -> Optional[pd.DataFrame]:
        """Check if there are any large step changes above the threshold in the OHLCV columns."""
        result_dict = {
            "instruments": [],
            "col_name": [],
            "date": [],
            "pct_change": [],
        }
        for filename, df in self.data.items():
            affected_columns = []
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    pct_change = df[col].pct_change(fill_method=None).abs()
                    threshold = self.large_step_threshold_volume if col == "volume" else self.large_step_threshold_price
                    if pct_change.max() > threshold:
                        large_steps = pct_change[pct_change > threshold]
                        result_dict["instruments"].append(filename)
                        result_dict["col_name"].append(col)
                        result_dict["date"].append(large_steps.index.to_list()[0][1].strftime("%Y-%m-%d"))
                        result_dict["pct_change"].append(pct_change.max())
                        affected_columns.append(col)

        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            return result_df
        else:
            logger.info(f"✅ There are no large step changes in the OHLCV column above the threshold.")
            return None

    def check_required_columns(self) -> Optional[pd.DataFrame]:
        """Check if any of the required columns (OLHCV) are missing in the DataFrame."""
        required_columns = ["open", "high", "low", "close", "volume"]
        result_dict = {
            "instruments": [],
            "missing_col": [],
        }
        for filename, df in self.data.items():
            if not all(column in df.columns for column in required_columns):
                missing_required_columns = [column for column in required_columns if column not in df.columns]
                result_dict["instruments"].append(filename)
                result_dict["missing_col"] += missing_required_columns

        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            return result_df
        else:
            logger.info(f"✅ The columns (OLHCV) are complete and not missing.")
            return None

    def check_missing_factor(self) -> Optional[pd.DataFrame]:
        """Check if the 'factor' column is missing in the DataFrame."""
        result_dict = {
            "instruments": [],
            "missing_factor_col": [],
            "missing_factor_data": [],
        }
        
        # 定義指數代碼，這些通常不需要 factor 調整
        index_codes = ["000300", "000903", "000905", "^TWII", "TWII"]  # 中國指數 + 台灣指數
        
        for filename, df in self.data.items():
            # 跳過指數，因為指數通常不需要復權因子
            is_index = any(index_code in str(filename) for index_code in index_codes)
            if is_index:
                continue
                
            if "factor" not in df.columns:
                result_dict["instruments"].append(filename)
                result_dict["missing_factor_col"].append(True)
                result_dict["missing_factor_data"].append(False)
            elif df["factor"].isnull().all():
                if filename in result_dict["instruments"]:
                    idx = result_dict["instruments"].index(filename)
                    result_dict["missing_factor_data"][idx] = True
                else:
                    result_dict["instruments"].append(filename)
                    result_dict["missing_factor_col"].append(False)
                    result_dict["missing_factor_data"].append(True)

        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            return result_df
        else:
            logger.info(f"✅ The `factor` column already exists and is not empty.")
            return None

    def check_data(self):
        check_missing_data_result = self.check_missing_data()
        check_large_step_changes_result = self.check_large_step_changes()
        check_required_columns_result = self.check_required_columns()
        check_missing_factor_result = self.check_missing_factor()
        
        has_issues = (
            check_missing_data_result is not None
            or check_large_step_changes_result is not None
            or check_required_columns_result is not None
            or check_missing_factor_result is not None
        )
        
        if has_issues:
            print(f"\nSummary of data health check ({len(self.data)} files checked):")
            print("-------------------------------------------------")
            if isinstance(check_missing_data_result, pd.DataFrame):
                logger.warning(f"There is missing data.")
                print(check_missing_data_result)
            if isinstance(check_large_step_changes_result, pd.DataFrame):
                logger.warning(f"The OHLCV column has large step changes.")
                print(check_large_step_changes_result)
            if isinstance(check_required_columns_result, pd.DataFrame):
                logger.warning(f"Columns (OLHCV) are missing.")
                print(check_required_columns_result)
            if isinstance(check_missing_factor_result, pd.DataFrame):
                logger.warning(f"The factor column does not exist or is empty")
                print(check_missing_factor_result)
        else:
            print(f"\n✅ Data health check passed! ({len(self.data)} files checked)")
            print("-------------------------------------------------")
            logger.info("All checks passed successfully.")


if __name__ == "__main__":
    fire.Fire(DataHealthChecker)
