## Requirements

```bash
pip install -r requirements.txt
```

## Collector Data

```bash
# 生成台灣加權指數成分股清單
python collector.py --index_name TWII --qlib_dir ~/.qlib/qlib_data/tw_data --method parse_instruments
python collector.py --index_name TW50 --qlib_dir ~/.qlib/qlib_data/tw_data --method parse_instruments
python collector.py --index_name TWMIDCAP --qlib_dir ~/.qlib/qlib_data/tw_data --method parse_instruments
```

