import src.data_loader.load_data as loaddata
from config import *
from src.database.influx_manager import InfluxDBManager, InfluxDBConfig, InfluxDBCallbacks

db_config = InfluxDBConfig(HOST, DATABASE, TOKEN)
db_manager = InfluxDBManager(db_config, InfluxDBCallbacks())
target_date = datetime(2025, 12, 12)
all_codes = db_manager.get_stock_code_list_by_date(target_date)

loaddata.get_data_processed_with_cache(db_manager, all_codes, train_range[0], train_range[1], "./data/processed/train_data.pkl")
loaddata.get_data_processed_with_cache(db_manager, all_codes,   val_range[0],   val_range[1], "./data/processed/val_data.pkl")
loaddata.get_data_processed_with_cache(db_manager, all_codes,  test_range[0],  test_range[1], "./data/processed/test_data.pkl")