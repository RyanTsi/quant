from datetime import datetime
from influx_manager import StockData, InfluxDBConfig, InfluxDBManager, InfluxDBCallbacks
import pandas as pd
from typing import Dict, List, Optional, Union

# ------- 配置信息 -------
HOST = "http://localhost:8181"
DATABASE = "stock_history_db"
TOKEN = "apiv3_yzu0u2VomPK9Bvsr94RFyVTGcUc-v06Q3YXen5T_cZfZoFuml2WEKecK1aHMxbQknTDm9kTZ2KWbNuhWb17lzA"
# ------------------------

def process_csv_and_upload(file_path: str, manager: InfluxDBManager):
    """
    读取指定格式的 CSV 文件并批量上传至 InfluxDB。
    
    :param file_path: CSV 文件路径
    :param manager: 已经初始化的 InfluxDBManager 实例
    """
    # 1. 定义字段映射（将 CSV 中的中文列名与逻辑对应）
    # 这样即使 CSV 列顺序变了，只要名字对就能读对
    field_cols = [
        '开盘', '收盘', '最高', '最低', '成交量', 
        '成交额', '振幅', '涨跌幅', '涨跌额', '换手率'
    ]

    print(f"📖 Reading CSV file: {file_path}")
    
    try:
        # 2. 加载数据
        # encoding='utf-8' 或 'gbk'，取决于你 CSV 的保存格式
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 3. 预处理
        # 转换日期格式
        df['日期'] = pd.to_datetime(df['日期'])
        # 确保股票代码是字符串（防止 000001 变成 1）
        df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
        
        # 4. 转换为 StockData 对象列表
        stock_data_list: List[StockData] = []
        
        for _, row in df.iterrows():
            # 提取 fields 字典
            fields = {col: float(row[col]) for col in field_cols if pd.notna(row[col])}
            
            # 创建自定义对象
            sd = StockData(
                timestamp=row['日期'],
                stock_code=row['股票代码'],
                fields=fields
            )
            stock_data_list.append(sd)

        # 5. 调用你之前实现的批量写入函数
        if stock_data_list:
            manager.write_stock_batch(stock_data_list)
            print(f"🚀 Successfully queued {len(stock_data_list)} rows for upload.")
            
    except Exception as e:
        print(f"❌ Error processing CSV: {e}")

def load_and_process_excel_data(file_path: str, sheet_name: Union[str, int] = 0) -> Optional[List[StockData]]:
    """
    读取 XLSX 文件并进行数据预处理，将每行数据转换为 StockData 对象。

    :param file_path: XLSX 文件路径。
    :param sheet_name: 工作表名称或索引 (默认为 0)。
    :return: 包含 StockData 对象的列表，失败则返回 None。
    """
    # 定义数值字段列表，它们将被转换为 InfluxDB 的 Field
    FIELD_COLUMNS = [
        '开盘', '收盘', '最高', '最低', '成交量', '成交额', 
        '振幅', '涨跌幅', '涨跌额', '换手率'
    ]
    
    try:
        # 1. 读取 Excel 文件
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Successfully loaded {len(df)} rows from {file_path} (Sheet: {sheet_name}).")
        
        # 2. 数据预处理
        
        # 检查关键列是否存在
        required_cols = ['日期', '股票代码'] + FIELD_COLUMNS
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"❌ Error: Missing required columns in Excel: {', '.join(missing)}")
            return None

        # 转换日期列：确保是 datetime 类型
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce') 
        # 移除日期为空的行
        df.dropna(subset=['日期'], inplace=True)
        
        # 转换数值列：确保是 float 类型
        for col in FIELD_COLUMNS:
            # 强制转换为数值类型，无法转换的设为 NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. 转换为 StockData 对象列表
        stock_data_list: List[StockData] = []
        
        # 迭代 DataFrame 的每一行
        for index, row in df.iterrows():
            # 提取 Tag 和 Timestamp
            timestamp: datetime = row['日期']
            stock_code: str = str(row['股票代码']) # 确保股票代码是字符串

            # 提取 Fields
            fields: Dict[str, Union[float, int]] = {}
            for col in FIELD_COLUMNS:
                value = row[col]
                if pd.notna(value): # 排除 NaN 值
                    # 简单判断，成交量等大数用 int 存储可能更好，但 float 更通用
                    if col in ['成交量', '成交额'] and value.is_integer():
                         fields[col] = int(value)
                    else:
                         fields[col] = float(value)
            
            # 创建 StockData 对象并添加到列表
            stock_data_list.append(
                StockData(timestamp=timestamp, stock_code=stock_code, fields=fields)
            )

        print(f"✅ Successfully processed {len(stock_data_list)} rows into StockData objects.")
        return stock_data_list
        
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return None
    except ImportError:
        print("❌ Error: 'openpyxl' library not found. Please install it using 'pip install openpyxl'.")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred during Excel processing: {e}")
        return None
    
if __name__ == '__main__':

    EXCEL_FILE_PATH = 'C:/Users/sola/Documents/quant/history/000002.xlsx'
    EXCEL_SHEET_NAME = 0
    processed_df = load_and_process_excel_data(EXCEL_FILE_PATH, EXCEL_SHEET_NAME)
    try:
        if processed_df:
            # 【重要】确保在循环中进行数据转换 (如上一轮回答所示)
            for stock_data in processed_df:
                config = InfluxDBConfig(HOST, DATABASE, TOKEN)
                callbacks = InfluxDBCallbacks()
                influx_manager = InfluxDBManager(config, callbacks)
                influx_manager.write_stock_single(stock_data)
                influx_manager.close() # 这一行是解决问题的核心
            # 注意：循环结束后，数据仍在缓冲区中！
        
    except Exception as e:
        print(f"致命错误发生: {e}")
