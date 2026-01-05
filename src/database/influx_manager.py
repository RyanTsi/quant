from datetime import datetime
from reactivex.scheduler import ThreadPoolScheduler
from typing import Dict, List, Optional, Union
import pandas as pd
from influxdb_client_3 import (
    InfluxDBClient3,
    write_client_options,
    WriteOptions,
    Point,
    InfluxDBError,
)
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from config import *




class StockData:
    """
    用于存储单个股票时间点数据的结构。
    """
    def __init__(self, timestamp: datetime, stock_code: str, fields: Dict[str, Union[float, int]]):
        self.timestamp = timestamp
        self.stock_code = stock_code
        self.fields = fields

    def __repr__(self):
        # 方便调试时查看对象内容
        return (f"StockData(timestamp={self.timestamp.strftime('%Y-%m-%d')}, "
                f"code='{self.stock_code}', fields_count={len(self.fields)})")

class InfluxDBCallbacks:
    @staticmethod
    def success(config: WriteOptions, data: str):
        """写入成功时调用"""
        print(f"✅ Success writing batch. Size: {len(data)} bytes.")

    @staticmethod
    def error(config: WriteOptions, data: str, err: InfluxDBError):
        """写入失败时调用"""
        print(f"❌ Error writing batch. Data: {data[:50]}..., Error: {err}")
        # 这里可以添加日志记录或告警逻辑

    @staticmethod
    def retry(config: WriteOptions, data: str, err: InfluxDBError):
        """写入重试时调用"""
        print(f"⚠️ Retry writing batch. Data: {data[:50]}..., Error: {err}")


class InfluxDBConfig:
    def __init__(self, host: str, database: str, token: str):
        self.host = host
        self.database = database
        self.token = token


class InfluxDBManager:
    STOCK_MEASUREMENT = "stock_prices"
    def __init__(self, config: InfluxDBConfig, callbacks: InfluxDBCallbacks):
        write_options = WriteOptions(
            batch_size=40_000,
            flush_interval=5_000,
            max_retry_delay=30_000,
            max_retry_time=300_000,
            max_close_wait=10000000,
            write_scheduler=ThreadPoolScheduler(max_workers=12)
        )
        wco = write_client_options( success_callback=callbacks.success,
                            error_callback=callbacks.error,
                            retry_callback=callbacks.retry,
                            write_options=write_options)
        self.client = InfluxDBClient3(host=config.host, database=config.database, token=config.token, write_client_options=wco)
        print(f"InfluxDB client initialized for {config.host}/{config.database}")
    
    def create_stock_point(
        self,
        stock_data : StockData,
        measurement_name: str = "stock_prices"
    ) -> Point:
        """
        根据给定的数据构造一个 InfluxDB Point 对象。
        """
        point = Point(measurement_name) \
            .tag("stock_code", stock_data.stock_code) \
            .time(stock_data.timestamp)

        # 添加所有数值字段
        for key, value in stock_data.fields.items():
            if key not in ["股票代码", "日期"]:
                # 自动处理整数和浮点数
                if isinstance(value, int):
                    point.field(key, int(value))
                elif isinstance(value, float):
                    point.field(key, float(value))
        return point
    
    def write_stock_single(
        self,
        stock_data : StockData,
        measurement_name: str = "stock_prices"
    ):
        """
        写入单个股票数据点到预定义的 Measurement。
        
        :param point: 包含股票数据的 InfluxDB Point 对象。
        """
        print(f"Attempting to write single point to '{self.STOCK_MEASUREMENT}'...")
        point = InfluxDBManager.create_stock_point(stock_data, measurement_name)
        try:
            self.client.write(point, measurement_name=self.STOCK_MEASUREMENT)
            print("Single point write initiated. Data buffered for asynchronous flush.")

        except Exception as e:
            print(f"❌ Failed to initiate single point write: {e}")
    
    def write_stock_batch(
        self, 
        stock_data_list: List[StockData], 
        measurement_name: str = "stock_prices"
    ):
        """
        批量写入股票数据列表。
        
        :param stock_data_list: 包含多个 StockData 对象的列表。
        :param measurement_name: InfluxDB 中的表名。
        """
        if not stock_data_list:
            print("No data to write.")
            return

        print(f"📦 Preparing to batch write {len(stock_data_list)} points to '{measurement_name}'...")

        try:
            # 1. 利用列表推导式将 StockData 转换成 Point 列表
            # 注意：这里的 create_stock_point 应当是你在类中定义的静态方法或实例方法
            points = [
                self.create_stock_point(sd, measurement_name) 
                for sd in stock_data_list
            ]

            # 2. 调用底层 client 的 write 方法
            # influxdb-client-3 的 write 方法原生支持接收一个 Point 列表
            self.client.write(record=points)

            print(f"✅ Batch write initiated. {len(points)} points sent to buffer.")

        except Exception as e:
            print(f"❌ Failed to batch write stock data: {e}")

    def get_stock_code_list_by_date(
        self, 
        target_date: datetime
    ) -> Optional[List[str]]:
        """
        查询指定日期有数据的所有股票代码。
        
        :param target_date: 目标日期 (datetime 对象)
        :return: 股票代码列表或 None
        """
        
        # 1. 构造 SQL 语句
        query_str = f"""
            SELECT DISTINCT("stock_code") FROM "{self.STOCK_MEASUREMENT}" 
            WHERE time >= $start AND time < $end
        """
        
        # 2. 定义查询参数
        start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + pd.Timedelta(days=1)
        params = {
            "start": start_of_day.isoformat(),
            "end": end_of_day.isoformat()
        }

        print(f"🔍 Querying stock codes for date {start_of_day.date()}...")

        try:
            # 3. 执行查询
            result = self.client.query(
                query=query_str, 
                language="sql", 
                mode="pandas", 
                query_parameters=params
            )
            if result is not None and not result.empty:
                stock_codes = result['stock_code'].unique().tolist()
                return stock_codes
            else:
                return None
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return None

    def get_stock_data_by_range(
        self, 
        stock_code: str, 
        start_time: datetime, 
        end_time: datetime, 
        mode: str = "pandas"
    ) -> Optional[pd.DataFrame]:
        """
        查询特定股票在指定时间段的数据。
        
        :param stock_code: 股票代码 (例如 'sh600000')
        :param start_time: 开始时间 (datetime 对象)
        :param end_time: 结束时间 (datetime 对象)
        :param mode: 返回模式，建议用 'pandas'
        :return: 包含数据的 DataFrame 或其他指定格式
        """
        
        # 1. 构造 SQL 语句
        # 注意：InfluxDB 3.0 中，measurement 相当于表名，tag 相当于列
        # 使用参数化查询防止注入（虽然是内部使用，但这是好习惯）
        query_str = f"""
            SELECT * FROM "{self.STOCK_MEASUREMENT}" 
            WHERE "stock_code" = $code 
            AND time >= $start 
            AND time <= $end
            ORDER BY time ASC
        """
        
        # 2. 定义查询参数
        # InfluxDB 3.0 的时间戳需要是 RFC3339 格式或 ISO 格式字符串
        params = {
            "code": stock_code,
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        }

        print(f"🔍 Querying data for {stock_code} from {params['start']} to {params['end']}...")

        try:
            # 3. 执行查询
            # 使用 self.client.query，它是你之前展示的那个函数
            result = self.client.query(
                query=query_str, 
                language="sql", 
                mode=mode, 
                query_parameters=params
            )
            return result
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return None

    def close(self):
        """
        关闭客户端，确保所有缓冲的异步写入数据被发送。
        """
        if self.client:
            self.client.close()
            print("Client closed and write buffer flushed.")

def parse_single_csv(file_path: str):
    """
    负责最耗时的解析工作，运行在独立的进程中
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        if df.empty: return []
        
        df['日期'] = pd.to_datetime(df['日期'])
        # 向量化转换比 iterrows 快得多
        stock_code = str(df['股票代码'].iloc[0]).zfill(6)
        field_cols = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
        
        # 预先过滤掉含空值的列，转为字典列表，减少跨进程传输开销
        records = df.to_dict('records')
        batch_results = []
        for row in records:
            fields = {col: float(row[col]) for col in field_cols if pd.notna(row[col])}
            # 传输基础类型字典，比传输 StockData 对象更轻量
            batch_results.append({
                'time': row['日期'],
                'code': stock_code,
                'fields': fields
            })
        return batch_results
    except Exception as e:
        return []
    
def store_file(dir):
    # 1. 初始化 InfluxDB
    config = InfluxDBConfig(HOST, DATABASE, TOKEN)
    manager = InfluxDBManager(config, InfluxDBCallbacks())
    
    csv_files = glob.glob(os.path.join(dir, "*.csv"))
    # csv_files = [ DIRECTORY + "/index_sh000001.csv"]
    print(f"🔥 启动多进程解析引擎 (Workers: {MAX_WORKERS})...")
    
    pending_buffer = []
    total_count = 0
    start_time = time.time()

    # 2. 使用进程池并行解析
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # as_completed 保证哪个文件先解析完就先处理哪个，不再按顺序死等
        future_to_file = {executor.submit(parse_single_csv, f): f for f in csv_files}
        
        for future in as_completed(future_to_file):
            data = future.result()
            if data:
                pending_buffer.extend(data)
                
                # 3. 达到批次大小，异步写入 IO
                if len(pending_buffer) >= BATCH_WRITE_SIZE:
                    # 转换为 StockData 对象
                    write_list = [StockData(timestamp=d['time'], stock_code=d['code'], fields=d['fields']) 
                                 for d in pending_buffer]
                    print(f"📦 缓冲区已满 ({len(write_list)}行)，正在提交 InfluxDB...")
                    manager.write_stock_batch(write_list)
                    total_count += len(write_list)
                    pending_buffer = []

    # 4. 清空剩余数据
    if pending_buffer:
        write_list = [StockData(timestamp=d['time'], stock_code=d['code'], fields=d['fields']) 
                     for d in pending_buffer]
        manager.write_stock_batch(write_list)
        total_count += len(write_list)

    end_time = time.time()
    manager.close()
    print(f"\n✨ 重构完成！")
    print(f"⏱️ 总耗时: {end_time - start_time:.2f} 秒")
    print(f"📈 总记录数: {total_count}")