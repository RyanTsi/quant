import akshare as ak
import pandas as pd
import os

def download_aligned_market_index(save_dir="history/past15year_stock_data_daily"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("正在下载并对齐大盘指数 (上证指数 sh000001)...")
    
    try:
        # 1. 使用东方财富接口 (字段更全，包含成交额)
        # symbol="sh000001" 代表上证指数
        df = ak.stock_zh_index_daily_em(symbol="sh000001")
        
        # 2. 筛选时间范围
        df['date'] = pd.to_datetime(df['date'])
        start = pd.to_datetime("2010-01-01")
        end = pd.to_datetime("2025-12-14")
        df = df[(df['date'] >= start) & (df['date'] <= end)].copy()
        
        # 3. 列名映射 (API返回列名 -> 你的标准列名)
        # EM接口通常返回: date, open, close, high, low, volume, amount
        rename_dict = {
            'date': '日期',
            'open': '开盘',
            'close': '收盘',
            'high': '最高',
            'low': '最低',
            'volume': '成交量',
            'amount': '成交额'
        }
        df.rename(columns=rename_dict, inplace=True)
        
        # 4. 补全缺失列 (计算衍生指标)
        # 预计算前收盘价 (Pre_Close)
        pre_close = df['收盘'].shift(1)
        
        # 补全: 股票代码
        df['股票代码'] = 'sh000001'
        
        # 补全: 涨跌额 (收盘 - 前收盘)
        df['涨跌额'] = df['收盘'] - pre_close
        
        # 补全: 涨跌幅 ((收盘 - 前收盘) / 前收盘 * 100)
        df['涨跌幅'] = (df['收盘'] / pre_close - 1) * 100
        
        # 补全: 振幅 ((最高 - 最低) / 前收盘 * 100)
        # 注意: 振幅的分母通常是"前收盘"，第一天会是NaN
        df['振幅'] = (df['最高'] - df['最低']) / pre_close * 100
        
        # 补全: 换手率 (指数无换手率，填0)
        df['换手率'] = 0.0
        
        # 5. 填补计算产生的 NaN (主要是第一天)
        df.fillna(0, inplace=True)

        # 6. === 强制列对齐 ===
        # 你指定的完整列顺序
        target_columns = [
            '日期', '股票代码', '开盘', '收盘', '最高', '最低', 
            '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率'
        ]
        
        # 检查是否缺少列 (防患于未然)
        for col in target_columns:
            if col not in df.columns:
                print(f"警告: 缺失列 {col}，已自动补0")
                df[col] = 0
                
        # 按指定顺序重排
        df_final = df[target_columns]
        
        # 7. 保存
        file_path = os.path.join(save_dir, "index_sh000001.csv")
        df_final.to_csv(file_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ 大盘指数处理成功！")
        print(f"数据路径: {file_path}")
        print(f"数据形状: {df_final.shape}")
        print(f"列预览: {df_final.columns.tolist()}")
        print(f"首行预览:\n{df_final.head(1)}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ 下载大盘指数失败: {e}")

if __name__ == "__main__":
    download_aligned_market_index()