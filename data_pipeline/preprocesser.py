import numpy as np
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt
import seaborn as sns

class Preprocesser:
    def __init__(self, raw_df):
        self.raw_df = raw_df
        self.tar_df = None
        self.not_feat = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'factor']
    
    def _clean_data(self):
        self.raw_df = self.raw_df[(self.raw_df['volume'] > 0) & (self.raw_df['close'] > 0)]

    def run(self):
        self._clean_data()
        processed_dfs = []
        for symbol, group_df in self.raw_df.groupby('symbol'):
            df_with_features = self._generate_features(group_df) 
            processed_dfs.append(df_with_features)
        self.tar_df = pd.concat(processed_dfs, axis=0)
        feature_cols = [c for c in self.tar_df.columns if c not in self.not_feat]
        def cs_zscore(x):
            return (x - x.mean()) / (x.std() + 1e-8)
        self.tar_df[feature_cols] = self.tar_df.groupby('date')[feature_cols].transform(cs_zscore)
        self.tar_df[feature_cols] = self.tar_df[feature_cols].clip(-3.0, 3.0)
        self.tar_df.dropna(inplace=True)
        self.tar_df.reset_index(drop=True, inplace=True)

    def _generate_features(self, df):
        """
        基于 OHLCV 生成高质量、去量纲、平稳化的深度学习特征
        """
        o = df['open'].astype('float64').values
        h = df['high'].astype('float64').values
        l = df['low'].astype('float64').values
        c = df['close'].astype('float64').values
        v = df['volume'].astype('float64').values
        
        window_array = [5, 21, 55]

        features = {}

        # ==========================================
        # 1. 基础对数收益率 (Log Returns)
        # ==========================================
        for n in [1, 5, 13]:
            shifted_c = df['close'].shift(n).values
            future_c = df['close'].shift(-n).values
            features[f'LOG_RET_{n}'] = np.log(c / (shifted_c))
            features[f'TARGET_LOG_RET_{n}'] = np.log(future_c / c)

        # ==========================================
        # 2. 动量与震荡 (Momentum)
        # ==========================================
        for n in window_array:
            # RSI 相对强弱（0~1）
            features[f'RSI_{n}'] = ta.RSI(c, timeperiod=n) / 100

        # ==========================================
        # 3. 均线偏离度 (Trend Deviation / BIAS)  
        # ==========================================
        for n in window_array:
            sma = ta.SMA(c, timeperiod=n)
            features[f'BIAS_{n}'] = (c - sma) / (sma + 1e-8)
            
        # MACD 相对化
        macd, macdsignal, macdhist = ta.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
        features['MACD_NORM'] = macd / (c + 1e-8)
        features['MACD_HIST_NORM'] = macdhist / (c + 1e-8)

        # ==========================================
        # 4. 波动率 (Volatility) 
        # ==========================================
        for n in window_array:
            features[f'NATR_{n}'] = ta.NATR(h, l, c, timeperiod=n) / 100.0
            
        # 布林带相关
        upper, middle, lower = ta.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        features['BOLL_POSITION'] = (c - lower) / (upper - lower + 1e-8) # 价格在布林带内的相对位置 (0~1)
        features['BOLL_WIDTH'] = (upper - lower) / (middle + 1e-8)       # 布林带开口宽度 (波动率)
        
        # ==========================================
        # 5. 量价动态 (Volume Dynamics) 
        # ==========================================
        # 成交量比率 (今天成交量 / 过去5天平均成交量)
        v_ma5 = ta.SMA(v, timeperiod=5)
        features['VOL_RATIO_5'] = v / (v_ma5 + 1e-8)
        
        # OBV 变动率 (能量潮的 5 日 ROC)
        obv = ta.OBV(c, v)
        features['OBV_ROC_5'] = ta.ROCP(obv, timeperiod=5)


        # ==========================================
        # 6. K 线形态综合信号 (K-Line Reversal Signal)
        # ==========================================
        range_hl = h - l + 1e-8 
        # 1. 实体比例 (Body Ratio): 范围 [-1, 1]。
        features['K_BODY_RATIO'] = (c - o) / range_hl
        # 2. 上影线比例 (Upper Shadow Ratio): 范围 [0, 1]。
        features['K_UP_SHADOW_RATIO'] = (h - np.maximum(o, c)) / range_hl
        # 3. 下影线比例 (Lower Shadow Ratio): 范围 [0, 1]。
        features['K_LOW_SHADOW_RATIO'] = (np.minimum(o, c) - l) / range_hl

        feature_df = pd.DataFrame(features, index=df.index)
        result_df = pd.concat([df, feature_df], axis=1)

        result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return result_df.dropna().reset_index(drop=True)


    def analyze_feature_correlation(self, df, threshold=0.8):
        """
        绘制相关性热力图，并打印高相关性特征对。
        """
        # 1. 筛选出所有的特征列
        cols_to_drop = ['date','symbol','open', 'close', 'high', 'low', 'volume']
        feature_cols = [c for c in df.columns if c not in cols_to_drop]

        print(feature_cols)

        if not feature_cols:
            print("未找到 feat_ 开头的特征列！")
            return

        # 计算相关性矩阵
        corr_matrix = df[feature_cols].corr()

        # ==========================
        # A. 绘制热力图 (Visual Check)
        # ==========================
        plt.figure(figsize=(14, 12))
        
        # 生成上三角掩码 (因为矩阵是对称的，看一半就够了)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # 绘图
        sns.heatmap(
            corr_matrix, 
            mask=mask, 
            cmap='coolwarm',  # 冷暖色调：红正相关，蓝负相关
            vmax=1, vmin=-1, center=0,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5},
            annot=True,       # 显示数值
            fmt=".2f"         # 保留两位小数
        )
        
        plt.title('LSTM Feature Correlation Matrix', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # ==========================
        # B. 自动报警 (Auto Alert)
        # ==========================
        print(f"\n[警报] 相关系数绝对值大于 {threshold} 的特征对：")
        print("-" * 60)
        
        # 遍历矩阵的上三角找出高相关性
        high_corr_pairs = []
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr_matrix.iloc[i, j]
                if abs(val) >= threshold:
                    high_corr_pairs.append((cols[i], cols[j], val))
        
        if not high_corr_pairs:
            print("完美！没有发现高度冗余的特征。")
        else:
            # 按相关性大小排序打印
            high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            for f1, f2, val in high_corr_pairs:
                print(f"{f1} <--> {f2}: {val:.4f}")
                # 给出简单的剔除建议
                print(f"   -> 建议: 二者保留其一 (通常保留计算逻辑更简单的那个)")

