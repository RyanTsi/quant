import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import copy

class QuantTransformer(nn.Module):
    def __init__(self, feature_dim=21, d_model=64, nhead=4, num_layers=2, dropout=0.3, seq_len=30):
        """
        量化时序 Transformer 模型
        :param feature_dim: 输入的特征数量 
        :param d_model: Transformer 内部的隐藏层维度 (建议 64 或 128)
        :param nhead: 多头注意力的头数 (让模型同时从不同角度寻找规律)
        :param num_layers: Transformer Encoder 的层数
        :param dropout: 防过拟合的神器，丢弃神经元的比例
        :param seq_len: 滑动窗口长度 (30 天)
        """
        super(QuantTransformer, self).__init__()
        
        # 1. 连续特征投影层 (把 21 维膨胀到 64 维)
        self.feature_proj = nn.Linear(feature_dim, d_model)
        
        # 2. 可学习的位置编码 (Learnable Positional Encoding)
        # 初始化一个服从正态分布的张量，模型在训练时会自动学会如何理解时间顺序
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # 3. Transformer 核心引擎
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, # 内部前馈网络放大倍数
            dropout=dropout,
            batch_first=True, # 🌟 极其重要！告诉 PyTorch 我们的输入格式是 [Batch, Seq, Feature]
            activation='gelu' # 相比 ReLU，GELU 在深度网络中表现更好
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 预测头 (Prediction Head)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1) # 最终压缩成 1 维，即 Alpha 分数
        )

    def forward(self, x):
        # 输入 x 形状: [Batch_Size, 30, 24]
        
        # [1] 投影到高维 -> [Batch_Size, 30, 64]
        x = self.feature_proj(x)
        
        # [2] 注入时间位置信息 (直接相加) -> [Batch_Size, 30, 64]
        x = x + self.pos_encoder
        
        # [3] 经过多头自注意力引擎 -> [Batch_Size, 30, 64]
        x = self.transformer_encoder(x)
        
        # [4] 提取特征：我们取时间序列最后一天 (idx = -1) 的状态作为决策依据
        # 因为在金融时序中，“今天”包含了最多的当期博弈信息，结合前面的上下文最准
        last_day_state = x[:, -1, :] # 形状变成 -> [Batch_Size, 64]
        
        # (替代方案：如果想综合 30 天特征，可以使用 x.mean(dim=1)，但实测最后一天更好)
        
        # [5] 输出打分 -> [Batch_Size, 1]
        out = self.fc_out(last_day_state)
        
        # 降维成 [Batch_Size]，和 Target 标签的形状完全对齐！
        return out.squeeze(dim=-1)
    

class StockTimeSeriesDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_len=30):
        """
        :param df: 预处理后的 DataFrame
        :param feature_cols: 特征列名 list
        :param target_col: 预测目标列名 (如 'TARGET_LOG_RET_5')
        :param seq_len: 滑动窗口长度 (默认30)
        """
        # 确保数据按股票代码和时间排序，这对于时序滑动窗口极其重要
        self.df = df.sort_values(by=['symbol', 'date']).reset_index(drop=True)
        
        self.x_data = self.df[feature_cols].values
        self.y_data = self.df[target_col].values
        self.seq_len = seq_len
        
        # 预先计算所有合法的滑动窗口结束索引
        self.valid_indices = []
        
        # 按股票分组，寻找合法的切片区间
        for symbol, group in self.df.groupby('symbol'):
            start_idx = group.index[0]
            end_idx = group.index[-1]
            
            # 如果该股票的数据总长度大于等于窗口长度，才有效
            if end_idx - start_idx + 1 >= seq_len:
                # 记录每一个可以作为窗口最后一天的 index
                self.valid_indices.extend(range(start_idx + seq_len - 1, end_idx + 1))
                
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # end_i 是当前窗口的最后一天
        end_i = self.valid_indices[idx]
        start_i = end_i - self.seq_len + 1
        
        # 提取窗口内的特征和最后一天的目标值
        x = torch.tensor(self.x_data[start_i : end_i + 1], dtype=torch.float32)
        y = torch.tensor(self.y_data[end_i], dtype=torch.float32)
        
        return x, y
    

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=1e-4, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 量化常用 MSELoss 作为基础损失
        self.criterion = nn.MSELoss()
        # AdamW 对权重衰减的处理更好，适合防过拟合
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in tqdm(self.train_loader, desc="Training"):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪：防止 Transformer 训练初期梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        # 计算 IC 值 (Pearson Correlation)
        preds_series = pd.Series(all_preds)
        targets_series = pd.Series(all_targets)
        ic = preds_series.corr(targets_series)
        
        return total_loss / len(self.val_loader), ic

    def fit(self, epochs=20, early_stopping_patience=5):
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_weights = None
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, val_ic = self.evaluate()
            
            print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val IC: {val_ic:.4f}")
            
            # 早停机制 (Early Stopping)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_weights = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}! Best Val Loss: {best_val_loss:.6f}")
                    break
                    
        # 加载最优权重
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
        return self.model