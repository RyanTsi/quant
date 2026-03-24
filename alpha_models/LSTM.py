import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, embedding_dim, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim,
            num_layers,
            batch_first=True,   # 输入形状为 (Batch, Seq, Feature)
            bidirectional=True,
            dropout=dropout
        )

        self.attention_query = nn.Linear(hidden_dim * 2, 1)
        
        self.fc_embedding = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim) # 归一化，利于下游模型 C 接收
        )
        
        self.head_1d_ret = nn.Linear(embedding_dim, 1)   # 预测 1d 涨幅
        self.head_5d_ret = nn.Linear(embedding_dim, 1)   # 预测 5d 涨幅
        self.head_5d_vol = nn.Linear(embedding_dim, 1)   # 预测 5d 波动率
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # 1. 通过双向 LSTM
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)
        
        # 2. 计算注意力权重 (Simple Self-Attention)
        # attn_scores shape: (batch_size, seq_len, 1)
        attn_scores = self.attention_query(lstm_out)
        attn_weights = F.softmax(attn_scores, dim=1) # 在时间步维度上做归一化
        
        # 3. 加权求和得到上下文向量 (Context Vector)
        # context shape: (batch_size, hidden_dim * 2)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # 4. 提取共享特征嵌入
        # embedding shape: (batch_size, embedding_dim)
        embedding = self.fc_embedding(context)
        
        # 5. 多任务输出
        ret_1d = self.head_1d_ret(embedding)
        ret_5d = self.head_5d_ret(embedding)
        vol_5d = self.head_5d_vol(embedding)
        
        # 返回三个目标的预测结果
        return ret_1d, ret_5d, vol_5d
    
class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=3):
        super(MultiTaskLoss, self).__init__()
        # 初始化 log_vars，参数为 0 意味着初始权重为 1 (exp(0)=1)
        # 这是一个可学习的参数 (Parameter)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, preds, targets):
        """
        preds: list of tensors [pred_1d, pred_5d, pred_vol]
        targets: list of tensors [target_1d, target_5d, target_vol]
        """
        loss_total = 0
        
        # 遍历每一个任务
        for i, (pred, target) in enumerate(zip(preds, targets)):
            # 1. 计算该任务的基础 MSE Loss
            mse_loss = F.mse_loss(pred, target)
            
            # 2. 获取该任务的“不确定性权重”
            # 公式: loss = (1 / 2*sigma^2) * MSE + log(sigma)
            # 这里的 self.log_vars[i] 相当于 log(sigma^2)
            precision = torch.exp(-self.log_vars[i])
            loss_total += precision * mse_loss + self.log_vars[i]
            
        return loss_total