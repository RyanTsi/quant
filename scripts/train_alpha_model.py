import pandas as pd
from alpha_models.quantTransformer import StockTimeSeriesDataset
from data_pipeline.preprocesser import Preprocesser

def main():
    
    tar_df = pd.read_csv('data/processed_data.csv')
    
    # ⚠️ 修正数据泄露：筛选真正的特征列（剔除 TARGET 前缀的列）
    all_not_feat = processor.not_feat + [c for c in tar_df.columns if 'TARGET' in c]
    feature_cols = [c for c in tar_df.columns if c not in all_not_feat]
    
    # 选定我们要预测的目标，例如预测未来 5 天收益率
    target_col = 'TARGET_LOG_RET_5'
    
    print(f"生成了 {len(feature_cols)} 个特征，预测目标为 {target_col}")
    
    # 2. 时序切分数据集 (Out-of-Time Validation)
    # 假设用 2022 年之前的数据训练，2022 年以后的数据验证
    tar_df['date'] = pd.to_datetime(tar_df['date'])
    split_date = pd.to_datetime('2022-01-01')
    
    train_df = tar_df[tar_df['date'] < split_date].copy()
    val_df = tar_df[tar_df['date'] >= split_date].copy()
    
    print(f"训练集样本: {len(train_df)}, 验证集样本: {len(val_df)}")
    
    # 3. 构造 DataLoader
    seq_len = 30
    batch_size = 512 # 根据显存调整
    
    train_dataset = StockTimeSeriesDataset(train_df, feature_cols, target_col, seq_len=seq_len)
    val_dataset = StockTimeSeriesDataset(val_df, feature_cols, target_col, seq_len=seq_len)
    
    # 训练集可以打乱 (Windows 之间打乱，Window 内部时序不变)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # 验证集不打乱
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 4. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = QuantTransformer(
        feature_dim=len(feature_cols), 
        d_model=64, 
        nhead=4, 
        num_layers=2, 
        seq_len=seq_len
    )
    
    # 5. 开始训练
    trainer = ModelTrainer(model, train_loader, val_loader, learning_rate=1e-4, device=device)
    trained_model = trainer.fit(epochs=30, early_stopping_patience=5)
    
    # 保存模型
    torch.save(trained_model.state_dict(), "quant_transformer_alpha.pth")
    print("训练完成并保存模型！")

if __name__ == '__main__':
    main()