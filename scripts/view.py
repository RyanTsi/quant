import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.workflow import R
from qlib.contrib.report import analysis_position, analysis_model
import os
from config.settings import settings

# 1. 唤醒 Qlib
provider_uri = r"C:/Users/sola/Documents/quant/.data/qlib_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)

# 2. 告诉 Qlib 去哪里找实验数据
R.set_uri(r"file:///C:/Users/sola/Documents/quant/mlruns")

# 2. 获取记录器
recorder_id = "6c6aaaec2fc4431eb78d5b17d709b348"
exp_id = "379677092195942384"
recorder = R.get_exp(experiment_id=exp_id).get_recorder(recorder_id=recorder_id)

pred_df = recorder.load_object("pred.pkl")
label_df = recorder.load_object("label.pkl")

label_df.columns = ['label']
pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)

print("正在生成 IC/IR 预测能力分析图表...")
# 将拼接好的完整数据交给 Qlib 画图
figs = analysis_model.model_performance_graph(pred_label, show_notebook=False)

path = os.path.join(settings.analysis_path, recorder_id)
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
    
# 4. 保存为本地 HTML 文件
if isinstance(figs, (list, tuple)):
    for i, f in enumerate(figs):
        if f is not None:
            f.write_html(os.path.join(path, f"B_预测能力分析_{i}.html"))
            print(f"✅ 成功保存子图：B_预测能力分析_{i}.html")
else:
    if figs is not None:
        figs.write_html(os.path.join(path, "B_预测能力分析.html"))
        print("✅ 成功保存图表：B_预测能力分析.html")
    
print("🎉 全部判卷完成！去看看那几个月的 IC 值吧。")


report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
figs = analysis_position.report_graph(report_normal_df, show_notebook=False)

print("正在保存收益图表...")
# Qlib 可能会返回多个图表组成的元组
if isinstance(figs, tuple):
    for i, f in enumerate(figs):
        if f is not None:
            f.write_html(os.path.join(path, f"A_策略收益曲线_{i}.html"))
else:
    if figs is not None:
        figs.write_html(os.path.join(path, "A_策略收益曲线.html"))