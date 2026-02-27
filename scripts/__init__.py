import sys
from pathlib import Path

# 获取 quant 根目录
root = str(Path(__file__).resolve().parent.parent)

# 注入到系统路径
if root not in sys.path:
    sys.path.insert(0, root)