import re
from datetime import datetime

def format_stock_code(code, prefix: bool = True, uppercase: bool = True, has_dot: bool = False):
    code_str = str(code).upper().strip()
    digits = re.sub(r'\D', '', code_str).zfill(6)
    market = ''.join(re.findall(r'[A-Z]+', code_str))
    if not market:
        if digits.startswith(('00', '30')):
            market = 'sz'
        elif digits.startswith(('60', '68')):
            market = 'sh'
        elif digits.startswith(('82', '83', '87', '88', '92')):
            market = 'bj'
        else:
            return digits
    market = market.upper() if uppercase else market.lower()
    if prefix:
        return f"{market}.{digits}" if has_dot else f"{market}{digits}"
    else:
        return f"{digits}.{market}" if has_dot else f"{digits}{market}"
    
# 支持的日期格式映射
DATE_FORMATS = {
    "YYYY-MM-DD":   "%Y-%m-%d",
    "DD/MM/YYYY":   "%d/%m/%Y",
    "MM/DD/YYYY":   "%m/%d/%Y",
    "YYYY年MM月DD日": "%Y年%m月%d日",
    "DD-MM-YYYY":   "%d-%m-%Y",
    "YYYYMMDD":     "%Y%m%d",
}

def format_date(date: str, format: str) -> str:
    """
    将日期字符串转换为目标格式。
    自动识别输入格式，无需手动指定。

    :param date: 输入的日期字符串，如 "2024-01-15"
    :param format: 目标格式的键，如 "DD/MM/YYYY"
    :return: 转换后的日期字符串
    """
    if format not in DATE_FORMATS:
        raise ValueError(f"不支持的目标格式: '{format}'，可选: {list(DATE_FORMATS.keys())}")

    # 自动尝试所有已知格式解析输入
    parsed_date = None
    for fmt_key, fmt_str in DATE_FORMATS.items():
        try:
            parsed_date = datetime.strptime(date, fmt_str)
            break
        except ValueError:
            continue

    if parsed_date is None:
        raise ValueError(f"无法识别的日期格式: '{date}'")

    return parsed_date.strftime(DATE_FORMATS[format])


def pure_stock_code(code):
    return "".join(c for c in str(code) if c.isdigit())