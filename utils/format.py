import re

def format_stock_code(code, prefix: bool = True, uppercase: bool = True):
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
        return f"{market}{digits}"
    else:
        return f"{digits}.{market}"