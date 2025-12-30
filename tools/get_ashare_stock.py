import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import logging
from urllib.parse import urljoin
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AShareStockCrawler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
            'Connection': 'keep-alive',
        })
        self.base_url = "https://s.askci.com"
        
    def get_total_pages(self, url):
        """获取总页数"""
        try:
            response = self.session.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 从隐藏输入框获取总页数
            total_page_input = soup.find('input', {'id': 'TotalPageNumHidden'})
            if total_page_input:
                total_pages = int(total_page_input.get('value', 1))
                return total_pages
            
            # 从分页控件获取
            pagination = soup.select('.Pagination')
            if pagination:
                # 尝试从文本中提取页数
                page_text = pagination[0].get_text()
                page_match = re.search(r'共\s*(\d+)\s*页', page_text)
                if page_match:
                    return int(page_match.group(1))
            
            return 1
        except Exception as e:
            logger.error(f"获取总页数失败: {e}")
            return 1
    
    def parse_stock_table(self, soup):
        """解析股票表格"""
        stocks = []
        
        try:
            # 查找股票表格
            table = soup.find('table', {'id': 'myTable04'})
            if not table:
                logger.warning("未找到股票表格")
                return stocks
            
            # 提取表头
            headers = []
            header_row = table.find('thead').find('tr')
            for th in header_row.find_all('th'):
                header_text = th.get_text().strip()
                # 清理表头文本
                header_text = re.sub(r'\s+', ' ', header_text)
                headers.append(header_text)
            
            # 提取数据行
            tbody = table.find('tbody')
            if not tbody:
                return stocks
                
            for row in tbody.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) < 14:  # 确保有足够的列
                    continue
                
                stock_data = {}
                
                try:
                    # 解析每个单元格
                    stock_data['序号'] = cells[0].get_text().strip()
                    
                    # 股票代码
                    code_elem = cells[1].find('a')
                    stock_data['股票代码'] = code_elem.get_text().strip() if code_elem else cells[1].get_text().strip()
                    stock_data['股票详情链接'] = urljoin(self.base_url, code_elem.get('href')) if code_elem else ''
                    
                    # 股票简称
                    name_elem = cells[2].find('a')
                    stock_data['股票简称'] = name_elem.get_text().strip() if name_elem else cells[2].get_text().strip()
                    
                    stock_data['公司名称'] = cells[3].get_text().strip()
                    stock_data['省份'] = cells[4].get_text().strip()
                    stock_data['城市'] = cells[5].get_text().strip()
                    
                    # 财务数据
                    stock_data['主营业务收入'] = self.parse_number(cells[6].get_text().strip())
                    stock_data['净利润'] = self.parse_number(cells[7].get_text().strip())
                    stock_data['员工人数'] = self.parse_number(cells[8].get_text().strip())
                    
                    stock_data['上市日期'] = cells[9].get_text().strip()
                    
                    # 招股书链接
                    prospectus_elem = cells[10].find('a')
                    if prospectus_elem and prospectus_elem.get('href'):
                        stock_data['招股书链接'] = prospectus_elem.get('href')
                    else:
                        stock_data['招股书链接'] = cells[10].get_text().strip()
                    
                    # 财报链接
                    report_elem = cells[11].find('a')
                    stock_data['财报链接'] = report_elem.get('href') if report_elem else ''
                    
                    stock_data['行业分类'] = cells[12].get_text().strip()
                    stock_data['产品类型'] = cells[13].get_text().strip()
                    stock_data['主营业务'] = cells[14].get_text().strip() if len(cells) > 14 else ''
                    
                    stocks.append(stock_data)
                    
                except Exception as e:
                    logger.error(f"解析股票行数据失败: {e}")
                    continue
            
            logger.info(f"成功解析 {len(stocks)} 条股票数据")
            
        except Exception as e:
            logger.error(f"解析表格失败: {e}")
        
        return stocks
    
    def parse_number(self, text):
        """解析数字，处理千分位和科学计数法"""
        if not text or text == '--':
            return None
        
        # 移除逗号和空格
        text = text.replace(',', '').replace(' ', '')
        
        try:
            # 尝试转换为浮点数
            return float(text)
        except ValueError:
            return text
    
    def crawl_page(self, page_num=1):
        """爬取指定页码的数据"""
        try:
            if page_num == 1:
                url = f"{self.base_url}/stock/a/"
            else:
                url = f"{self.base_url}/stock/a/0-0?reportTime=2025-06-30&pageNum={page_num}#QueryCondition"
            
            logger.info(f"爬取第 {page_num} 页: {url}")
            response = self.session.get(url, timeout=10)
            response.encoding = 'utf-8'
            
            if response.status_code != 200:
                logger.error(f"请求失败: {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            stocks = self.parse_stock_table(soup)
            
            return stocks
            
        except Exception as e:
            logger.error(f"爬取第 {page_num} 页失败: {e}")
            return []
    
    def crawl_all_stocks(self, max_pages=None):
        """爬取所有股票数据"""
        all_stocks = []
        
        # 先获取第一页来确定总页数
        first_page_stocks = self.crawl_page(1)
        all_stocks.extend(first_page_stocks)
        
        total_pages = self.get_total_pages(f"{self.base_url}/stock/a/")
        logger.info(f"总共 {total_pages} 页数据")
        
        if max_pages:
            total_pages = min(total_pages, max_pages)
        
        # 爬取后续页面
        for page in range(2, total_pages + 1):
            stocks = self.crawl_page(page)
            all_stocks.extend(stocks)
            logger.info(f"已爬取 {len(all_stocks)} 条股票数据")
            
            # 延迟避免请求过快
            time.sleep(1)
            
            # 每10页保存一次进度
            if page % 10 == 0:
                self.save_progress(all_stocks, f"progress_page_{page}.json")
        
        return all_stocks
    
    def save_to_excel(self, stocks, filename=None):
        """保存到Excel文件"""
        if not filename:
            filename = f"a_share_stocks_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        if not stocks:
            logger.warning("没有数据可保存")
            return
        
        # 创建输出目录
        os.makedirs('stock_data', exist_ok=True)
        filepath = os.path.join('stock_data', filename)
        
        try:
            df = pd.DataFrame(stocks)
            
            # 重新排列列顺序
            column_order = [
                '序号', '股票代码', '股票简称', '公司名称', '省份', '城市',
                '主营业务收入', '净利润', '员工人数', '上市日期', '行业分类',
                '产品类型', '主营业务', '股票详情链接', '招股书链接', '财报链接'
            ]
            
            # 只保留存在的列
            existing_columns = [col for col in column_order if col in df.columns]
            df = df[existing_columns]
            
            # 保存为Excel
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='A股上市公司', index=False)
                
                # 自动调整列宽
                worksheet = writer.sheets['A股上市公司']
                for idx, col in enumerate(df.columns):
                    max_len = max(df[col].astype(str).str.len().max(), len(col)) + 2
                    worksheet.column_dimensions[chr(65 + idx)].width = min(max_len, 50)
            
            logger.info(f"数据已保存到 {filepath}, 共 {len(stocks)} 条记录")
            
            # 同时保存为CSV
            csv_filepath = filepath.replace('.xlsx', '.csv')
            df.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
            logger.info(f"数据同时保存为CSV: {csv_filepath}")
            
        except Exception as e:
            logger.error(f"保存文件失败: {e}")
    
    def save_progress(self, stocks, filename):
        """保存进度"""
        try:
            import json
            filepath = os.path.join('stock_data', filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(stocks, f, ensure_ascii=False, indent=2)
            logger.info(f"进度已保存: {filepath}")
        except Exception as e:
            logger.error(f"保存进度失败: {e}")
    
    def get_industry_stocks(self, industry_id, industry_name, max_pages=5):
        """获取特定行业的股票"""
        logger.info(f"开始爬取行业: {industry_name}")
        
        url = f"{self.base_url}/stock/a/{industry_id}-0?#QueryCondition"
        response = self.session.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        stocks = self.parse_stock_table(soup)
        
        # 如果需要可以继续爬取分页
        total_pages = min(self.get_total_pages(url), max_pages)
        
        for page in range(2, total_pages + 1):
            page_url = f"{self.base_url}/stock/a/{industry_id}-0?reportTime=2025-06-30&pageNum={page}#QueryCondition"
            page_stocks = self.crawl_specific_url(page_url)
            stocks.extend(page_stocks)
            time.sleep(1)
        
        logger.info(f"行业 {industry_name} 爬取完成，共 {len(stocks)} 条数据")
        return stocks
    
    def crawl_specific_url(self, url):
        """爬取特定URL的数据"""
        try:
            response = self.session.get(url, timeout=10)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.content, 'html.parser')
            return self.parse_stock_table(soup)
        except Exception as e:
            logger.error(f"爬取URL失败 {url}: {e}")
            return []

def main():
    """主函数"""
    crawler = AShareStockCrawler()
    
    print("开始爬取A股上市公司数据...")
    
    all_stocks = crawler.crawl_all_stocks()
    
    if all_stocks:
        print(f"\n爬取完成！共获取 {len(all_stocks)} 条股票数据")
        
        # 显示前5条数据
        print("\n前5条股票数据:")
        for i, stock in enumerate(all_stocks[:5], 1):
            print(f"{i}. {stock['股票代码']} {stock['股票简称']} - {stock['公司名称']}")
            print(f"   收入: {stock.get('主营业务收入', 'N/A')} | 净利润: {stock.get('净利润', 'N/A')}")
            print(f"   行业: {stock.get('行业分类', 'N/A')} | 地点: {stock.get('省份', 'N/A')}-{stock.get('城市', 'N/A')}")
            print()
        
        # 保存数据
        crawler.save_to_excel(all_stocks)
        
        # 按行业统计
        industry_stats = {}
        for stock in all_stocks:
            industry = stock.get('行业分类', '未知')
            industry_stats[industry] = industry_stats.get(industry, 0) + 1
        
        print("各行业股票数量统计:")
        for industry, count in sorted(industry_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {industry}: {count}家")
    
    else:
        print("没有获取到数据")

if __name__ == "__main__":
    main()