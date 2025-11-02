import requests
from bs4 import BeautifulSoup
import json
import time
import datetime
import re
import logging
from urllib.parse import urljoin
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialNewsCrawler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
            'Connection': 'keep-alive',
        })
        self.base_url = "http://www.financialnews.com.cn"
        self.today = datetime.datetime.now().date()
        
        # 定义所有主题及其对应的节点ID
        self.topics = {
            '金融管理': '3001',
            '要闻': '3002', 
            '评论': '3003',
            '深度': '3004',
            '银行': '3005',
            '证券': '3006',
            '保险': '3007',
            '国际': '3008',
            '地方': '3009',
            '公司': '3010',
            '理论': '3011',
            '金融科技': '3012',
            '农村金融': '3013',
            '可视化': '3014',
            '文化': '3015',
            '信批平台': '3016',
            '中国金融家': '3017',
            '专题': '3018',
            '头条': '3025',
            '品牌': '3026'
        }
    
    def get_topic_news(self, topic_name, topic_id, max_pages=3):
        """获取指定主题的新闻"""
        news_list = []
        
        for page in range(1, max_pages + 1):
            try:
                if page == 1:
                    url = f"{self.base_url}/node_{topic_id}.html"
                else:
                    url = f"{self.base_url}/node_{topic_id}_{page}.html"
                
                logger.info(f"爬取 {topic_name} 第 {page} 页: {url}")
                response = self.session.get(url, timeout=10)
                response.encoding = 'utf-8'
                
                if response.status_code != 200:
                    logger.warning(f"请求失败: {response.status_code}")
                    break
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 查找新闻列表
                news_items = soup.select('.news-list dl')
                if not news_items:
                    logger.info(f"{topic_name} 第 {page} 页没有找到新闻条目")
                    break
                
                page_news_count = 0
                for item in news_items:
                    try:
                        # 提取标题和链接
                        title_elem = item.select_one('h4 a')
                        if not title_elem:
                            continue
                            
                        title = title_elem.get_text().strip()
                        link = title_elem.get('href')
                        if not link.startswith('http'):
                            link = urljoin(self.base_url, link)
                        
                        # 提取摘要
                        summary_elem = item.select_one('p')
                        summary = summary_elem.get_text().strip() if summary_elem else ""
                        
                        # 提取元信息（日期、作者、来源）
                        meta_elem = item.select_one('h6')
                        if meta_elem:
                            spans = meta_elem.select('span')
                            date_str = spans[0].get_text().strip() if len(spans) > 0 else ""
                            author = spans[1].get_text().strip() if len(spans) > 1 else ""
                            source = spans[2].get_text().strip() if len(spans) > 2 else ""
                            
                            # 如果没有第三个span，第二个可能是来源
                            if len(spans) == 2 and "来源" in spans[1].get_text():
                                source = spans[1].get_text().strip()
                                author = ""
                        else:
                            date_str, author, source = "", "", ""
                        
                        # 检查是否为当天新闻
                        if self.is_today_news(date_str):
                            news_item = {
                                'title': title,
                                'link': link,
                                'summary': summary,
                                'date': date_str,
                                'author': author,
                                'source': source,
                                'topic': topic_name
                            }
                            
                            # 获取新闻详情
                            detail_content = self.get_news_detail(link)
                            if detail_content:
                                news_item.update(detail_content)
                            
                            news_list.append(news_item)
                            page_news_count += 1
                            logger.info(f"找到新闻: {title}")
                    
                    except Exception as e:
                        logger.error(f"解析新闻条目出错: {e}")
                        continue
                
                logger.info(f"{topic_name} 第 {page} 页找到 {page_news_count} 条当天新闻")
                
                # 如果没有找到当天新闻，停止翻页
                if page_news_count == 0 and page > 1:
                    break
                    
                # 延迟避免请求过快
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"爬取 {topic_name} 第 {page} 页出错: {e}")
                continue
        
        return news_list
    
    def get_news_detail(self, url):
        """获取新闻详情内容"""
        try:
            response = self.session.get(url, timeout=10)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取详情内容
            detail_content = {}
            
            # 提取正文
            content_elem = soup.select_one('.detail-cont')
            if content_elem:
                # 移除不需要的元素
                for elem in content_elem.select('.banquan, script, style'):
                    elem.decompose()
                
                content = content_elem.get_text().strip()
                # 清理多余的空格和换行
                content = re.sub(r'\s+', ' ', content)
                detail_content['content'] = content
            else:
                detail_content['content'] = ""
            
            # 提取发布时间（详情页可能有更精确的时间）
            time_elem = soup.select_one('.detail-title p span:contains("发布时间")')
            if time_elem:
                detail_content['publish_time'] = time_elem.get_text().replace('发布时间：', '').strip()
            else:
                detail_content['publish_time'] = ""
            
            return detail_content
            
        except Exception as e:
            logger.error(f"获取新闻详情出错 {url}: {e}")
            return {}
    
    def is_today_news(self, date_str):
        """判断是否为当天新闻"""
        if not date_str:
            return False
        
        # 匹配日期格式 "2025-10-30"
        date_match = re.search(r'(\d{4}-\d{1,2}-\d{1,2})', date_str)
        if date_match:
            news_date_str = date_match.group(1)
            try:
                news_date = datetime.datetime.strptime(news_date_str, '%Y-%m-%d').date()
                return news_date == self.today
            except ValueError:
                return False
        
        return False
    
    def crawl_all_topics(self, max_pages_per_topic=2):
        """爬取所有主题的新闻"""
        all_news = []
        
        for topic_name, topic_id in self.topics.items():
            logger.info(f"开始爬取主题: {topic_name}")
            try:
                topic_news = self.get_topic_news(topic_name, topic_id, max_pages_per_topic)
                all_news.extend(topic_news)
                logger.info(f"主题 {topic_name} 爬取完成，共 {len(topic_news)} 条新闻")
                
                # 主题间延迟
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"爬取主题 {topic_name} 时出错: {e}")
                continue
        
        # 按日期排序
        all_news.sort(key=lambda x: x.get('date', ''), reverse=True)
        
        return all_news
    
    def save_to_file(self, news_list, filename=None):
        """保存新闻到文件"""
        if not filename:
            filename = f"financial_news_{self.today.strftime('%Y%m%d')}.json"
        
        # 创建输出目录
        os.makedirs('news_data', exist_ok=True)
        filepath = os.path.join('news_data', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(news_list, f, ensure_ascii=False, indent=2)
        
        logger.info(f"新闻已保存到 {filepath}, 共 {len(news_list)} 条")
        
        # 同时生成简化的文本报告
        self.generate_report(news_list)
    
    def generate_report(self, news_list):
        """生成文本格式的报告"""
        report_file = f"news_data/financial_news_report_{self.today.strftime('%Y%m%d')}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"中国金融新闻网 - 当天新闻汇总\n")
            f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"共找到 {len(news_list)} 条新闻\n")
            f.write("=" * 80 + "\n\n")
            
            # 按主题分组
            topics_news = {}
            for news in news_list:
                topic = news['topic']
                if topic not in topics_news:
                    topics_news[topic] = []
                topics_news[topic].append(news)
            
            for topic, news_in_topic in topics_news.items():
                f.write(f"\n【{topic}】({len(news_in_topic)}条)\n")
                f.write("-" * 50 + "\n")
                
                for i, news in enumerate(news_in_topic, 1):
                    f.write(f"{i}. {news['title']}\n")
                    f.write(f"   时间: {news['date']} | 来源: {news['source']}\n")
                    if news.get('author'):
                        f.write(f"   作者: {news['author']}\n")
                    if news['summary']:
                        f.write(f"   摘要: {news['summary'][:100]}...\n")
                    f.write(f"   链接: {news['link']}\n")
                    f.write("\n")

def main():
    """主函数"""
    crawler = FinancialNewsCrawler()
    
    logger.info("开始爬取中国金融新闻网当天新闻...")
    news_list = crawler.crawl_all_topics(max_pages_per_topic=2)
    
    logger.info(f"爬取完成！共找到 {len(news_list)} 条当天新闻")
    
    # 打印摘要
    print(f"\n=== 中国金融新闻网当天新闻摘要 ({crawler.today}) ===")
    
    # 按主题统计
    topic_count = {}
    for news in news_list:
        topic = news['topic']
        topic_count[topic] = topic_count.get(topic, 0) + 1
    
    print("各主题新闻数量:")
    for topic, count in sorted(topic_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  {topic}: {count}条")
    
    # 显示前5条新闻
    print(f"\n最新5条新闻:")
    for i, news in enumerate(news_list[:5], 1):
        print(f"{i}. [{news['topic']}] {news['title']}")
        print(f"   时间: {news['date']} | 来源: {news['source']}")
    
    # 保存到文件
    if news_list:
        crawler.save_to_file(news_list)
    else:
        logger.info("没有找到当天的新闻")
    
    return news_list

if __name__ == "__main__":
    news = main()