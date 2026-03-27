"""
爬虫基类实现
"""
import logging
import time
from abc import ABC
from typing import List, Optional
from datetime import date, timedelta
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

from ..schemas import NewsItem
from ..interfaces import ScraperInterface, ScraperException
from ..config import config

logger = logging.getLogger(__name__)


class BaseScraper(ScraperInterface, ABC):
    """
    爬虫基类
    
    提供通用的爬虫功能：
    - HTTP请求处理
    - 重试机制
    - 基础解析工具
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.scraper.user_agent
        })
        self.timeout = config.scraper.timeout
        self.retry_times = config.scraper.retry_times
        self.retry_delay = config.scraper.retry_delay
    
    def _make_request(self, url: str, method: str = 'GET', **kwargs) -> Optional[requests.Response]:
        """
        发起HTTP请求（带重试机制）
        
        Args:
            url: 请求URL
            method: 请求方法
            **kwargs: 其他请求参数
            
        Returns:
            Response对象或None
        """
        last_error = None
        
        for attempt in range(self.retry_times):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    timeout=self.timeout,
                    **kwargs
                )
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                last_error = e
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.retry_times}): {url} - {str(e)}")
                if attempt < self.retry_times - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        logger.error(f"Request failed after {self.retry_times} attempts: {url}")
        raise ScraperException(
            message=f"Failed to fetch URL after {self.retry_times} attempts: {str(last_error)}",
            source=self.source_name,
            date=date.today()
        )
    
    def _parse_html(self, html: str) -> BeautifulSoup:
        """解析HTML"""
        return BeautifulSoup(html, 'html.parser')
    
    def _build_date_url(self, base_url: str, target_date: date, format: str = None) -> str:
        """
        构建包含日期的URL
        
        Args:
            base_url: 基础URL
            target_date: 目标日期
            format: 日期格式，默认为 YYYY/MM/DD
            
        Returns:
            构建后的URL
        """
        if format is None:
            format = "%Y/%m/%d"
        date_str = target_date.strftime(format)
        return urljoin(base_url, date_str)
    
    def _extract_summary(self, content: str, max_length: int = 200) -> str:
        """提取内容摘要"""
        # 移除多余空白字符
        text = ' '.join(content.split())
        if len(text) <= max_length:
            return text
        return text[:max_length] + '...'
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        if not text:
            return ""
        # 移除多余空白
        text = ' '.join(text.split())
        # 移除特殊字符
        text = text.replace('\u200b', '')  # 零宽空格
        return text.strip()
    
    def fetch_with_date_range(self, days: int = 7) -> List[NewsItem]:
        """
        便捷方法：获取最近N天的新闻
        
        Args:
            days: 天数
            
        Returns:
            新闻列表
        """
        all_news = []
        today = date.today()
        
        for i in range(days):
            target_date = today - timedelta(days=i)
            try:
                news_items = self.fetch(target_date)
                all_news.extend(news_items)
            except ScraperException as e:
                logger.warning(f"Failed to fetch {target_date}: {e}")
                continue
        
        return all_news