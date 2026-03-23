"""
抽象接口定义
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import date

from .schemas import NewsItem


class ScraperInterface(ABC):
    """
    爬虫接口
    
    所有具体的新闻源爬虫必须实现此接口
    """
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """
        返回新闻源标识符
        
        Returns:
            str: 新闻源名称，如 'sina', 'yahoo', 'bloomberg'
        """
        pass
    
    @property
    def source_display_name(self) -> str:
        """返回用于显示的新闻源名称"""
        return self.source_name
    
    @property
    def base_url(self) -> str:
        """返回新闻源基础URL"""
        return ""
    
    @abstractmethod
    def fetch(self, target_date: date) -> List[NewsItem]:
        """
        抓取指定日期的新闻
        
        Args:
            target_date: 目标日期
            
        Returns:
            List[NewsItem]: 新闻列表
            
        Raises:
            ScraperException: 抓取失败时抛出
        """
        pass
    
    def fetch_date_range(self, start_date: date, end_date: date) -> List[NewsItem]:
        """
        抓取日期范围内的新闻（默认实现）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            List[NewsItem]: 新闻列表
        """
        all_news = []
        current = start_date
        while current <= end_date:
            news_items = self.fetch(current)
            all_news.extend(news_items)
            current = date(current.year, current.month, current.day + 1) if current.day < 28 else date(current.year, current.month + 1 if current.month < 12 else current.year + 1, 1)
        return all_news
    
    def validate_date(self, target_date: date) -> bool:
        """
        验证日期是否可抓取
        
        Args:
            target_date: 目标日期
            
        Returns:
            bool: 是否可抓取
        """
        today = date.today()
        return target_date <= today


class RepositoryInterface(ABC):
    """
    数据仓储接口
    """
    
    @abstractmethod
    def save(self, news_item: NewsItem) -> bool:
        """保存单条新闻"""
        pass
    
    @abstractmethod
    def save_bulk(self, news_items: List[NewsItem]) -> Dict[str, int]:
        """批量保存新闻，返回统计信息"""
        pass
    
    @abstractmethod
    def get_by_id(self, news_id: int) -> NewsItem:
        """根据ID获取新闻"""
        pass
    
    @abstractmethod
    def get_by_date_range(self, start_date: date, end_date: date, source: str = None) -> List[NewsItem]:
        """根据日期范围获取新闻"""
        pass
    
    @abstractmethod
    def get_by_url(self, url: str) -> NewsItem:
        """根据URL获取新闻"""
        pass
    
    @abstractmethod
    def delete_old_news(self, before_date: date) -> int:
        """删除指定日期之前的新闻"""
        pass


class ScraperException(Exception):
    """爬虫异常"""
    
    def __init__(self, message: str, source: str = None, date: date = None):
        self.message = message
        self.source = source
        self.date = date
        super().__init__(self.message)
    
    def __str__(self):
        base_msg = f"ScraperError: {self.message}"
        if self.source:
            base_msg += f" (source: {self.source})"
        if self.date:
            base_msg += f" (date: {self.date})"
        return base_msg