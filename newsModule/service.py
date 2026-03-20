"""
业务逻辑层
"""
import logging
from typing import List, Dict, Type, Optional
from datetime import date, timedelta

from sqlalchemy.orm import Session

from .interfaces import ScraperInterface
from .repository import NewsRepository
from .schemas import NewsItem, NewsFilter, CrawlResult
from .scrapers.mock_scraper import MockFinancialScraper

logger = logging.getLogger(__name__)


class NewsCrawlerService:
    """
    金融新闻爬取服务
    
    协调爬虫和数据仓储，提供统一的业务接口
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.repo = NewsRepository(db_session)
        self.scrapers: Dict[str, ScraperInterface] = {}
        
        # 注册默认爬虫
        self._register_default_scrapers()
    
    def _register_default_scrapers(self):
        """注册默认爬虫"""
        self.register_scraper(MockFinancialScraper())
    
    def register_scraper(self, scraper: ScraperInterface):
        """
        注册新闻爬虫
        
        Args:
            scraper: 爬虫实例
        """
        self.scrapers[scraper.source_name] = scraper
        logger.info(f"Registered scraper: {scraper.source_name} ({scraper.source_display_name})")
    
    def get_available_sources(self) -> List[Dict]:
        """获取所有可用的新闻源"""
        return [
            {
                'key': key,
                'name': scraper.source_display_name,
                'base_url': scraper.base_url
            }
            for key, scraper in self.scrapers.items()
        ]
    
    def crawl_news(
        self, 
        source_key: str, 
        target_date: date,
        auto_save: bool = True
    ) -> CrawlResult:
        """
        爬取指定日期的新闻
        
        Args:
            source_key: 新闻源标识
            target_date: 目标日期
            auto_save: 是否自动保存到数据库
            
        Returns:
            CrawlResult: 爬取结果
        """
        scraper = self.scrapers.get(source_key)
        
        if not scraper:
            return CrawlResult(
                source=source_key,
                target_date=target_date,
                fetched_count=0,
                saved_count=0,
                skipped_count=0,
                status='failed',
                message=f"Unknown source: {source_key}"
            )
        
        try:
            # 验证日期
            if not scraper.validate_date(target_date):
                return CrawlResult(
                    source=source_key,
                    target_date=target_date,
                    fetched_count=0,
                    saved_count=0,
                    skipped_count=0,
                    status='failed',
                    message=f"Cannot fetch future date: {target_date}"
                )
            
            # 抓取新闻
            logger.info(f"Starting crawl: {source_key} - {target_date}")
            news_items = scraper.fetch(target_date)
            fetched_count = len(news_items)
            
            if not auto_save:
                return CrawlResult(
                    source=source_key,
                    target_date=target_date,
                    fetched_count=fetched_count,
                    saved_count=0,
                    skipped_count=0,
                    status='success',
                    message=f"Fetched {fetched_count} items (auto_save=False)"
                )
            
            # 保存到数据库
            save_stats = self.repo.save_bulk(news_items)
            
            return CrawlResult(
                source=source_key,
                target_date=target_date,
                fetched_count=fetched_count,
                saved_count=save_stats['saved'],
                skipped_count=save_stats['skipped'],
                status='success',
                message=f"Successfully crawled and saved"
            )
            
        except Exception as e:
            logger.error(f"Crawl failed: {source_key} - {target_date}: {str(e)}")
            return CrawlResult(
                source=source_key,
                target_date=target_date,
                fetched_count=0,
                saved_count=0,
                skipped_count=0,
                status='failed',
                message=f"Error: {str(e)}"
            )
    
    def crawl_date_range(
        self,
        source_key: str,
        start_date: date,
        end_date: date
    ) -> List[CrawlResult]:
        """
        爬取日期范围内的新闻
        
        Args:
            source_key: 新闻源标识
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            List[CrawlResult]: 每天的爬取结果
        """
        results = []
        current = start_date
        
        while current <= end_date:
            result = self.crawl_news(source_key, current)
            results.append(result)
            
            # 避免请求过于频繁
            import time
            time.sleep(1)
            
            current = current + timedelta(days=1)
        
        return results
    
    def get_stored_news(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        source: Optional[str] = None,
        category: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[NewsItem]:
        """
        从数据库获取新闻
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            source: 新闻源
            category: 分类
            keyword: 关键词
            limit: 数量限制
            offset: 偏移量
            
        Returns:
            List[NewsItem]: 新闻列表
        """
        return self.repo.get_by_date_range(
            start_date=start_date,
            end_date=end_date,
            source=source,
            category=category,
            keyword=keyword,
            limit=limit,
            offset=offset
        )
    
    def get_news_by_id(self, news_id: int) -> Optional[NewsItem]:
        """根据ID获取新闻"""
        return self.repo.get_by_id(news_id)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return self.repo.get_statistics()
    
    def cleanup_old_news(self, days: int = 90) -> int:
        """
        清理旧新闻
        
        Args:
            days: 保留最近N天的新闻
            
        Returns:
            int: 删除的数量
        """
        cutoff_date = date.today() - timedelta(days=days)
        return self.repo.delete_old_news(cutoff_date)