"""
数据访问层实现
"""
import logging
from typing import List, Dict, Optional
from datetime import date, datetime

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from .models import NewsModel
from .schemas import NewsItem, NewsFilter
from .interfaces import RepositoryInterface

logger = logging.getLogger(__name__)


class NewsRepository(RepositoryInterface):
    """
    新闻数据仓储实现
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def _to_orm(self, news_item: NewsItem) -> NewsModel:
        """将Pydantic模型转换为ORM模型"""
        return NewsModel(
            title=news_item.title,
            content=news_item.content,
            url=news_item.url,
            source=news_item.source,
            publish_date=news_item.publish_date,
            summary=news_item.summary,
            category=news_item.category,
            crawled_at=datetime.now()
        )
    
    def _to_schema(self, orm_model: NewsModel) -> NewsItem:
        """将ORM模型转换为Pydantic模型"""
        return NewsItem(
            title=orm_model.title,
            content=orm_model.content,
            url=orm_model.url,
            source=orm_model.source,
            publish_date=orm_model.publish_date,
            summary=orm_model.summary,
            category=orm_model.category
        )
    
    def save(self, news_item: NewsItem) -> bool:
        """保存单条新闻"""
        try:
            orm_model = self._to_orm(news_item)
            self.db.add(orm_model)
            self.db.commit()
            self.db.refresh(orm_model)
            logger.info(f"Saved news: {news_item.title[:30]}...")
            return True
        except IntegrityError:
            self.db.rollback()
            logger.debug(f"Duplicate URL skipped: {news_item.url}")
            return False
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to save news: {str(e)}")
            raise
    
    def save_bulk(self, news_items: List[NewsItem]) -> Dict[str, int]:
        """
        批量保存新闻
        
        Returns:
            Dict包含: total, saved, skipped, failed
        """
        stats = {
            'total': len(news_items),
            'saved': 0,
            'skipped': 0,
            'failed': 0
        }
        
        for item in news_items:
            try:
                orm_model = self._to_orm(item)
                self.db.add(orm_model)
                stats['saved'] += 1
            except IntegrityError:
                self.db.rollback()
                stats['skipped'] += 1
                logger.debug(f"Duplicate URL skipped: {item.url}")
            except Exception as e:
                self.db.rollback()
                stats['failed'] += 1
                logger.error(f"Failed to save {item.url}: {str(e)}")
        
        try:
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            logger.error(f"Batch commit failed: {str(e)}")
        
        logger.info(f"Bulk save completed: saved={stats['saved']}, skipped={stats['skipped']}, failed={stats['failed']}")
        return stats
    
    def get_by_id(self, news_id: int) -> Optional[NewsItem]:
        """根据ID获取新闻"""
        orm_model = self.db.query(NewsModel).filter(NewsModel.id == news_id).first()
        return self._to_schema(orm_model) if orm_model else None
    
    def get_by_date_range(
        self, 
        start_date: date, 
        end_date: date, 
        source: Optional[str] = None,
        category: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[NewsItem]:
        """根据日期范围获取新闻"""
        query = self.db.query(NewsModel)
        
        # 日期过滤
        if start_date:
            query = query.filter(NewsModel.publish_date >= start_date)
        if end_date:
            query = query.filter(NewsModel.publish_date <= end_date)
        
        # 来源过滤
        if source:
            query = query.filter(NewsModel.source == source.lower())
        
        # 分类过滤
        if category:
            query = query.filter(NewsModel.category == category)
        
        # 关键词搜索
        if keyword:
            keyword_filter = or_(
                NewsModel.title.like(f'%{keyword}%'),
                NewsModel.content.like(f'%{keyword}%')
            )
            query = query.filter(keyword_filter)
        
        # 分页
        query = query.order_by(NewsModel.publish_date.desc())
        query = query.offset(offset).limit(limit)
        
        results = query.all()
        return [self._to_schema(r) for r in results]
    
    def get_by_url(self, url: str) -> Optional[NewsItem]:
        """根据URL获取新闻"""
        orm_model = self.db.query(NewsModel).filter(NewsModel.url == url).first()
        return self._to_schema(orm_model) if orm_model else None
    
    def get_by_source(self, source: str) -> List[NewsItem]:
        """获取指定来源的所有新闻"""
        results = self.db.query(NewsModel).filter(
            NewsModel.source == source.lower()
        ).order_by(NewsModel.publish_date.desc()).all()
        return [self._to_schema(r) for r in results]
    
    def delete_old_news(self, before_date: date) -> int:
        """删除指定日期之前的新闻"""
        count = self.db.query(NewsModel).filter(
            NewsModel.publish_date < before_date
        ).delete()
        self.db.commit()
        logger.info(f"Deleted {count} old news records before {before_date}")
        return count
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        from sqlalchemy import func
        
        total = self.db.query(func.count(NewsModel.id)).scalar()
        by_source = self.db.query(
            NewsModel.source,
            func.count(NewsModel.id)
        ).group_by(NewsModel.source).all()
        
        return {
            'total': total,
            'by_source': dict(by_source)
        }