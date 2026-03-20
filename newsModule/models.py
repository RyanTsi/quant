"""
SQLAlchemy ORM模型定义
"""
from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Text, Date, DateTime, Index, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class NewsModel(Base):
    """金融新闻数据库模型"""
    
    __tablename__ = 'financial_news'
    
    # 复合唯一约束：同一来源同一URL只能存在一条
    __table_args__ = (
        UniqueConstraint('url', name='uq_news_url'),
        Index('idx_source_date', 'source', 'publish_date'),
    )
    
    # 主键
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 必需字段
    title = Column(String(500), nullable=False, index=True)
    content = Column(Text, nullable=False)
    url = Column(String(1024), nullable=False, unique=True, index=True)
    source = Column(String(50), nullable=False, index=True)
    publish_date = Column(Date, nullable=False, index=True)
    
    # 元数据
    crawled_at = Column(DateTime, nullable=False, default=datetime.now)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)
    
    # 分类标签
    category = Column(String(50), nullable=True, index=True)
    
    def __repr__(self):
        return f"<NewsModel(id={self.id}, title='{self.title[:30]}...', source='{self.source}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'url': self.url,
            'source': self.source,
            'publish_date': self.publish_date.isoformat() if self.publish_date else None,
            'crawled_at': self.crawled_at.isoformat() if self.crawled_at else None,
            'summary': self.summary,
            'category': self.category
        }