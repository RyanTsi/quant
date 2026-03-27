"""
Pydantic数据校验模型
"""
from datetime import date, datetime
from typing import Optional, List
from pydantic import BaseModel, HttpUrl, Field, field_validator


class NewsItem(BaseModel):
    """单条新闻数据模型"""
    
    title: str = Field(..., min_length=1, max_length=500, description="新闻标题")
    content: str = Field(..., min_length=1, description="新闻正文内容")
    url: str = Field(..., max_length=1024, description="原文链接")
    source: str = Field(..., min_length=1, max_length=50, description="来源标识")
    publish_date: date = Field(..., description="发布日期")
    
    # 可选字段
    summary: Optional[str] = Field(None, max_length=500, description="摘要")
    category: Optional[str] = Field(None, max_length=50, description="分类")
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """验证URL格式"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v
    
    @field_validator('source')
    @classmethod
    def validate_source(cls, v: str) -> str:
        """标准化来源名称"""
        return v.strip().lower()
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "title": "A股三大指数收盘涨跌不一",
                "content": "今日A股市场...",
                "url": "https://finance.sina.com.cn/stock/...",
                "source": "sina",
                "publish_date": "2024-01-15",
                "category": "stock"
            }
        }


class NewsFilter(BaseModel):
    """新闻查询过滤器"""
    
    start_date: Optional[date] = Field(None, description="开始日期")
    end_date: Optional[date] = Field(None, description="结束日期")
    source: Optional[str] = Field(None, description="新闻来源")
    category: Optional[str] = Field(None, description="分类")
    keyword: Optional[str] = Field(None, description="关键词搜索")
    limit: int = Field(100, ge=1, le=1000, description="返回数量限制")
    offset: int = Field(0, ge=0, description="偏移量")
    
    @field_validator('end_date')
    @classmethod
    def validate_dates(cls, v, info):
        """验证日期范围"""
        start = info.data.get('start_date')
        if start and v and v < start:
            raise ValueError('end_date must be >= start_date')
        return v


class NewsResponse(BaseModel):
    """新闻响应模型"""
    
    total: int = Field(..., description="总数")
    items: List[NewsItem] = Field(..., description="新闻列表")
    
    class Config:
        from_attributes = True


class CrawlResult(BaseModel):
    """爬取结果模型"""
    
    source: str = Field(..., description="新闻源")
    target_date: date = Field(..., description="目标日期")
    fetched_count: int = Field(..., description="抓取数量")
    saved_count: int = Field(..., description="入库数量")
    skipped_count: int = Field(..., description="跳过数量(重复)")
    status: str = Field(..., description="状态: success/failed")
    message: str = Field(..., description="详细信息")
    
    class Config:
        from_attributes = True