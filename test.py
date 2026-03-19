"""
金融新闻爬取模块使用示例
"""
import logging
from datetime import date, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 导入模块
from financial_crawler.database import db_manager
from financial_crawler.service import NewsCrawlerService
from financial_crawler.schemas import NewsItem, CrawlResult


def main():
    """主函数演示"""
    
    # ===== 1. 初始化数据库 =====
    print("=" * 50)
    print("1. 初始化数据库")
    print("=" * 50)
    
    # 创建数据库表
    db_manager.create_tables()
    print("✓ 数据库表已创建")
    
    # 获取数据库会话
    db = db_manager.get_session_direct()
    
    # ===== 2. 创建服务实例 =====
    print("\n" + "=" * 50)
    print("2. 创建爬取服务")
    print("=" * 50)
    
    service = NewsCrawlerService(db)
    
    # 查看可用的新闻源
    sources = service.get_available_sources()
    print(f"✓ 已注册的新闻源: {len(sources)}")
    for s in sources:
        print(f"  - {s['key']}: {s['name']}")
    
    # ===== 3. 爬取今天和昨天的新闻 =====
    print("\n" + "=" * 50)
    print("3. 爬取新闻")
    print("=" * 50)
    
    today = date.today()
    yesterday = today - timedelta(days=1)
    
    # 爬取今天的新闻
    print(f"\n正在爬取 {today} 的新闻...")
    result_today = service.crawl_news("mock", today)
    print(f"  状态: {result_today.status}")
    print(f"  抓取: {result_today.fetched_count} 条")
    print(f"  入库: {result_today.saved_count} 条")
    print(f"  跳过: {result_today.skipped_count} 条")
    
    # 爬取昨天的新闻
    print(f"\n正在爬取 {yesterday} 的新闻...")
    result_yesterday = service.crawl_news("mock", yesterday)
    print(f"  状态: {result_yesterday.status}")
    print(f"  抓取: {result_yesterday.fetched_count} 条")
    print(f"  入库: {result_yesterday.saved_count} 条")
    
    # ===== 4. 从数据库查询新闻 =====
    print("\n" + "=" * 50)
    print("4. 查询新闻")
    print("=" * 50)
    
    # 查询最近7天的新闻
    news_list = service.get_stored_news(
        start_date=today - timedelta(days=7),
        end_date=today,
        limit=10
    )
    
    print(f"\n最近7天共有 {len(news_list)} 条新闻 (显示前10条):")
    for i, news in enumerate(news_list, 1):
        print(f"\n  [{i}] {news.title}")
        print(f"      来源: {news.source} | 日期: {news.publish_date}")
        print(f"      摘要: {news.summary[:50] if news.summary else '无'}...")
    
    # ===== 5. 查看统计信息 =====
    print("\n" + "=" * 50)
    print("5. 统计信息")
    print("=" * 50)
    
    stats = service.get_statistics()
    print(f"\n总新闻数: {stats['total']}")
    print("按来源统计:")
    for source, count in stats['by_source'].items():
        print(f"  - {source}: {count} 条")
    
    # ===== 6. 再次爬取测试去重 =====
    print("\n" + "=" * 50)
    print("6. 测试去重功能")
    print("=" * 50)
    
    # 再次爬取今天的新闻（应该被去重）
    print(f"\n再次爬取 {today} 的新闻...")
    result_duplicate = service.crawl_news("mock", today)
    print(f"  状态: {result_duplicate.status}")
    print(f"  抓取: {result_duplicate.fetched_count} 条")
    print(f"  入库: {result_duplicate.saved_count} 条 (应为0，因为URL已存在)")
    print(f"  跳过: {result_duplicate.skipped_count} 条")
    
    # 关闭数据库连接
    db.close()
    
    print("\n" + "=" * 50)
    print("演示完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()