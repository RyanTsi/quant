# news_module/

Financial news scraping service with clean architecture (interface → scraper → repository → service).

## Files

| File                    | Role                                           |
|-------------------------|------------------------------------------------|
| `interfaces.py`         | Abstract base: `ScraperInterface`, `RepositoryInterface` |
| `schemas.py`            | Data classes: `NewsItem`, `NewsFilter`, `CrawlResult` |
| `service.py`            | `NewsCrawlerService` — business logic layer     |
| `repository.py`         | `NewsRepository` — SQLAlchemy persistence       |
| `models.py`             | ORM model definitions                           |
| `scrapers/base.py`      | Scraper base utilities                          |
| `scrapers/mock_scraper.py` | Mock scraper for testing                     |

## Architecture

```
ScraperInterface  ←  MockFinancialScraper (add real scrapers here)
       ↓
NewsCrawlerService  →  NewsRepository  →  SQLAlchemy DB
```

## Status

Work in progress. Only mock scraper implemented. To add a real scraper, implement `ScraperInterface` and register it in the service.

## See Also

- `config/settings.py` — scraper timeout settings
