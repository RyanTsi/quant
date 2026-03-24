# news_module/

**Status: Work In Progress** — Only mock scraper implemented. Known broken imports (`news_module.config` missing, `models.py` references non-existent `summary` field). Do not depend on this module from the main pipeline.

## Architecture

```
ScraperInterface  ←  MockFinancialScraper (add real scrapers here)
       ↓
NewsCrawlerService  →  NewsRepository  →  SQLAlchemy DB
```

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

## See Also

- `config/settings.py` — scraper timeout settings
