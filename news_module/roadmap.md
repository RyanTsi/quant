# Roadmap: news_module/

## 1. Overview

**Status: Work In Progress** — only mock scraper implemented. Known broken imports (`news_module.config` missing, `models.py` references non-existent `summary` field). Keep this module isolated until fixed.

## 2. Architecture

```
ScraperInterface  ←  MockFinancialScraper (add real scrapers here)
       ↓
NewsCrawlerService  →  NewsRepository  →  SQLAlchemy DB
```

## 3. File-Role Mapping

| File / Subdirectory | Role / Description |
| :--- | :--- |
| `interfaces.py` | Abstract base: `ScraperInterface`, `RepositoryInterface` |
| `schemas.py` | Data classes: `NewsItem`, `NewsFilter`, `CrawlResult` |
| `service.py` | `NewsCrawlerService` — business logic layer |
| `repository.py` | `NewsRepository` — SQLAlchemy persistence |
| `models.py` | ORM model definitions |
| `scrapers/base.py` | Scraper base utilities |
| `scrapers/mock_scraper.py` | Mock scraper for testing |

## 5. Navigation

| If you want to... | Go to... |
| :--- | :--- |
| Add a real scraper implementation | `scrapers/` |
| Adjust crawl orchestration/business logic | `service.py` |
| Adjust persistence layer | `repository.py` / `models.py` |
| Fix schemas / dataclasses | `schemas.py` |

## 6. Conventions

- Treat as isolated until imports and schema are fixed; do not wire into `scheduler` without review.
