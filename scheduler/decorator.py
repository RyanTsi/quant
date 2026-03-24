import logging
import traceback
from datetime import datetime
from functools import wraps

logger = logging.getLogger("scheduler")


def task(name: str):
    """Decorator that wraps a task with logging, timing, and error handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = datetime.now()
            logger.info(f"[{name}] started at {start:%H:%M:%S}")
            try:
                result = func(*args, **kwargs)
                elapsed = (datetime.now() - start).total_seconds()
                logger.info(f"[{name}] finished in {elapsed:.1f}s")
                return result
            except Exception:
                logger.error(f"[{name}] failed:\n{traceback.format_exc()}")
                return None
        wrapper.task_name = name
        return wrapper
    return decorator
