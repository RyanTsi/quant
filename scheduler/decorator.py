import logging
import traceback
from datetime import datetime
from functools import wraps

logger = logging.getLogger("scheduler")


class TaskFailed(RuntimeError):
    def __init__(self, task_name: str, message: str):
        super().__init__(message)
        self.task_name = task_name


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
                tb = traceback.format_exc()
                logger.error(f"[{name}] failed:\n{tb}")
                raise TaskFailed(name, tb)
        wrapper.task_name = name
        return wrapper
    return decorator
