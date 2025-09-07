import time
import logging
from typing import Any, Dict
from functools import wraps

def timing_decorator(func):
    """Decorator to measure function execution time"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds

        # Add timing info to result if it's a dictionary
        if isinstance(result, dict):
            result["execution_time_ms"] = execution_time

        logging.info(f"{func.__name__} executed in {execution_time:.2f}ms")
        return result

    return wrapper


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("ai_evolution.log")
        ]
    )


class PerformanceTracker:
    """Track performance metrics across the system"""

    def __init__(self):
        self.metrics = {
            "requests_processed": 0,
            "total_execution_time": 0,
            "average_response_time": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

    def record_request(self, execution_time: float):
        """Record a processed request"""
        self.metrics["requests_processed"] += 1
        self.metrics["total_execution_time"] += execution_time
        self.metrics["average_response_time"] = (
                self.metrics["total_execution_time"] / self.metrics["requests_processed"]
        )

    def record_cache_hit(self):
        """Record a cache hit"""
        self.metrics["cache_hits"] += 1

    def record_cache_miss(self):
        """Record a cache miss"""
        self.metrics["cache_misses"] += 1

    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_cache_requests = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        if total_cache_requests == 0:
            return 0.0
        return self.metrics["cache_hits"] / total_cache_requests

    def get_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""
        metrics = self.metrics.copy()
        metrics["cache_hit_rate"] = self.get_cache_hit_rate()
        return metrics