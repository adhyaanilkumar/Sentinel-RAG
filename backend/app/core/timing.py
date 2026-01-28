"""
Timing utilities for measuring operation latency.

Provides decorators and utilities for performance measurement.
"""

import time
import logging
from functools import wraps
from typing import Callable, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and stores timing metrics."""
    
    def __init__(self):
        self._metrics: dict[str, list[float]] = defaultdict(list)
    
    def record(self, operation: str, elapsed: float):
        """Record a timing measurement."""
        self._metrics[operation].append(elapsed)
        
    def get_stats(self, operation: str) -> dict[str, float]:
        """Get statistics for an operation."""
        times = self._metrics.get(operation, [])
        if not times:
            return {"count": 0, "avg": 0, "min": 0, "max": 0}
        return {
            "count": len(times),
            "avg": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
        }
    
    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for all operations."""
        return {op: self.get_stats(op) for op in self._metrics}
    
    def clear(self):
        """Clear all collected metrics."""
        self._metrics.clear()


# Global metrics collector
metrics = MetricsCollector()


def timed(operation_name: str):
    """
    Decorator to measure and log operation timing.
    
    Usage:
        @timed("image_analysis")
        async def analyze_image(image: bytes) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                logger.info(f"{operation_name}: {elapsed:.3f}s")
                metrics.record(operation_name, elapsed)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                logger.info(f"{operation_name}: {elapsed:.3f}s")
                metrics.record(operation_name, elapsed)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, operation_name: str, log: bool = True):
        self.operation_name = operation_name
        self.log = log
        self.start: float = 0
        self.elapsed: float = 0
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        if self.log:
            logger.info(f"{self.operation_name}: {self.elapsed:.3f}s")
        metrics.record(self.operation_name, self.elapsed)
    
    @property
    def elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds."""
        return int(self.elapsed * 1000)
