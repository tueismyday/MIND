"""
Performance monitoring utilities for the Agentic RAG Medical Documentation System.
Provides decorators and utilities for profiling function execution.
"""

import time
import tracemalloc
from functools import wraps

def profile(func):
    """
    Decorator function that measures and reports execution time of the wrapped function.
    
    Args:
        func: The function to be profiled.
        
    Returns:
        wrapper: The wrapped function with profiling capabilities.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        tracemalloc.stop()
        print(f"\n\n[PROFILE] {func.__name__} took {end_time - start_time:.2f}s\n\n")
        return result
    return wrapper