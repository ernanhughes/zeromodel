# zeromodel/timing.py
"""
Lightweight timing decorator for ZeroModel.

This decorator adds timing functionality with ZERO overhead when debug logging
is disabled. It only measures function execution time and logs it when debug
level logging is enabled.

Key features:
- Zero performance impact in production (when debug logging is off)
- Automatically detects class methods to include class name in logs
- Smart time formatting (ms for short operations, seconds for longer ones)
- Simple integration with existing logging system
"""

import logging
import time
from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar('T')
logger = logging.getLogger(__name__)
logger.propagate = True

def timeit(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to time function execution with minimal overhead.
    
    Only incurs timing cost when debug logging is enabled.
    Automatically detects if the function is a method to include class name.
    
    Example:
        @timeit
        def process_data(data):
            # processing logic
            
        @timeit
        def _internal_helper(x):
            # helper logic
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        # Only time if debug logging is enabled - this check has negligible overhead
        if not logger.isEnabledFor(logging.DEBUG):
            return func(*args, **kwargs)
        
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            
            # Get class name if this is a method
            class_name = ""
            if args and hasattr(args[0], '__class__'):
                class_name = f"{args[0].__class__.__name__}."
                
            # Format time appropriately (ms for short times, seconds for longer)
            if elapsed < 0.001:
                time_str = f"{elapsed * 1_000_000:.1f} μs"
            elif elapsed < 0.1:
                time_str = f"{elapsed * 1000:.3f} ms"
            else:
                time_str = f"{elapsed:.6f} seconds"
                
            logger.debug(f"⏱️ {class_name}{func.__name__} completed in {time_str}")
    
    return wrapper