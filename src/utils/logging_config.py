"""
Logging Configuration for Media Recommender System

Structured logging with support for JSON output, request tracing,
and integration with monitoring systems.
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar
import traceback
from functools import wraps
import time

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


# Context variables for request tracing
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_var: ContextVar[Optional[int]] = ContextVar("user_id", default=None)


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add context variables
        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id
        
        user_id = user_id_var.get()
        if user_id:
            log_data["user_id"] = user_id
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


class PrettyFormatter(logging.Formatter):
    """Pretty formatter for development."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build message
        msg = f"{color}{timestamp} | {record.levelname:8} | {record.name} | {record.getMessage()}{self.RESET}"
        
        # Add context
        request_id = request_id_var.get()
        if request_id:
            msg = f"[{request_id[:8]}] {msg}"
        
        # Add exception
        if record.exc_info:
            msg += f"\n{traceback.format_exception(*record.exc_info)}"
        
        return msg


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter with extra context."""
    
    def process(self, msg: str, kwargs: Dict[str, Any]):
        extra = kwargs.get("extra", {})
        
        # Add context variables
        request_id = request_id_var.get()
        if request_id:
            extra["request_id"] = request_id
        
        user_id = user_id_var.get()
        if user_id:
            extra["user_id"] = user_id
        
        kwargs["extra"] = extra
        return msg, kwargs


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: Optional[str] = None
):
    """
    Configure logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Use JSON format for logs
        log_file: Optional file path for log output
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    if json_output:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(PrettyFormatter())
    
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())  # Always JSON for files
        root_logger.addHandler(file_handler)
    
    # Configure structlog if available
    if STRUCTLOG_AVAILABLE:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    logging.info(f"Logging configured: level={level}, json={json_output}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


def log_execution_time(logger: Optional[logging.Logger] = None):
    """Decorator to log function execution time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or logging.getLogger(func.__module__)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                _logger.info(
                    f"{func.__name__} completed",
                    extra={"duration_ms": elapsed * 1000}
                )
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                _logger.error(
                    f"{func.__name__} failed: {e}",
                    extra={"duration_ms": elapsed * 1000},
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


class RequestLogger:
    """Context manager for request logging."""
    
    def __init__(
        self,
        request_id: str,
        user_id: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.request_id = request_id
        self.user_id = user_id
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self._request_id_token = None
        self._user_id_token = None
    
    def __enter__(self):
        self._request_id_token = request_id_var.set(self.request_id)
        if self.user_id:
            self._user_id_token = user_id_var.set(self.user_id)
        
        self.start_time = time.time()
        self.logger.info("Request started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        
        if exc_type:
            self.logger.error(
                f"Request failed: {exc_val}",
                extra={"duration_ms": elapsed * 1000},
                exc_info=(exc_type, exc_val, exc_tb)
            )
        else:
            self.logger.info(
                "Request completed",
                extra={"duration_ms": elapsed * 1000}
            )
        
        # Reset context
        request_id_var.reset(self._request_id_token)
        if self._user_id_token:
            user_id_var.reset(self._user_id_token)
        
        return False  # Don't suppress exceptions


class MetricLogger:
    """Logger for recommendation metrics."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("metrics")
    
    def log_recommendation(
        self,
        user_id: int,
        num_recommendations: int,
        model: str,
        latency_ms: float,
        cache_hit: bool = False
    ):
        """Log recommendation request metrics."""
        self.logger.info(
            "recommendation_served",
            extra={
                "metric_type": "recommendation",
                "user_id": user_id,
                "num_recommendations": num_recommendations,
                "model": model,
                "latency_ms": latency_ms,
                "cache_hit": cache_hit
            }
        )
    
    def log_interaction(
        self,
        user_id: int,
        item_id: int,
        interaction_type: str,
        rating: Optional[float] = None
    ):
        """Log user interaction."""
        self.logger.info(
            "interaction_logged",
            extra={
                "metric_type": "interaction",
                "user_id": user_id,
                "item_id": item_id,
                "interaction_type": interaction_type,
                "rating": rating
            }
        )
    
    def log_model_performance(
        self,
        model: str,
        metrics: Dict[str, float]
    ):
        """Log model performance metrics."""
        self.logger.info(
            "model_metrics",
            extra={
                "metric_type": "model_performance",
                "model": model,
                **metrics
            }
        )


if __name__ == "__main__":
    # Example usage
    setup_logging(level="DEBUG", json_output=False)
    
    logger = get_logger(__name__)
    
    # Regular logging
    logger.info("Application started")
    logger.debug("Debug message")
    logger.warning("Warning message")
    
    # With context
    with RequestLogger(request_id="abc123", user_id=42):
        logger.info("Processing request")
    
    # Metrics
    metric_logger = MetricLogger()
    metric_logger.log_recommendation(
        user_id=123,
        num_recommendations=10,
        model="hybrid",
        latency_ms=45.2,
        cache_hit=False
    )
