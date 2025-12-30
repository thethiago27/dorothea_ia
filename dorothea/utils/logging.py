"""
Structured logging utilities for Dorothea AI.
"""
import json
import logging
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional, Generator

@dataclass
class LogContext:
    """Context information for structured logging."""
    component: str
    operation: str
    metadata: Dict[str, Any] = None
class StructuredLogger:
    """Structured logger that outputs JSON-formatted logs with stack traces and context information."""
    
    def __init__(self, name: str, level: str = "INFO"):
        """Initialize the structured logger.
        
        Args:
            name: Logger name (typically module name)
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(self._create_json_formatter())
        self.logger.addHandler(handler)
        self.logger.propagate = False
        self._context: Optional[LogContext] = None
    def _create_json_formatter(self) -> logging.Formatter:
        """Create a JSON formatter for structured logging."""
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage()
                }
                if hasattr(record, 'context') and record.context:
                    log_entry["context"] = asdict(record.context)
                if hasattr(record, 'extra_fields') and record.extra_fields:
                    log_entry.update(record.extra_fields)
                if record.exc_info:
                    log_entry["exception"] = {
                        "type": record.exc_info[0].__name__,
                        "message": str(record.exc_info[1]),
                        "traceback": traceback.format_exception(*record.exc_info)
                    }
                return json.dumps(log_entry, default=str)
        return JSONFormatter()
    def _log(self, level: str, message: str, exc_info: bool = False, **kwargs) -> None:
        """Internal logging method."""
        extra = {
            'context': self._context,
            'extra_fields': kwargs
        }
        getattr(self.logger, level.lower())(
            message,
            extra=extra,
            exc_info=exc_info
        )
    def info(self, message: str, **kwargs) -> None:
        """Log an info message.
        
        Args:
            message: Log message
            **kwargs: Additional fields to include in log
        """
        self._log("INFO", message, **kwargs)
    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log an error message.
        
        Args:
            message: Log message
            exc_info: Include exception information
            **kwargs: Additional fields to include in log
        """
        self._log("ERROR", message, exc_info=exc_info, **kwargs)
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message.
        
        Args:
            message: Log message
            **kwargs: Additional fields to include in log
        """
        self._log("WARNING", message, **kwargs)
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message.
        
        Args:
            message: Log message
            **kwargs: Additional fields to include in log
        """
        self._log("DEBUG", message, **kwargs)
    def metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
        """
        metric_data = {
            "metric_name": name,
            "metric_value": value
        }
        if tags:
            metric_data["tags"] = tags
        self._log("INFO", f"Metric: {name}", **metric_data)
    def with_context(self, context: LogContext) -> 'StructuredLogger':
        """Create a new logger instance with additional context.
        
        Args:
            context: LogContext with component and operation info
            
        Returns:
            New StructuredLogger instance with context
        """
        level_name = logging.getLevelName(self.logger.level)
        new_logger = StructuredLogger(self.name, level_name)
        new_logger._context = context
        return new_logger
    @contextmanager
    def operation_context(self, component: str, operation: str, **metadata) -> Generator['StructuredLogger', None, None]:
        """Context manager for operation logging with automatic start/end logging.
        
        Args:
            component: Component name performing the operation
            operation: Operation name
            **metadata: Additional metadata for the operation
            
        Yields:
            StructuredLogger instance with operation context
        """
        context = LogContext(
            component=component,
            operation=operation,
            metadata=metadata
        )
        contextual_logger = self.with_context(context)
        contextual_logger.info(
            f"Starting operation: {operation}",
            operation_status="started"
        )
        start_time = datetime.utcnow()
        try:
            yield contextual_logger
            duration = (datetime.utcnow() - start_time).total_seconds()
            contextual_logger.info(
                f"Completed operation: {operation}",
                operation_status="completed",
                duration_seconds=duration
            )
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            contextual_logger.error(
                f"Failed operation: {operation}",
                exc_info=True,
                operation_status="failed",
                duration_seconds=duration,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise
    def log_config(self, config: Dict[str, Any], exclude_secrets: bool = True) -> None:
        """Log configuration with optional secret filtering.
        
        Args:
            config: Configuration dictionary to log
            exclude_secrets: Whether to filter out sensitive information
        """
        if exclude_secrets:
            secret_keys = {
                'password', 'secret', 'key', 'token', 'api_key', 'auth'
            }
            safe_config = {}
            for key, value in config.items():
                key_lower = key.lower()
                if any(secret_key in key_lower for secret_key in secret_keys):
                    safe_config[key] = "***REDACTED***"
                elif isinstance(value, dict):
                    safe_config[key] = self._filter_secrets(value, secret_keys)
                else:
                    safe_config[key] = value
            config = safe_config
        self.info("Configuration loaded", config=config)
    def _filter_secrets(self, data: Dict[str, Any], secret_keys: set) -> Dict[str, Any]:
        filtered = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(secret_key in key_lower for secret_key in secret_keys):
                filtered[key] = "***REDACTED***"
            elif isinstance(value, dict):
                filtered[key] = self._filter_secrets(value, secret_keys)
            else:
                filtered[key] = value
        return filtered
def get_logger(name: str, level: str = "INFO") -> StructuredLogger:
    """Factory function to create a StructuredLogger instance.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, level)