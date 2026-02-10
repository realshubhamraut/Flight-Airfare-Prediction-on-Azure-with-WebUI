"""
ETL Logging Utilities
=====================
Structured logging for ETL pipeline with support for multiple outputs
(console, file, structured JSON).
"""

import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from functools import wraps
import time


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data
            
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)


class ETLLogger:
    """
    Structured logger for ETL pipelines.
    Supports console, file, and JSON logging.
    """
    
    def __init__(
        self,
        name: str = "ETL",
        log_level: int = logging.INFO,
        log_dir: Optional[Path] = None,
        enable_json: bool = False
    ):
        """
        Initialize ETL logger.
        
        Args:
            name: Logger name
            log_level: Logging level
            log_dir: Directory for log files
            enable_json: Enable JSON structured logging
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"etl_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            
            if enable_json:
                file_handler.setFormatter(JSONFormatter())
            else:
                file_handler.setFormatter(console_format)
            
            self.logger.addHandler(file_handler)
            
        self._step_count = 0
        self._pipeline_start: Optional[datetime] = None
        
    def start_pipeline(self, pipeline_name: str, config: Optional[Dict] = None):
        """Log pipeline start."""
        self._pipeline_start = datetime.now()
        self._step_count = 0
        self.logger.info("=" * 70)
        self.logger.info(f"PIPELINE START: {pipeline_name}")
        self.logger.info(f"Started at: {self._pipeline_start.isoformat()}")
        if config:
            self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        self.logger.info("=" * 70)
        
    def end_pipeline(self, pipeline_name: str, success: bool = True):
        """Log pipeline end."""
        end_time = datetime.now()
        duration = (end_time - self._pipeline_start).total_seconds() if self._pipeline_start else 0
        
        self.logger.info("=" * 70)
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"PIPELINE END: {pipeline_name} - {status}")
        self.logger.info(f"Total steps completed: {self._step_count}")
        self.logger.info(f"Total duration: {duration:.2f} seconds")
        self.logger.info("=" * 70)
        
    def step(self, step_name: str):
        """Log pipeline step."""
        self._step_count += 1
        self.logger.info(f"[Step {self._step_count}] {step_name}")
        
    def step_complete(self, step_name: str, metrics: Optional[Dict] = None):
        """Log step completion with optional metrics."""
        msg = f"[Step {self._step_count}] ✓ {step_name} complete"
        if metrics:
            msg += f" | Metrics: {json.dumps(metrics)}"
        self.logger.info(msg)
        
    def data_quality(self, check_name: str, passed: bool, details: Optional[Dict] = None):
        """Log data quality check result."""
        status = "PASS" if passed else "FAIL"
        msg = f"[Quality Check] {check_name}: {status}"
        if details:
            msg += f" | {json.dumps(details)}"
        
        if passed:
            self.logger.info(msg)
        else:
            self.logger.warning(msg)
            
    def metrics(self, stage: str, metrics: Dict[str, Any]):
        """Log metrics for a stage."""
        self.logger.info(f"[Metrics] {stage}: {json.dumps(metrics)}")
        
    def info(self, message: str, **kwargs):
        """Log info message."""
        if kwargs:
            message += f" | {json.dumps(kwargs)}"
        self.logger.info(message)
        
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        if kwargs:
            message += f" | {json.dumps(kwargs)}"
        self.logger.warning(message)
        
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message."""
        if kwargs:
            message += f" | {json.dumps(kwargs)}"
        self.logger.error(message, exc_info=exc_info)
        
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        if kwargs:
            message += f" | {json.dumps(kwargs)}"
        self.logger.debug(message)


def log_execution_time(logger: Optional[ETLLogger] = None):
    """
    Decorator to log function execution time.
    
    Args:
        logger: ETL logger instance
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            
            if logger:
                logger.info(f"{func.__name__} completed in {duration:.2f}s")
            
            return result
        return wrapper
    return decorator


def log_dataframe_info(df, name: str, logger: ETLLogger):
    """
    Log DataFrame information.
    
    Args:
        df: PySpark DataFrame
        name: DataFrame name for logging
        logger: ETL logger instance
    """
    count = df.count()
    columns = len(df.columns)
    logger.info(
        f"DataFrame '{name}': {count:,} rows, {columns} columns",
        columns=df.columns
    )
