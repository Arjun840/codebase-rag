"""Logging utilities for the RAG system."""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger as loguru_logger
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    use_loguru: bool = True,
    format_string: Optional[str] = None
) -> None:
    """Set up logging configuration."""
    
    if use_loguru:
        setup_loguru_logging(level, log_file, format_string)
    else:
        setup_standard_logging(level, log_file, format_string)


def setup_loguru_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> None:
    """Set up loguru logging."""
    
    # Remove default handler
    loguru_logger.remove()
    
    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    
    # Console handler
    loguru_logger.add(
        sys.stdout,
        format=format_string,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        loguru_logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation="10 MB",
            retention="1 week",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
    
    # Intercept standard logging
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = loguru_logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            loguru_logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    # Replace handlers for specific loggers
    for logger_name in ["uvicorn", "fastapi", "transformers", "sentence_transformers"]:
        logger = logging.getLogger(logger_name)
        logger.handlers = [InterceptHandler()]
        logger.propagate = False


def setup_standard_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> None:
    """Set up standard Python logging."""
    
    # Default format
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


def log_function_call(func_name: str, args: Dict[str, Any], level: str = "DEBUG") -> None:
    """Log a function call with arguments."""
    logger = get_logger(__name__)
    log_level = getattr(logging, level.upper())
    
    args_str = ", ".join([f"{k}={v}" for k, v in args.items()])
    logger.log(log_level, f"Calling {func_name}({args_str})")


def log_performance(func_name: str, duration: float, level: str = "INFO") -> None:
    """Log performance metrics."""
    logger = get_logger(__name__)
    log_level = getattr(logging, level.upper())
    
    logger.log(log_level, f"Performance: {func_name} took {duration:.4f}s")


def log_error(error: Exception, context: Optional[str] = None, level: str = "ERROR") -> None:
    """Log an error with context."""
    logger = get_logger(__name__)
    log_level = getattr(logging, level.upper())
    
    error_msg = f"Error: {str(error)}"
    if context:
        error_msg = f"{context} - {error_msg}"
    
    logger.log(log_level, error_msg, exc_info=True)


def log_memory_usage(context: str = "", level: str = "INFO") -> None:
    """Log current memory usage."""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        logger = get_logger(__name__)
        log_level = getattr(logging, level.upper())
        
        memory_mb = memory_info.rss / 1024 / 1024
        logger.log(log_level, f"Memory usage{' (' + context + ')' if context else ''}: {memory_mb:.2f} MB")
        
    except ImportError:
        logger = get_logger(__name__)
        logger.warning("psutil not available, cannot log memory usage")


def log_gpu_usage(level: str = "INFO") -> None:
    """Log GPU usage if available."""
    try:
        import torch
        
        if torch.cuda.is_available():
            logger = get_logger(__name__)
            log_level = getattr(logging, level.upper())
            
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                memory_cached = torch.cuda.memory_reserved(i) / 1024 / 1024
                
                logger.log(
                    log_level,
                    f"GPU {i}: {memory_allocated:.2f} MB allocated, {memory_cached:.2f} MB cached"
                )
    except ImportError:
        pass


class PerformanceLogger:
    """Context manager for logging performance."""
    
    def __init__(self, operation_name: str, level: str = "INFO"):
        self.operation_name = operation_name
        self.level = level
        self.start_time = None
        self.logger = get_logger(__name__)
    
    def __enter__(self):
        self.start_time = datetime.now()
        log_level = getattr(logging, self.level.upper())
        self.logger.log(log_level, f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        log_level = getattr(logging, self.level.upper())
        
        if exc_type:
            self.logger.log(log_level, f"Failed {self.operation_name} after {duration:.4f}s")
        else:
            self.logger.log(log_level, f"Completed {self.operation_name} in {duration:.4f}s")


def configure_library_logging():
    """Configure logging for third-party libraries."""
    # Reduce verbosity of common libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)


def setup_dev_logging():
    """Set up logging for development."""
    setup_logging(
        level="DEBUG",
        log_file=Path("logs/codebase_rag.log"),
        use_loguru=True
    )
    configure_library_logging()


def setup_prod_logging():
    """Set up logging for production."""
    setup_logging(
        level="INFO",
        log_file=Path("logs/codebase_rag.log"),
        use_loguru=True
    )
    configure_library_logging() 