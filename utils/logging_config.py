import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class BacktestLogger:
    """Centralized logging configuration for backtest framework"""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
        """Get or create a logger with consistent formatting"""
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        if not logger.handlers:
            formatter = logging.Formatter(
                fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            if log_file:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def log_performance(cls, logger: logging.Logger, operation: str, start_time: datetime, end_time: datetime):
        """Log performance timing information"""
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Performance | {operation} | Duration: {duration:.2f}s")
    
    @classmethod
    def log_error_with_context(cls, logger: logging.Logger, error: Exception, context: dict):
        """Log error with contextual information"""
        context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
        logger.error(f"Error: {str(error)} | Context: {context_str}")