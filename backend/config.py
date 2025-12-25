"""
Configuration and logging setup for the codebase analyzer.
"""

import os
import logging
import logging.handlers
from pathlib import Path


def setup_logging(log_level: str = None, log_file: str = None):
    """Setup logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
    """
    # Get log level from environment or use INFO as default
    level = log_level or os.environ.get('LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, level, logging.INFO)
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger('celery').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('redis').setLevel(logging.WARNING)
    
    logging.info(f"Logging configured with level: {level}")


def get_config():
    """Get application configuration from environment variables."""
    return {
        'CELERY_BROKER_URL': os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
        'CELERY_RESULT_BACKEND': os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
        'ANALYZER_ASYNC': os.environ.get('ANALYZER_ASYNC', '0') == '1',
        'LOG_LEVEL': os.environ.get('LOG_LEVEL', 'INFO'),
        'LOG_FILE': os.environ.get('LOG_FILE', 'logs/analyzer.log'),
        'FLASK_ENV': os.environ.get('FLASK_ENV', 'development'),
        'FLASK_DEBUG': os.environ.get('FLASK_DEBUG', 'true').lower() == 'true',
    }
