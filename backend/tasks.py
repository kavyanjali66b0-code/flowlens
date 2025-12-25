# tasks.py (Corrected)

import logging
from analyzer import CodebaseAnalyzer
from config import setup_logging, get_config

# Import the central celery instance
from celery_app import celery

# Setup logging for the Celery worker
config = get_config()
setup_logging(config['LOG_LEVEL'], config['LOG_FILE'])


@celery.task(name='analyzer.run')
def run_analysis(folder_path: str):
    """Celery task to run the codebase analysis."""
    logging.info(f"Starting analysis task for: {folder_path}")
    try:
        analyzer = CodebaseAnalyzer()
        result = analyzer.analyze(folder_path)
        logging.info(f"Analysis task completed successfully for: {folder_path}")
        return result
    except Exception as e:
        logging.error(f"Analysis task failed for {folder_path}: {e}", exc_info=True)
        raise