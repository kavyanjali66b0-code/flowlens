# tasks.py

import logging
import uuid
from typing import Dict, Any
from analyzer import CodebaseAnalyzer
from config import setup_logging, get_config
from celery_app import celery

config = get_config()
setup_logging(config['LOG_LEVEL'], config['LOG_FILE'])


@celery.task(name='analyzer.run', bind=True)
def run_analysis(self, folder_path: str, session_id: str = None):
    if not session_id:
        session_id = str(uuid.uuid4())
    
    logging.info(f"Starting analysis task for: {folder_path} (session: {session_id})")
    
    def progress_callback(status: Dict[str, Any]):
        try:
            self.update_state(
                state='PROGRESS',
                meta={
                    'session_id': status['session_id'],
                    'phase': status['phase'],
                    'overall_progress': status['overall_progress'],
                    'current_progress': status['current_progress'],
                    'current': status['current'],
                    'total': status['total'],
                    'message': status['message'],
                    'elapsed_seconds': status['elapsed_seconds'],
                    'errors': status['errors']
                }
            )
        except Exception as e:
            logging.warning(f"Failed to update task state: {e}")
    
    try:
        analyzer = CodebaseAnalyzer()
        result = analyzer.analyze(
            folder_path, 
            session_id=session_id,
            progress_callback=progress_callback
        )
        
        logging.info(f"Analysis task completed successfully for: {folder_path}")
        result['session_id'] = session_id
        return result
        
    except Exception as e:
        logging.error(f"Analysis task failed for {folder_path}: {e}", exc_info=True)
        self.update_state(
            state='FAILURE',
            meta={
                'session_id': session_id,
                'error': str(e),
                'exc_type': type(e).__name__
            }
        )
        raise
