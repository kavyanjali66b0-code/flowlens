# run.py (Corrected)

import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from celery.result import AsyncResult

from celery_app import celery
from tasks import run_analysis
from config import setup_logging, get_config

# Setup logging and configuration
config = get_config()
setup_logging(config['LOG_LEVEL'], config['LOG_FILE'])

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# --- FIX: Update Celery configuration keys to modern, lowercase format ---
app.config.update(
    broker_url=config['CELERY_BROKER_URL'],
    result_backend=config['CELERY_RESULT_BACKEND']
)
celery.conf.update(app.config)
# --- END FIX ---

class ContextTask(celery.Task):
    """Celery task with Flask application context."""
    def __call__(self, *args, **kwargs):
        with app.app_context():
            return self.run(*args, **kwargs)

celery.Task = ContextTask

# Configuration
USE_ASYNC = config['ANALYZER_ASYNC']


@app.route('/parse', methods=['POST'])
def parse_codebase():
    """API endpoint to parse a codebase."""
    try:
        data = request.json or {}
        folder_path = data.get('folder_path')
        if not folder_path: return jsonify({"error": "folder_path is required"}), 400
        if not os.path.exists(folder_path): return jsonify({"error": f"Path {folder_path} does not exist"}), 404
        logging.info(f"Received parse request for: {folder_path}")
        if USE_ASYNC:
            task = run_analysis.delay(folder_path)
            logging.info(f"Queued analysis task with ID: {task.id}")
            return jsonify({"job_id": task.id, "status": "queued"}), 202
        else:
            logging.info("Running analysis synchronously")
            result = run_analysis(folder_path)
            return jsonify(result)
    except Exception as e:
        logging.exception("Parse endpoint failed")
        return jsonify({"error": str(e)}), 500

@app.route('/status/<job_id>', methods=['GET'])
def status(job_id: str):
    """API endpoint to check job status."""
    if not USE_ASYNC: return jsonify({"error": "Async mode disabled"}), 400
    try:
        res: AsyncResult = AsyncResult(job_id, app=celery)
        state = res.state
        if state == 'PENDING':
            return jsonify({"job_id": job_id, "status": "pending"}), 200
        elif state in {'RECEIVED', 'STARTED'}:
            return jsonify({"job_id": job_id, "status": "running"}), 200
        elif state == 'FAILURE':
            error_info = str(res.info) if res.info else "Unknown error"
            logging.error(f"Task {job_id} failed: {error_info}")
            return jsonify({"job_id": job_id, "status": "failed", "error": error_info}), 500
        elif state == 'SUCCESS':
            logging.info(f"Task {job_id} completed successfully")
            return jsonify({"job_id": job_id, "status": "completed", "result": res.result}), 200
        else:
            return jsonify({"job_id": job_id, "status": state.lower()}), 200
    except Exception as e:
        logging.exception(f"Status endpoint failed for job {job_id}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "Codebase Analyzer", "async_mode": USE_ASYNC})

if __name__ == '__main__':
    app.run(debug=config['FLASK_DEBUG'], port=5000)