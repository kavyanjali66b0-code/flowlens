# run.py (Corrected)

import os
import tempfile
import git  # pip install GitPython
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from celery.result import AsyncResult
from analyzer import CodebaseAnalyzer
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
    data = request.json or {}
    repo_url = data.get('repoUrl')
    folder_path = data.get('folder_path')
    
    # Support both repoUrl (git clone) and folder_path (local folder)
    if folder_path and os.path.isdir(folder_path):
        # Use local folder directly
        temp_dir = folder_path
    elif repo_url:
        # Clone repo to temp folder
        temp_dir = tempfile.mkdtemp()
        try:
            git.Repo.clone_from(repo_url, temp_dir)
        except Exception as e:
            return jsonify({"error": f"Failed to clone repo: {str(e)}"}), 500
    else:
        return jsonify({"error": "repoUrl or folder_path is required"}), 400

    # Run the analyzer on the folder
    analyzer = CodebaseAnalyzer(enable_enhanced_analysis=True)
    try:
        result = analyzer.analyze(temp_dir)
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    return jsonify(result)


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