 # run.py (Corrected)

import os
import tempfile
import git  # pip install GitPython
import logging
import numpy as np
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

# Custom JSON encoder to handle NumPy types from CodeBERT
from flask.json.provider import DefaultJSONProvider

class NumpyJSONProvider(DefaultJSONProvider):
    """JSON provider that handles NumPy types for Flask 3.x."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Attach the custom JSON provider
app.json = NumpyJSONProvider(app)

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
    enable_enhanced = data.get('enableEnhanced', False)
    
    # Validation
    if not repo_url:
        return jsonify({
            "error": "repoUrl is required",
            "message": "Please provide a repository URL to analyze"
        }), 400
    
    # Basic URL validation
    if not repo_url.startswith(('http://', 'https://', 'git@', 'git://')):
        return jsonify({
            "error": "Invalid repository URL",
            "message": "URL must start with http://, https://, git@, or git://",
            "example": "https://github.com/user/repo.git"
        }), 400

    # Clone repo to temp folder
    temp_dir = tempfile.mkdtemp()
    try:
        logging.info(f"Cloning {repo_url} to {temp_dir}")
        # Use shallow clone for speed in development
        git.Repo.clone_from(repo_url, temp_dir, depth=1)
        logging.info(f"Successfully cloned repository")
    except git.exc.GitCommandError as e:
        logging.error(f"Git clone failed: {e}")
        return jsonify({
            "error": "Failed to clone repository",
            "message": "Git clone command failed. Check that the URL is correct and the repository is accessible.",
            "details": str(e),
            "repo_url": repo_url
        }), 500
    except Exception as e:
        logging.error(f"Unexpected clone error: {e}")
        return jsonify({
            "error": "Failed to clone repository", 
            "message": f"Unexpected error: {str(e)}",
            "type": type(e).__name__
        }), 500

    # Run the analyzer on cloned folder
    analyzer = CodebaseAnalyzer(enable_enhanced_analysis=enable_enhanced)
    try:
        logging.info(f"Starting analysis of {temp_dir}")
        result = analyzer.analyze(temp_dir)
        logging.info(f"Analysis complete. Found {result.get('statistics', {}).get('total_nodes', 0)} nodes")
        return jsonify(result)
    except Exception as e:
        logging.error(f"Analysis failed: {e}", exc_info=True)
        return jsonify({
            "error": "Analysis failed",
            "message": str(e),
            "type": type(e).__name__,
            "details": "An error occurred during code analysis. Check the logs for more information."
        }), 500
    finally:
        # Cleanup temp directory
        try:
            import shutil
            shutil.rmtree(temp_dir)
            logging.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logging.warning(f"Failed to cleanup {temp_dir}: {e}")


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
    # Disable reloader to avoid WERKZEUG_SERVER_FD issues
    app.run(debug=config['FLASK_DEBUG'], port=5000, use_reloader=False)