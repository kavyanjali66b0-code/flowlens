# run.py (Corrected)

import os
import sys
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

# Import custom exceptions for error handling
from analyzer.exceptions import (
    AnalyzerError, 
    ParsingError, 
    MemoryLimitExceeded, 
    ConfigurationError,
    ProjectTooLargeError,
    TimeoutError as AnalyzerTimeoutError,
    InvalidProjectError,
    UnsupportedLanguageError
)

# Setup logging and configuration
config = get_config()
setup_logging(config['LOG_LEVEL'], config['LOG_FILE'])

# FIX: Prevent Flask initialization in multiprocessing child processes on Windows
# This guards against the '__mp_main__' error when using parallel parsing
if __name__ != '__mp_main__':
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
else:
    # Running in multiprocessing child - skip Flask initialization
    app = None

class ContextTask(celery.Task):
    """Celery task with Flask application context."""
    def __call__(self, *args, **kwargs):
        if app:
            with app.app_context():
                return self.run(*args, **kwargs)
        return self.run(*args, **kwargs)

celery.Task = ContextTask

# Configuration
USE_ASYNC = config['ANALYZER_ASYNC']

# Only define routes if app is initialized (not in child process)
if app:
    @app.route('/parse', methods=['POST'])
    def parse_codebase():
        data = request.json or {}
        repo_url = data.get('repoUrl')
        if not repo_url:
            return jsonify({"error": "repoUrl is required"}), 400

        # Clone repo to temp folder
        temp_dir = tempfile.mkdtemp()
        try:
            git.Repo.clone_from(repo_url, temp_dir)
        except Exception as e:
            return jsonify({"error": f"Failed to clone repo: {str(e)}"}), 500

        # Run the analyzer on cloned folder
        analyzer = CodebaseAnalyzer()
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


    # =============================================================================
    # ERROR HANDLERS
    # =============================================================================

    @app.errorhandler(MemoryLimitExceeded)
    def handle_memory_limit_exceeded(error):
        """Handle memory limit exceeded errors."""
        logging.error(f"Memory limit exceeded: {error}")
        return jsonify(error.to_dict()), 507


    @app.errorhandler(ParsingError)
    def handle_parsing_error(error):
        """Handle parsing errors."""
        logging.error(f"Parsing error: {error}")
        return jsonify(error.to_dict()), 422


    @app.errorhandler(ConfigurationError)
    def handle_configuration_error(error):
        """Handle configuration errors."""
        logging.error(f"Configuration error: {error}")
        return jsonify(error.to_dict()), 400


    @app.errorhandler(ProjectTooLargeError)
    def handle_project_too_large(error):
        """Handle project too large errors."""
        logging.error(f"Project too large: {error}")
        return jsonify(error.to_dict()), 413


    @app.errorhandler(AnalyzerTimeoutError)
    def handle_analyzer_timeout(error):
        """Handle timeout errors."""
        logging.error(f"Analysis timeout: {error}")
        return jsonify(error.to_dict()), 504


    @app.errorhandler(InvalidProjectError)
    def handle_invalid_project(error):
        """Handle invalid project errors."""
        logging.error(f"Invalid project: {error}")
        return jsonify(error.to_dict()), 400


    @app.errorhandler(UnsupportedLanguageError)
    def handle_unsupported_language(error):
        """Handle unsupported language errors."""
        logging.error(f"Unsupported language: {error}")
        return jsonify(error.to_dict()), 400


    @app.errorhandler(AnalyzerError)
    def handle_analyzer_error(error):
        """Handle generic analyzer errors (base class)."""
        logging.error(f"Analyzer error: {error}")
        return jsonify(error.to_dict()), 500


    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        """Handle unexpected errors that are not AnalyzerError subclasses."""
        logging.exception(f"Unexpected error: {error}")
        return jsonify({
            "error": "InternalServerError",
            "message": "An unexpected error occurred during analysis",
            "details": str(error)
        }), 500

if __name__ == '__main__':
    if app:
        app.run(debug=config['FLASK_DEBUG'], port=5000)