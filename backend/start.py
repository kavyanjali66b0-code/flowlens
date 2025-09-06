#!/usr/bin/env python3
"""
Startup script for the codebase analyzer.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# This ensures the script itself can find the 'config' module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import setup_logging, get_config

def check_redis_connection():
    """Checks if the Redis server is running."""
    try:
        import redis
        config = get_config()
        redis_url = config.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
        r = redis.from_url(redis_url)
        r.ping()
        print("✓ Redis is running and accessible.")
        return True
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        print("  Please ensure the Redis server is running and accessible at the configured URL.")
        return False

def start_process(command):
    """Starts a subprocess with the correct environment to find modules."""
    env = os.environ.copy()
    current_python_path = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f"{os.getcwd()}{os.pathsep}{current_python_path}"
    return subprocess.Popen(command, env=env)

def main():
    """Main startup function."""
    parser = argparse.ArgumentParser(description="Start the codebase analyzer")
    parser.add_argument("--worker-only", action="store_true", help="Start only Celery worker")
    parser.add_argument("--app-only", action="store_true", help="Start only Flask app")
    parser.add_argument("--check-redis", action="store_true", help="Check Redis connection")

    args = parser.parse_args()

    config = get_config()
    setup_logging(config['LOG_LEVEL'], config['LOG_FILE'])

    if args.check_redis:
        check_redis_connection()
        return

    if not check_redis_connection():
        return

    processes = []

    try:
        if not args.app_only:
            print("Starting Celery worker...")
            # --- THE DEFINITIVE FIX FOR WINDOWS ---
            # Add "-P solo" to force a single-threaded worker pool that is Windows-compatible.
            celery_cmd = [sys.executable, "-m", "celery", "-A", "tasks", "worker", "--loglevel=INFO", "-P", "solo"]
            # --- END FIX ---
            worker_process = start_process(celery_cmd)
            processes.append(worker_process)

        if not args.worker_only:
            print("Starting Flask application...")
            flask_cmd = [sys.executable, "run.py"]
            app_process = start_process(flask_cmd)
            processes.append(app_process)

        print("\n✓ Codebase analyzer started successfully!")
        print(f"  Flask app: http://localhost:5000")
        print(f"  Health check: http://localhost:5000/health")
        print("\nPress Ctrl+C to stop all services")

        for process in processes:
            process.wait()

    except KeyboardInterrupt:
        print("\nShutting down...")
        for process in processes:
            process.terminate()
            process.wait()

if __name__ == "__main__":
    main()