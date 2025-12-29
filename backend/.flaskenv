# Flask environment configuration
# This file is automatically loaded by Flask when using python-dotenv

# Use stat reloader instead of watchdog to avoid .venv monitoring issues
FLASK_RUN_RELOAD_TYPE=stat

# Development mode
FLASK_ENV=development
FLASK_DEBUG=1

# Server configuration
FLASK_RUN_HOST=0.0.0.0
FLASK_RUN_PORT=5000

# Exclude patterns for auto-reloader (prevents watching .venv, __pycache__, etc.)
FLASK_RUN_EXTRA_FILES=
