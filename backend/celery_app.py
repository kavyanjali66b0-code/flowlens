# celery_app.py (Corrected)

import os
from celery import Celery

# Get config from environment variables
broker_url = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
backend_url = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Create the Celery instance that other modules will import
# The first argument is the name of the main module where tasks are defined.
celery = Celery(
    'tasks',
    broker=broker_url,
    backend=backend_url,
    include=['tasks']  # Explicitly tell Celery where to find the tasks
)

# Optional: Set a default configuration
celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)