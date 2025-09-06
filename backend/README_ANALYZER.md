# Codebase Analyzer - Refactored Architecture

## Overview

This is a production-ready codebase analyzer that has been refactored from a monolithic structure into a clean, modular architecture. The analyzer can parse various programming languages and generate workflow graphs representing project structure.

## Architecture

### New Structure

```
/dev-scope
├── analyzer/                 # Core analysis package
│   ├── __init__.py          # Package initialization
│   ├── models.py            # Data models (Node, Edge, Enums)
│   ├── scanner.py           # Project type detection
│   ├── entry_points.py      # Entry point identification
│   ├── parser.py            # Language parsing (Tree-sitter)
│   └── main.py              # Main orchestrator
├── celery_app.py            # Celery configuration
├── tasks.py                 # Celery task definitions
├── run.py                   # Flask application
├── config.py                # Configuration and logging
├── start.py                 # Startup script
├── requirements.txt         # Dependencies
└── README_ANALYZER.md       # This file
```

### Key Improvements

1. **Fixed Tree-sitter Loading**: Robust language loading with proper error handling
2. **Modular Architecture**: Clean separation of concerns
3. **Production Logging**: Comprehensive logging with rotation
4. **Error Handling**: Specific exception catching and detailed error messages
5. **Performance**: Optimized parsing and better node ID generation
6. **Configuration**: Environment-based configuration management

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Redis** (required for Celery):
   ```bash
   # Windows: Download Redis from https://redis.io/download
   # Linux/Mac: redis-server
   ```

3. **Verify Installation**:
   ```bash
   python start.py --check-redis
   ```

## Usage

### Quick Start

```bash
# Start all services (Flask app + Celery worker)
python start.py

# Start only Celery worker
python start.py --worker-only

# Start only Flask app
python start.py --app-only
```

### API Endpoints

- **POST /parse**: Analyze a codebase
  ```json
  {
    "folder_path": "/path/to/your/project"
  }
  ```

- **GET /status/<job_id>**: Check analysis status

- **GET /health**: Health check

### Example Usage

```bash
# Start the analyzer
python start.py

# In another terminal, test the API
curl -X POST http://localhost:5000/parse \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "/path/to/your/project"}'

# Check status
curl http://localhost:5000/status/<job_id>
```

## Configuration

Environment variables:

- `CELERY_BROKER_URL`: Redis broker URL (default: redis://localhost:6379/0)
- `CELERY_RESULT_BACKEND`: Redis result backend (default: redis://localhost:6379/0)
- `ANALYZER_ASYNC`: Enable async mode (default: 1)
- `LOG_LEVEL`: Logging level (default: INFO)
- `LOG_FILE`: Log file path (default: logs/analyzer.log)
- `FLASK_ENV`: Flask environment (default: development)
- `FLASK_DEBUG`: Flask debug mode (default: true)

## Supported Languages

- **Python**: AST-based parsing
- **JavaScript/TypeScript**: Tree-sitter parsing
- **Java**: Tree-sitter parsing
- **HTML**: Basic template detection

## Project Types

- React/Vite
- Angular
- Django
- Spring Boot
- Maven/Gradle Java
- Express.js
- Android
- Generic Python

## Tree-sitter Fix

The critical tree-sitter loading issue has been resolved:

1. **Removed tree-sitter-languages dependency**: Direct import from individual packages
2. **Robust fallback logic**: Proper error handling for missing languages
3. **Windows compatibility**: Fixed language loading on Windows
4. **Better error messages**: Detailed logging for debugging

## Logging

The application now uses proper logging with:

- **Console output**: Real-time progress monitoring
- **File rotation**: Automatic log file rotation (10MB, 5 backups)
- **Structured format**: Timestamp, logger name, level, message
- **Configurable levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL

## Performance Improvements

1. **Better Node IDs**: Hash-based unique IDs to prevent collisions
2. **Efficient Parsing**: Skip already parsed files
3. **Memory Management**: Proper cleanup and resource management
4. **Concurrent Processing**: Celery-based async processing

## Error Handling

- **Specific Exceptions**: Catch ImportError, UnicodeDecodeError, etc.
- **Detailed Logging**: Full stack traces for debugging
- **Graceful Degradation**: Continue processing even if some files fail
- **User-Friendly Messages**: Clear error messages in API responses

## Development

### Running Tests

```bash
# Test the analyzer directly
python -c "from analyzer import CodebaseAnalyzer; print('Import successful')"
```

### Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python start.py
```

### Adding New Languages

1. Add language support in `analyzer/parser.py`
2. Update `requirements.txt` with new tree-sitter package
3. Add project type detection in `analyzer/scanner.py`

## Production Deployment

1. **Use Gunicorn**:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 run:app
   ```

2. **Configure Redis** for production

3. **Set environment variables**:
   ```bash
   export FLASK_ENV=production
   export FLASK_DEBUG=false
   export LOG_LEVEL=INFO
   ```

4. **Use process manager** (systemd, supervisor, etc.) for Celery workers

## Troubleshooting

### Common Issues

1. **Tree-sitter languages not loading**:
   - Check if packages are installed: `pip list | grep tree-sitter`
   - Verify Python version compatibility
   - Check Windows PATH variables

2. **Redis connection failed**:
   - Ensure Redis is running: `redis-cli ping`
   - Check Redis port (default: 6379)

3. **Empty analysis results**:
   - Check logs for parsing errors
   - Verify project structure
   - Test with a simple project first

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python start.py
```

Check the log file for detailed information about parsing failures.

## Migration from Old Structure

The old `app.py` has been completely refactored. Key changes:

1. **Import changes**: Use `from analyzer import CodebaseAnalyzer`
2. **Configuration**: Use environment variables instead of hardcoded values
3. **Logging**: Replace `print()` statements with proper logging
4. **Error handling**: More specific exception handling

## License

This project maintains the same license as the original codebase.
