"""
Custom exceptions for FlowLens analyzer.

Provides structured error handling with detailed context for debugging and user feedback.
"""

class AnalyzerError(Exception):
    """Base exception for analyzer errors."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def to_dict(self) -> dict:
        """Convert to API error response format."""
        return {
            'type': self.__class__.__name__,
            'message': self.message,
            'details': self.details
        }


class ParsingError(AnalyzerError):
    """Failed to parse file."""
    
    def __init__(self, file_path: str, reason: str, line: int = None):
        super().__init__(
            f"Failed to parse {file_path}: {reason}",
            details={
                'file': file_path,
                'reason': reason,
                'line': line
            }
        )


class MemoryLimitExceeded(AnalyzerError):
    """Memory limit exceeded during analysis."""
    
    def __init__(self, current_mb: float, limit_mb: float):
        super().__init__(
            f"Memory limit exceeded: {current_mb:.1f}MB > {limit_mb}MB",
            details={
                'current_mb': round(current_mb, 2),
                'limit_mb': round(limit_mb, 2)
            }
        )


class ConfigurationError(AnalyzerError):
    """Invalid configuration."""
    
    def __init__(self, message: str, config_file: str = None):
        super().__init__(
            message,
            details={'config_file': config_file}
        )


class ProjectTooLargeError(AnalyzerError):
    """Project exceeds maximum file limit."""
    
    def __init__(self, file_count: int, max_files: int):
        super().__init__(
            f"Project too large: {file_count} files > {max_files} limit",
            details={
                'file_count': file_count,
                'max_files': max_files
            }
        )


class TimeoutError(AnalyzerError):
    """Analysis exceeded time limit."""
    
    def __init__(self, elapsed_seconds: float, limit_seconds: float):
        super().__init__(
            f"Analysis timeout: {elapsed_seconds:.1f}s > {limit_seconds}s",
            details={
                'elapsed_seconds': round(elapsed_seconds, 2),
                'limit_seconds': round(limit_seconds, 2)
            }
        )


class InvalidProjectError(AnalyzerError):
    """Project path is invalid or inaccessible."""
    
    def __init__(self, project_path: str, reason: str):
        super().__init__(
            f"Invalid project: {project_path} - {reason}",
            details={
                'project_path': project_path,
                'reason': reason
            }
        )


class UnsupportedLanguageError(AnalyzerError):
    """Language/framework not supported."""
    
    def __init__(self, language: str, file_extension: str = None):
        super().__init__(
            f"Unsupported language: {language}",
            details={
                'language': language,
                'file_extension': file_extension
            }
        )
