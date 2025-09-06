"""
Plugin system for the codebase analyzer.

This package contains various plugins for analyzing different programming languages
and frameworks.
"""

from .base import BasePlugin
from .languages import *
from .scanners import *

__all__ = [
    'BasePlugin',
    # Language plugins
    'JavaPlugin',
    'JavaScriptPlugin', 
    'PythonPlugin',
    # Scanner plugins
    'AndroidPlugin',
    'AngularPlugin',
    'DjangoPlugin',
    'ExpressPlugin',
    'MavenPlugin',
    'SpringBootPlugin',
    'VitePlugin',
]
