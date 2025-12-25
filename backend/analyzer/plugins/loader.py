"""
Plugin loader for the analyzer system.
"""

import sys
import inspect
import importlib
import pkgutil
import logging
from typing import List, Dict, Any, Type

from .base import LanguagePlugin

def discover_language_plugins(project_path: str, user_config: Dict[str, Any] = None, symbol_table=None) -> List[LanguagePlugin]:
    """
    Discover and instantiate available language plugins.
    
    Args:
        project_path: Path to the project root
        user_config: User configuration dictionary
        symbol_table: SymbolTable instance
        
    Returns:
        List of instantiated LanguagePlugin objects
    """
    plugins = []
    
    # Import languages package to ensure it's loaded
    from . import languages
    
    # Iterate through modules in the languages package
    package_path = languages.__path__
    for _, name, _ in pkgutil.iter_modules(package_path):
        try:
            # Import the module
            module_name = f"{languages.__name__}.{name}"
            module = importlib.import_module(module_name)
            
            # Find classes that inherit from LanguagePlugin
            for _, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, LanguagePlugin) and 
                    obj != LanguagePlugin):
                    
                    try:
                        # Instantiate the plugin
                        # Check signature to see if it accepts symbol_table
                        sig = inspect.signature(obj.__init__)
                        if 'symbol_table' in sig.parameters:
                            plugin = obj(project_path, user_config, symbol_table)
                        else:
                            plugin = obj(project_path, user_config)
                        
                        plugins.append(plugin)
                        logging.debug(f"Loaded plugin: {obj.__name__}")
                    except Exception as e:
                        logging.error(f"Failed to instantiate plugin {obj.__name__}: {e}")
                        
        except ImportError as e:
            logging.error(f"Failed to import plugin module {name}: {e}")
            
    return plugins
