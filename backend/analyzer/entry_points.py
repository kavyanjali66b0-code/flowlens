"""
Entry point identification for different project types.
"""

import os
import re
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

from .models import ProjectType, ConfigFile
from .entrypoint_plugins import get_all_plugins


class EntryPointIdentifier:
    """Identifies entry points in a project based on its type."""
    
    def __init__(self, project_path: str, config_files: List[ConfigFile], project_type: ProjectType):
        """Initialize the entry point identifier.
        
        Args:
            project_path: Path to the project directory
            config_files: List of configuration files found in the project
            project_type: Detected project type
        """
        self.project_path = Path(project_path)
        self.config_files = config_files
        self.project_type = project_type
        self.entry_points: List[Dict] = []
        
    def identify(self) -> List[Dict]:
        """
        Identify entry points using the plugin system.
        The return value must be a list of dictionaries as before.
        """
        logging.info(f"Identifying entry points for project type: {self.project_type.value}")

        all_entry_points: List[Dict] = []
        plugin_found = False

        for plugin_class in get_all_plugins():
            try:
                plugin = plugin_class(self)
                if plugin.is_applicable():
                    logging.info(f"Using '{plugin.name}' plugin to find entry points.")
                    all_entry_points.extend(plugin.find_entry_points())
                    plugin_found = True
                    break
            except Exception as e:
                logging.warning(f"Plugin {getattr(plugin_class, 'name', plugin_class.__name__)} failed: {e}")

        # Preserve generic fallback
        if not plugin_found:
            self._find_generic_entry()
            all_entry_points.extend(self.entry_points)

        logging.info(f"Found {len(all_entry_points)} entry points")
        return all_entry_points
    
    # All framework-specific methods removed in favor of plugins.
    
    def _find_generic_entry(self):
        """Generic entry point detection."""
        # Look for main.* files
        for ext in ['py', 'js', 'ts', 'java']:
            main_path = self.project_path / f'main.{ext}'
            if main_path.exists():
                self.entry_points.append({
                    "type": "generic_main",
                    "value": f"main.{ext}",
                    "source": "convention"
                })
                logging.debug(f"Found generic main file: main.{ext}")
                break
