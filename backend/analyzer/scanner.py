"""
Project scanner for detecting project types and configuration files.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Set

from .models import ProjectType, ConfigFile
from .plugins.scanners import (
    VitePlugin, MavenPlugin, DjangoPlugin, AngularPlugin,
    ExpressPlugin, SpringBootPlugin, AndroidPlugin
)


class ProjectScanner:
    """Scans a project directory to detect project type and configuration files using plugins."""
    
    def __init__(self, project_path: str):
        """Initialize the project scanner.
        
        Args:
            project_path: Path to the project directory to scan
        """
        self.project_path = Path(project_path)
        self.config_files: List[ConfigFile] = []
        self.detected_types: Set[ProjectType] = set()
        
        # Load user configuration
        from .config_loader import ConfigLoader
        self.user_config = ConfigLoader.load(self.project_path)
        logging.info(f"Loaded config for scanner: {self.user_config.project_type or 'auto-detect'}")
        
        # Initialize scanner plugins
        self.scanner_plugins = [
            VitePlugin(project_path),
            MavenPlugin(project_path),
            DjangoPlugin(project_path),
            AngularPlugin(project_path),
            ExpressPlugin(project_path),
            SpringBootPlugin(project_path),
            AndroidPlugin(project_path)
        ]
        
    def scan(self) -> Dict[str, Any]:
        """Scan project using scanner plugins to detect project type and configuration files.
        
        Returns:
            Dictionary containing config files and detected project types
        """
        if not self.project_path.exists():
            raise ValueError(f"Path {self.project_path} does not exist")
            
        logging.info(f"Scanning project at: {self.project_path}")
        
        # Get all files in the project
        all_files = []
        for root, dirs, files in os.walk(self.project_path):
            # Filter directories using user config exclusion patterns
            dirs[:] = [d for d in dirs if not self._should_exclude(d)]
            
            for file in files:
                file_path = Path(root) / file
                all_files.append(file_path)
        
        # Try each scanner plugin
        for plugin in self.scanner_plugins:
            if plugin.is_applicable(all_files):
                project_type = plugin.get_project_type()
                config_data = plugin.get_config_files()
                
                self.detected_types.add(project_type)
                logging.info(f"Detected project type: {project_type.value} using {plugin.__class__.__name__}")
                
                # Convert config data to ConfigFile objects
                for config_dict in config_data:
                    config_file = ConfigFile(
                        path=config_dict['path'],
                        type=config_dict['type'],
                        content=config_dict.get('content')
                    )
                    self.config_files.append(config_file)
                    logging.debug(f"Found config file: {config_dict['path']} -> {config_dict['type']}")
                
                # Stop after first successful detection
                break
        
        # If no plugin detected anything, mark as unknown
        if not self.detected_types:
            self.detected_types.add(ProjectType.UNKNOWN)
            logging.warning("No project type detected, marking as UNKNOWN")
        
        logging.info(f"Scan complete. Found {len(self.config_files)} config files, "
                    f"detected types: {[t.value for t in self.detected_types]}")
        
        return {
            "config_files": [{"path": c.path, "type": c.type} for c in self.config_files],
            "detected_types": [t.value for t in self.detected_types] or [ProjectType.UNKNOWN.value]
        }
    
    def _should_exclude(self, directory_name: str) -> bool:
        """Check if a directory should be excluded from scanning.
        
        Args:
            directory_name: Name of the directory to check
            
        Returns:
            bool: True if directory should be excluded, False otherwise
        """
        import fnmatch
        
        # Check against user-configured exclusion patterns
        for pattern in self.user_config.exclude:
            # Support both exact matches and glob patterns
            if fnmatch.fnmatch(directory_name, pattern) or directory_name == pattern:
                logging.debug(f"Excluding directory: {directory_name} (matched pattern: {pattern})")
                return True
        
        return False
    
    def _enhance_detection_from_content(self):
        """Enhance project type detection based on config file content.
        
        This method is kept for backward compatibility but is no longer used
        since detection is now handled by plugins.
        """
        pass
