"""
Vite/React project scanner plugin.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from ..base import ScannerPlugin
from ...models import ProjectType, ConfigFile


class VitePlugin(ScannerPlugin):
    """Plugin for detecting Vite/React projects."""
    
    def is_applicable(self, files: List[Path]) -> bool:
        """Check if this is a Vite project."""
        file_names = [f.name for f in files]
        
        # Check for Vite config files
        vite_configs = ['vite.config.js', 'vite.config.ts', 'vite.config.mjs']
        if any(config in file_names for config in vite_configs):
            return True
        
        # Check for package.json with Vite dependencies
        package_json_path = self.project_path / 'package.json'
        if package_json_path.exists():
            try:
                with open(package_json_path, 'r', encoding='utf-8') as f:
                    package_data = json.load(f)
                    deps = package_data.get('dependencies', {})
                    dev_deps = package_data.get('devDependencies', {})
                    all_deps = {**deps, **dev_deps}
                    
                    # Check for Vite
                    if 'vite' in all_deps:
                        return True
                    
                    # Check for React with Vite-like setup
                    if 'react' in all_deps and 'index.html' in file_names:
                        return True
            except (json.JSONDecodeError, OSError):
                pass
        
        return False
    
    def get_project_type(self) -> ProjectType:
        return ProjectType.REACT_VITE
    
    def get_config_files(self) -> List[Dict[str, Any]]:
        """Return Vite configuration files."""
        config_files = []
        
        # Check for Vite config files
        vite_configs = ['vite.config.js', 'vite.config.ts', 'vite.config.mjs']
        for config_name in vite_configs:
            config_path = self.project_path / config_name
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    config_files.append({
                        'type': 'vite.config',
                        'path': config_name,
                        'content': content
                    })
                except (OSError, UnicodeDecodeError):
                    pass
        
        # Check for package.json
        package_json_path = self.project_path / 'package.json'
        if package_json_path.exists():
            try:
                with open(package_json_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                config_files.append({
                    'type': 'package.json',
                    'path': 'package.json',
                    'content': content
                })
            except (json.JSONDecodeError, OSError):
                pass
        
        # Check for index.html
        index_html_path = self.project_path / 'index.html'
        if index_html_path.exists():
            try:
                with open(index_html_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                config_files.append({
                    'type': 'index.html',
                    'path': 'index.html',
                    'content': content
                })
            except (OSError, UnicodeDecodeError):
                pass
        
        return config_files
