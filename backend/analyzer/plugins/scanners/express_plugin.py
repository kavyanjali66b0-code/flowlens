"""
Express.js project scanner plugin.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from ..base import ScannerPlugin
from ...models import ProjectType, ConfigFile


class ExpressPlugin(ScannerPlugin):
    """Plugin for detecting Express.js projects."""
    
    def is_applicable(self, files: List[Path]) -> bool:
        """Check if this is an Express.js project."""
        file_names = [f.name for f in files]
        
        # Check for package.json with Express dependencies
        package_json_path = self.project_path / 'package.json'
        if package_json_path.exists():
            try:
                with open(package_json_path, 'r', encoding='utf-8') as f:
                    package_data = json.load(f)
                    deps = package_data.get('dependencies', {})
                    dev_deps = package_data.get('devDependencies', {})
                    all_deps = {**deps, **dev_deps}
                    
                    # Check for Express
                    if 'express' in all_deps:
                        return True
            except (json.JSONDecodeError, OSError):
                pass
        
        # Check for common Express entry files
        express_files = ['app.js', 'server.js', 'index.js']
        if any(express_file in file_names for express_file in express_files):
            return True
        
        return False
    
    def get_project_type(self) -> ProjectType:
        return ProjectType.EXPRESS_NODE
    
    def get_config_files(self) -> List[Dict[str, Any]]:
        """Return Express.js configuration files."""
        config_files = []
        
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
        
        # Check for common Express entry files
        express_files = ['app.js', 'server.js', 'index.js']
        for express_file in express_files:
            file_path = self.project_path / express_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    config_files.append({
                        'type': 'express_entry',
                        'path': express_file,
                        'content': content
                    })
                except (OSError, UnicodeDecodeError):
                    pass
        
        return config_files
