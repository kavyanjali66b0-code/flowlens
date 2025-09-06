"""
Angular project scanner plugin.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from ..base import ScannerPlugin
from ...models import ProjectType, ConfigFile


class AngularPlugin(ScannerPlugin):
    """Plugin for detecting Angular projects."""
    
    def is_applicable(self, files: List[Path]) -> bool:
        """Check if this is an Angular project."""
        file_names = [f.name for f in files]
        
        # Check for angular.json
        if 'angular.json' in file_names:
            return True
        
        # Check for package.json with Angular dependencies
        package_json_path = self.project_path / 'package.json'
        if package_json_path.exists():
            try:
                with open(package_json_path, 'r', encoding='utf-8') as f:
                    package_data = json.load(f)
                    deps = package_data.get('dependencies', {})
                    dev_deps = package_data.get('devDependencies', {})
                    all_deps = {**deps, **dev_deps}
                    
                    # Check for Angular
                    if '@angular/core' in all_deps or 'angular' in all_deps:
                        return True
            except (json.JSONDecodeError, OSError):
                pass
        
        return False
    
    def get_project_type(self) -> ProjectType:
        return ProjectType.ANGULAR
    
    def get_config_files(self) -> List[Dict[str, Any]]:
        """Return Angular configuration files."""
        config_files = []
        
        # Check for angular.json
        angular_json_path = self.project_path / 'angular.json'
        if angular_json_path.exists():
            try:
                with open(angular_json_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                config_files.append({
                    'type': 'angular.json',
                    'path': 'angular.json',
                    'content': content
                })
            except (json.JSONDecodeError, OSError):
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
        
        return config_files
