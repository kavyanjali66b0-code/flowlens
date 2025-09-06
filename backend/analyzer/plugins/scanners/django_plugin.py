"""
Django project scanner plugin.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

from ..base import ScannerPlugin
from ...models import ProjectType, ConfigFile


class DjangoPlugin(ScannerPlugin):
    """Plugin for detecting Django projects."""
    
    def is_applicable(self, files: List[Path]) -> bool:
        """Check if this is a Django project."""
        file_names = [f.name for f in files]
        
        # Check for Django-specific files
        django_files = ['manage.py', 'settings.py', 'urls.py', 'wsgi.py', 'asgi.py']
        if any(django_file in file_names for django_file in django_files):
            return True
        
        # Check for requirements.txt with Django
        requirements_path = self.project_path / 'requirements.txt'
        if requirements_path.exists():
            try:
                with open(requirements_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    if 'django' in content:
                        return True
            except (OSError, UnicodeDecodeError):
                pass
        
        return False
    
    def get_project_type(self) -> ProjectType:
        return ProjectType.DJANGO
    
    def get_config_files(self) -> List[Dict[str, Any]]:
        """Return Django configuration files."""
        config_files = []
        
        # Check for manage.py
        manage_py_path = self.project_path / 'manage.py'
        if manage_py_path.exists():
            try:
                with open(manage_py_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                config_files.append({
                    'type': 'manage.py',
                    'path': 'manage.py',
                    'content': content
                })
            except (OSError, UnicodeDecodeError):
                pass
        
        # Check for settings.py files
        for root, dirs, files in os.walk(self.project_path):
            for file in files:
                if file == 'settings.py':
                    settings_path = Path(root) / file
                    relative_path = settings_path.relative_to(self.project_path)
                    try:
                        with open(settings_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        config_files.append({
                            'type': 'settings.py',
                            'path': str(relative_path),
                            'content': content
                        })
                    except (OSError, UnicodeDecodeError):
                        pass
        
        # Check for urls.py files
        for root, dirs, files in os.walk(self.project_path):
            for file in files:
                if file == 'urls.py':
                    urls_path = Path(root) / file
                    relative_path = urls_path.relative_to(self.project_path)
                    try:
                        with open(urls_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        config_files.append({
                            'type': 'urls.py',
                            'path': str(relative_path),
                            'content': content
                        })
                    except (OSError, UnicodeDecodeError):
                        pass
        
        return config_files
