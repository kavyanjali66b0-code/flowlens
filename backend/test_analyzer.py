#!/usr/bin/env python3
"""
Test script to verify the refactored analyzer works correctly.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import setup_logging, get_config
from analyzer import CodebaseAnalyzer


def create_test_project():
    """Create a simple test project for analysis."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir) / "test_project"
        project_path.mkdir()
        
        # Create a simple React project structure
        (project_path / "package.json").write_text('''{
  "name": "test-project",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.0.0",
    "vite": "^4.0.0"
  }
}''')
        
        (project_path / "index.html").write_text('''<!DOCTYPE html>
<html>
<head><title>Test App</title></head>
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.tsx"></script>
</body>
</html>''')
        
        # Create src directory
        src_dir = project_path / "src"
        src_dir.mkdir()
        
        # Create main.tsx
        (src_dir / "main.tsx").write_text('''import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

function main() {
  const root = ReactDOM.createRoot(document.getElementById('root'));
  root.render(<App />);
}

main();''')
        
        # Create App.tsx
        (src_dir / "App.tsx").write_text('''import React from 'react';

function App() {
  return (
    <div className="App">
      <h1>Hello World</h1>
    </div>
  );
}

export default App;''')
        
        return str(project_path)


def test_analyzer():
    """Test the analyzer with a simple project."""
    print("Testing refactored codebase analyzer...")
    
    # Setup logging
    config = get_config()
    setup_logging('INFO', None)  # Console only for testing
    
    try:
        # Create test project
        print("Creating test project...")
        test_project_path = create_test_project()
        print(f"Test project created at: {test_project_path}")
        
        # Test analyzer
        print("Running analysis...")
        analyzer = CodebaseAnalyzer()
        result = analyzer.analyze(test_project_path)
        
        # Check results
        print("\n=== Analysis Results ===")
        print(f"Project type: {result['project_info']['type']}")
        print(f"Config files found: {len(result['project_info']['config_files'])}")
        print(f"Entry points found: {len(result['project_info']['entry_points'])}")
        print(f"Nodes generated: {result['graph']['metadata']['total_nodes']}")
        print(f"Edges generated: {result['graph']['metadata']['total_edges']}")
        
        # Show some nodes
        if result['graph']['nodes']:
            print("\n=== Sample Nodes ===")
            for node in result['graph']['nodes'][:3]:
                print(f"  - {node['name']} ({node['type']}) in {node['file']}")
        
        # Show some edges
        if result['graph']['edges']:
            print("\n=== Sample Edges ===")
            for edge in result['graph']['edges'][:3]:
                print(f"  - {edge['source']} -> {edge['target']} ({edge['type']})")
        
        print("\n✓ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_analyzer()
    sys.exit(0 if success else 1)
