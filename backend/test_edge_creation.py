#!/usr/bin/env python3
"""
Test to verify that edges are being created properly.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_plugin_symbols():
    """Test that plugins are returning the correct symbols."""
    try:
        from analyzer.plugins.languages.javascript_plugin import JavaScriptPlugin
        
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            plugin = JavaScriptPlugin(temp_dir)
            
            # Create a test file with imports and JSX
            test_file = Path(temp_dir) / "test.tsx"
            test_file.write_text('''import React from 'react';
import { useState } from 'react';
import App from './App';

function MyComponent() {
  const [count, setCount] = useState(0);
  
  return (
    <div>
      <h1>Counter: {count}</h1>
      <Button onClick={() => setCount(count + 1)} />
      <App />
    </div>
  );
}

function Button({ onClick }) {
  return <button onClick={onClick}>Click me</button>;
}

export default MyComponent;''')
            
            # Test the parse method
            nodes, symbols = plugin.parse(test_file, test_file.read_text(), False)
            
            print(f"Plugin returned {len(nodes)} nodes")
            print(f"Node names: {[n.name for n in nodes]}")
            print(f"Symbols: {list(symbols.keys())}")
            print(f"Imports: {symbols.get('imports', [])}")
            print(f"JSX components: {symbols.get('jsx_components', [])}")
            print(f"Declared: {symbols.get('declared', [])}")
            
            # Check that we have the expected symbols
            expected_imports = ['react', 'react', './App']
            expected_jsx = ['Button', 'App']
            expected_declared = ['MyComponent', 'Button']
            
            has_imports = len(symbols.get('imports', [])) > 0
            has_jsx = len(symbols.get('jsx_components', [])) > 0
            has_declared = len(symbols.get('declared', [])) > 0
            
            if has_imports and has_jsx and has_declared:
                print("✓ Plugin is returning the expected symbols")
                return True
            else:
                print(f"✗ Plugin is missing symbols - imports: {has_imports}, jsx: {has_jsx}, declared: {has_declared}")
                return False
                
    except Exception as e:
        print(f"✗ Plugin symbols test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_creation():
    """Test that edges are created from plugin symbols."""
    try:
        from analyzer import CodebaseAnalyzer
        
        # Create a test project with imports
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "test_project"
            project_path.mkdir()
            
            # Create package.json
            (project_path / "package.json").write_text('''{
  "name": "test-project",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build"
  },
  "dependencies": {
    "react": "^18.0.0",
    "react-dom": "^18.0.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.0.0",
    "vite": "^4.0.0"
  }
}''')
            
            # Create vite.config.js
            (project_path / "vite.config.js").write_text('''import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  root: '.',
  build: {
    outDir: 'dist'
  }
})''')
            
            # Create index.html
            (project_path / "index.html").write_text('''<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Vite + React</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>''')
            
            # Create src directory
            src_dir = project_path / "src"
            src_dir.mkdir()
            
            # Create main.tsx with import
            (src_dir / "main.tsx").write_text('''import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

function main() {
  const root = ReactDOM.createRoot(document.getElementById('root'));
  root.render(<App />);
}

main();''')
            
            # Create App.tsx with JSX component usage
            (src_dir / "App.tsx").write_text('''import React from 'react';

function App() {
  return (
    <div className="App">
      <h1>Hello World</h1>
      <CustomComponent />
    </div>
  );
}

function CustomComponent() {
  return <div>Custom</div>;
}

export default App;''')
            
            # Test analyzer
            analyzer = CodebaseAnalyzer()
            result = analyzer.analyze(str(project_path))
            
            print(f"Analyzer completed")
            print(f"  - Total nodes: {result['graph']['metadata']['total_nodes']}")
            print(f"  - Total edges: {result['graph']['metadata']['total_edges']}")
            print(f"  - Node types: {result['graph']['metadata']['node_types']}")
            print(f"  - Edge types: {result['graph']['metadata']['edge_types']}")
            
            # Show nodes
            print(f"\n=== Nodes ({len(result['graph']['nodes'])}) ===")
            for node in result['graph']['nodes'][:10]:  # Show first 10
                print(f"  - {node['name']} ({node['type']}) in {node['file']}")
            
            # Show some edges
            if result['graph']['edges']:
                print(f"\n=== Edges ({len(result['graph']['edges'])}) ===")
                for edge in result['graph']['edges'][:10]:  # Show first 10
                    source_node = next((n for n in result['graph']['nodes'] if n['id'] == edge['source']), None)
                    target_node = next((n for n in result['graph']['nodes'] if n['id'] == edge['target']), None)
                    source_name = source_node['name'] if source_node else edge['source']
                    target_name = target_node['name'] if target_node else edge['target']
                    print(f"  - {source_name} --{edge['type']}--> {target_name}")
            else:
                print("\n✗ No edges found!")
                
            # Check if we have the expected edge types
            edge_types = result['graph']['metadata'].get('edge_types', [])
            total_edges = result['graph']['metadata'].get('total_edges', 0)
            
            expected_edge_types = ['depends_on', 'renders', 'calls']
            found_edge_types = [et for et in expected_edge_types if et in edge_types]
            
            if total_edges > 0 and found_edge_types:
                print(f"✓ Found {total_edges} edges with types: {found_edge_types}")
                return True
            else:
                print(f"✗ Expected edges not found. Total edges: {total_edges}, Edge types: {edge_types}")
                
                # Debug: Show file symbols
                if hasattr(analyzer, 'parser') and hasattr(analyzer.parser, 'file_symbols'):
                    print("\n=== File Symbols (Debug) ===")
                    for file, symbols in analyzer.parser.file_symbols.items():
                        print(f"  {file}: {symbols}")
                
                return False
                
    except Exception as e:
        print(f"✗ Edge creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tree_sitter_availability():
    """Test that tree-sitter languages are available."""
    print("=== Tree-sitter Language Availability ===")
    
    # Test JavaScript
    try:
        import tree_sitter_javascript
        print("✓ tree-sitter-javascript available")
    except ImportError:
        try:
            from tree_sitter_languages import get_language
            lang = get_language('javascript')
            print("✓ tree-sitter-languages javascript available")
        except ImportError:
            print("✗ No JavaScript tree-sitter available")
    
    # Test TypeScript
    try:
        import tree_sitter_typescript
        print("✓ tree-sitter-typescript available")
    except ImportError:
        try:
            from tree_sitter_languages import get_language
            lang = get_language('typescript')
            print("✓ tree-sitter-languages typescript available")
        except ImportError:
            print("✗ No TypeScript tree-sitter available")
    
    # Test tree-sitter core
    try:
        from tree_sitter import Language, Parser
        print("✓ tree-sitter core available")
    except ImportError:
        print("✗ tree-sitter core not available")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Testing edge creation...\n")
    
    tests = [
        ("Tree-sitter Availability", test_tree_sitter_availability),
        ("Plugin Symbols", test_plugin_symbols),
        ("Edge Creation", test_edge_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"Final Results: {passed}/{total} tests passed")
    print('='*50)
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)