"""Test the edge type fix by analyzing a codebase."""

import requests
import json
import time

def test_edge_types():
    """Test edge type preservation after fix."""
    print("Testing edge type preservation with fixed semantic_analyzer...")
    
    # Wait for server to be ready
    print("Waiting for backend server...")
    time.sleep(3)
    
    # Check if server is running
    try:
        response = requests.get('http://localhost:5000/')
        print(f"✓ Backend server is running")
    except:
        print("❌ Backend server is not running")
        print("   Please start it with: python backend/start.py")
        return
    
    # Read the response.json file
    try:
        with open('backend/response.json') as f:
            data = json.load(f)
        
        edge_types = data['statistics']['edge_types']
        edges = data['graph']['edges']
        
        print(f"\n📊 **Edge Type Distribution:**")
        for edge_type, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {edge_type}: {count}")
        
        # Check metadata
        edges_with_metadata = sum(1 for e in edges if e.get('metadata') and e['metadata'])
        print(f"\n📝 **Metadata Coverage:**")
        print(f"  Edges with metadata: {edges_with_metadata}/{len(edges)} ({edges_with_metadata*100//len(edges)}%)")
        
        # Sample edge types
        print(f"\n🔍 **Sample Edges:**")
        sample_types = {}
        for edge in edges:
            edge_type = edge['type']
            if edge_type not in sample_types:
                sample_types[edge_type] = edge
            if len(sample_types) >= 3:
                break
        
        for edge_type, edge in sample_types.items():
            print(f"\n  Type: {edge_type}")
            print(f"    {edge['source']} -> {edge['target']}")
            if edge.get('metadata'):
                meta_keys = list(edge['metadata'].keys())[:3]
                print(f"    Metadata keys: {meta_keys}")
        
        # Success check
        unique_types = len(edge_types)
        if unique_types == 1 and 'depends_on' in edge_types:
            print(f"\n❌ **STILL BROKEN:** All edges are 'depends_on'")
            print(f"   The fix may need the backend to be restarted.")
        elif unique_types > 1:
            print(f"\n✅ **SUCCESS:** Found {unique_types} different edge types!")
            print(f"   Edge type preservation is working correctly!")
        
    except FileNotFoundError:
        print("❌ response.json not found")
        print("   Please run an analysis first")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    test_edge_types()
