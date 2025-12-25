"""
Quick diagnostic script to check JSX component data in file_symbols
"""
import json

with open('response.json', 'r') as f:
    data = json.load(f)

# Check if we have component nodes
components = [n for n in data['graph']['nodes'] if n['type'] == 'component']
print(f"Total components: {len(components)}")
for comp in components:
    print(f"  - {comp['name']} ({comp['file']})")

# Check edge types
print(f"\nEdge type distribution:")
for edge_type, count in data['statistics']['edge_types'].items():
    print(f"  {edge_type}: {count}")
