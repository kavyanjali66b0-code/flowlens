# Visualization Improvements: Next Steps

## Current State Analysis

### 🔴 **Critical Issues Found**

#### 1. **All Edges Are "depends_on"**
**Problem:**
- Response shows **102 edges, ALL are "depends_on"**
- Code has edge types: `CALLS`, `IMPORTS`, `RENDERS`, `ASYNC_CALLS`, etc.
- But they're all being converted to `depends_on` somewhere

**Evidence:**
```json
"edge_types": {
    "depends_on": 102  // Should have variety!
}
```

**Root Cause:**
Looking at `parser.py:357` and `parser.py:401`, edges are being created with `EdgeType.DEPENDS_ON` instead of using the proper types from semantic analysis.

#### 2. **Empty Metadata**
**Problem:**
- All edges have `"metadata": {}`
- No context about relationships (line numbers, call sites, etc.)
- Frontend can't show meaningful information

**Impact:**
- Can't distinguish between import vs call vs render
- No way to show "where" the relationship happens
- Visualization looks generic

#### 3. **Cryptic Node IDs**
**Problem:**
- Node IDs like `"src_main_tsx_module_86894175"` are not human-readable
- Frontend has to extract labels from file paths
- Hard to debug and understand

#### 4. **No Hierarchical Structure**
**Problem:**
- Files, classes, and methods are all flat nodes
- No parent-child relationships
- Can't show "file contains class contains method"

**Current Structure:**
```
- src_main_tsx_module (file)
- src_App_tsx_module (file)  
- Index_def65c7f (component)
- handleExport_fbdf63a0 (function)
```
All at same level - no hierarchy!

#### 5. **Layout Algorithm is Basic**
**Problem:**
- `createDependencyLayout` uses simple importance scoring
- Not hierarchical (files → classes → methods)
- Doesn't group related components
- Hard to see code structure

---

## What to Do Next (Priority Order)

### 🟢 **IMMEDIATE (This Week)**

#### 1. **Fix Edge Types - Preserve Semantic Analysis Results**

**Problem:** Edges from semantic analyzer are being lost/overwritten.

**Fix in `backend/analyzer/main.py`:**
```python
# Line 210-216 - This merges edges but might be losing types
existing = {(e.source, e.target, e.type.value) for e in self.parser.edges}
for e in derived_edges:
    key = (e.source, e.target, e.type.value)
    if key not in existing:
        self.parser.edges.append(e)
        existing.add(key)
```

**Issue:** The deduplication uses `(source, target, type)` as key, so if parser creates `depends_on` first, semantic analyzer's `CALLS` edge gets rejected!

**Solution:**
```python
# backend/analyzer/main.py - Fix edge merging
# Don't deduplicate by type - allow multiple edge types between same nodes
existing = {(e.source, e.target) for e in self.parser.edges}
for e in derived_edges:
    # Check if this specific edge type already exists
    edge_exists = any(
        existing_edge.source == e.source and 
        existing_edge.target == e.target and 
        existing_edge.type == e.type
        for existing_edge in self.parser.edges
    )
    if not edge_exists:
        self.parser.edges.append(e)
```

#### 2. **Enrich Edge Metadata**

**Problem:** Edges have empty metadata.

**Fix in `backend/analyzer/semantic_analyzer.py`:**
```python
# Line 188-194 - Already has metadata, but check if it's being preserved
edge_metadata = {
    'confidence_score': confidence,
    'context': 'calls relationship',
    'flow_type': 'function_call',
    'verified_from_ast': True,
    'is_async': reference.context == 'async_call',
    # ADD THESE:
    'line_number': reference.line_number,  # Where the call happens
    'column': reference.column,
    'file': source_node.file,
    'call_site': reference.context_snippet,  # Code snippet
}
```

**Also fix in `backend/analyzer/parser.py`:**
```python
# Line 357 - When creating DEPENDS_ON edges, add metadata
edge = Edge(
    source=source_module_id,
    target=target_module_id,
    type=EdgeType.DEPENDS_ON,
    metadata={
        'reason': 'import_statement',  # Why this edge exists
        'import_path': import_path,     # What was imported
        'line_number': import_line,     # Where in file
    }
)
```

#### 3. **Improve Node Labels**

**Problem:** Node IDs are cryptic hashes.

**Fix in `backend/analyzer/parser.py`:**
```python
# When creating nodes, ensure name is human-readable
def _create_node_id(self, file_path: str, name: str, node_type: NodeType) -> str:
    """Create a readable but unique node ID."""
    # Use readable format: file_path::name::type
    sanitized_file = file_path.replace('\\', '/').replace('/', '_').replace('.', '_')
    sanitized_name = name.replace(' ', '_').replace('-', '_')
    return f"{sanitized_file}::{sanitized_name}::{node_type.value}"
```

**Or better - use the name directly if unique:**
```python
# In models.py Node class, ensure name is always set
# Use file basename + symbol name as default
if not node.name:
    node.name = f"{Path(node.file).stem}::{symbol_name}"
```

---

### 🟡 **SHORT TERM (Next 2 Weeks)**

#### 4. **Add Hierarchical Node Structure**

**Goal:** Show files → classes → methods hierarchy.

**Implementation:**

**Backend - Add parent relationships:**
```python
# backend/analyzer/models.py - Add to Node
@dataclass
class Node:
    # ... existing fields ...
    parent_id: Optional[str] = None  # Parent node ID (file → class → method)
    children: List[str] = field(default_factory=list)  # Child node IDs
```

**Backend - Build hierarchy in parser:**
```python
# backend/analyzer/parser.py - Track hierarchy
def _parse_file(self, file_path: Path, is_entry: bool = False):
    # Create file node first
    file_node = Node(
        id=f"file::{file_path}",
        name=file_path.stem,
        type=NodeType.MODULE,
        file=str(file_path),
        # ... other fields
    )
    self.nodes.append(file_node)
    
    # Parse classes/components
    for class_def in classes:
        class_node = Node(
            id=f"class::{file_path}::{class_def.name}",
            name=class_def.name,
            type=NodeType.COMPONENT,
            file=str(file_path),
            parent_id=file_node.id,  # ← Set parent
            # ... other fields
        )
        file_node.children.append(class_node.id)  # ← Add to parent's children
        self.nodes.append(class_node)
        
        # Parse methods
        for method in class_def.methods:
            method_node = Node(
                id=f"method::{file_path}::{class_def.name}::{method.name}",
                name=method.name,
                type=NodeType.FUNCTION,
                file=str(file_path),
                parent_id=class_node.id,  # ← Set parent
            )
            class_node.children.append(method_node.id)
            self.nodes.append(method_node)
```

**Frontend - Use React Flow Groups:**
```typescript
// flowlens-code-explorer/src/services/dataService.ts
function buildHierarchicalNodes(nodes: any[]): GraphNode[] {
  // Group nodes by parent
  const nodesByParent = new Map<string, any[]>();
  const rootNodes: any[] = [];
  
  nodes.forEach(node => {
    if (node.parent_id) {
      if (!nodesByParent.has(node.parent_id)) {
        nodesByParent.set(node.parent_id, []);
      }
      nodesByParent.get(node.parent_id)!.push(node);
    } else {
      rootNodes.push(node);
    }
  });
  
  // Create parent nodes as groups
  return rootNodes.map(parentNode => ({
    id: parentNode.id,
    type: 'group',  // React Flow group node
    position: parentNode.position,
    data: { label: parentNode.name },
    style: { width: 400, height: 300 },  // Container size
    // Child nodes will be positioned relative to this
  }));
}
```

#### 5. **Better Layout Algorithm**

**Current:** Simple importance-based layout (flat)

**Better:** Hierarchical layout with grouping

**Implementation:**
```typescript
// flowlens-code-explorer/src/services/hierarchicalLayout.ts
import dagre from 'dagre';

export function createHierarchicalLayout(
  nodes: GraphNode[],
  edges: GraphEdge[]
): Map<string, { x: number; y: number }> {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: 'TB', nodesep: 50, ranksep: 100 });
  
  // Add nodes with hierarchy levels
  nodes.forEach(node => {
    const width = node.type === 'file' ? 200 : node.type === 'class' ? 150 : 100;
    const height = node.type === 'file' ? 100 : node.type === 'class' ? 80 : 60;
    
    g.setNode(node.id, { width, height });
  });
  
  // Add edges
  edges.forEach(edge => {
    g.setEdge(edge.source, edge.target);
  });
  
  // Calculate layout
  dagre.layout(g);
  
  // Extract positions
  const positions = new Map<string, { x: number; y: number }>();
  g.nodes().forEach(nodeId => {
    const node = g.node(nodeId);
    positions.set(nodeId, { x: node.x, y: node.y });
  });
  
  return positions;
}
```

**Install dependency:**
```bash
cd flowlens-code-explorer
npm install dagre @types/dagre
```

#### 6. **Edge Type Visualization**

**Current:** All edges look the same

**Better:** Different styles for different edge types

**Fix in `dataService.ts`:**
```typescript
// Line 262-276 - Enhance edge styling
const edgeData = {
  id: `edge-${sourceId}-${targetId}`,
  source: sourceId,
  target: targetId,
  type: 'default',
  animated: edge.type === 'calls' || edge.type === 'async_calls',
  label: edge.type || '',
  style: getEdgeStyle(edge.type),  // ← New function
  edgeType: edge.type || 'depends_on',
  metadata: edge.metadata || {},
};

function getEdgeStyle(edgeType: string) {
  const styles = {
    'calls': { 
      strokeWidth: 3, 
      stroke: '#f59e0b',  // Orange
      strokeDasharray: '5,5'  // Dashed
    },
    'async_calls': {
      strokeWidth: 3,
      stroke: '#8b5cf6',  // Purple
      strokeDasharray: '10,5'  // Different dash
    },
    'imports': {
      strokeWidth: 2,
      stroke: '#3b82f6',  // Blue
      opacity: 0.7
    },
    'renders': {
      strokeWidth: 2.5,
      stroke: '#10b981',  // Green
    },
    'depends_on': {
      strokeWidth: 1.5,
      stroke: '#94a3b8',  // Gray
      opacity: 0.5
    }
  };
  return styles[edgeType] || styles['depends_on'];
}
```

---

### 🟢 **MEDIUM TERM (Next Month)**

#### 7. **Add Edge Labels with Context**

**Show what the relationship is:**
```typescript
// In GraphView.tsx - Add edge labels
<ReactFlow
  // ... existing props
  defaultEdgeOptions={{
    labelStyle: { fill: '#666', fontWeight: 500 },
    labelBgStyle: { fill: '#fff', fillOpacity: 0.8 },
  }}
>
  {/* Edges will show labels from edge.label */}
</ReactFlow>
```

**Backend - Set meaningful labels:**
```python
# In edge creation
edge = Edge(
    source=source_node.id,
    target=target_node.id,
    type=EdgeType.CALLS,
    metadata={
        'label': f"calls {target_node.name}()",  # Human-readable
        'line_number': 42,
    }
)
```

#### 8. **Filter by Edge Type**

**Add UI controls:**
```typescript
// In GraphView.tsx - Add edge type filter
const [visibleEdgeTypes, setVisibleEdgeTypes] = useState<Set<string>>(
  new Set(['calls', 'imports', 'renders'])
);

const filteredEdges = useMemo(() => {
  return edges.filter(edge => 
    visibleEdgeTypes.has(edge.edgeType)
  );
}, [edges, visibleEdgeTypes]);
```

#### 9. **Show Node Details on Hover**

**Enhance CustomNode component:**
```typescript
// In GraphView.tsx - Add tooltip
const CustomNode: React.FC<{ data: any; selected: boolean }> = ({ data, selected }) => {
  return (
    <Tooltip content={
      <div>
        <div><strong>File:</strong> {data.file}</div>
        <div><strong>Type:</strong> {data.type}</div>
        {data.metadata?.c4_level && (
          <div><strong>Level:</strong> {data.metadata.c4_level}</div>
        )}
        {data.metadata?.is_entry && (
          <div><strong>Entry Point</strong></div>
        )}
      </div>
    }>
      {/* Existing node content */}
    </Tooltip>
  );
};
```

---

## Quick Wins (Do These First!)

### 1. **Fix Edge Type Preservation** (30 minutes)
- Modify `main.py` edge merging logic
- Test that CALLS, IMPORTS edges appear in response

### 2. **Add Edge Metadata** (1 hour)
- Add line numbers, file paths to edge metadata
- Test that metadata appears in JSON

### 3. **Improve Edge Visualization** (1 hour)
- Update `dataService.ts` to style edges by type
- Test that different edge types look different

### 4. **Better Node Labels** (30 minutes)
- Ensure node.name is always set to readable value
- Test that nodes show proper names in visualization

---

## Testing Checklist

After implementing fixes:

- [ ] Response JSON shows multiple edge types (not just `depends_on`)
- [ ] Edge metadata contains line numbers, file paths
- [ ] Node names are human-readable (not just IDs)
- [ ] Visualization shows different edge styles (colors, dashes)
- [ ] Hovering nodes shows file path and type
- [ ] Layout groups related nodes together
- [ ] Can filter by edge type in UI

---

## Expected Results

**Before:**
- 102 edges, all `depends_on`
- Empty metadata
- Cryptic node IDs
- Flat layout

**After:**
- Mix of `calls`, `imports`, `renders`, `depends_on`
- Rich metadata (line numbers, context)
- Readable node names
- Hierarchical layout with grouping
- Color-coded edges by type

---

## Next Steps Summary

1. **This Week:** Fix edge types and metadata (2-3 hours)
2. **Next Week:** Improve layout and hierarchy (4-6 hours)
3. **Next Month:** Add filtering and better UX (8-10 hours)

Start with the "Quick Wins" - they'll have immediate visual impact!

