# Edge Type Strategy: Balancing Flexibility & Simplicity

## The Core Question

**Should we:**
1. Hardcode edge types (calls, imports, renders)?
2. Make it fully dynamic (support all 50+ relationship types)?
3. Use LLM to determine edge types?

## My Recommendation: **Hybrid Approach**

### ✅ **Three-Layer System**

```
Layer 1: Semantic Categories (for visualization)
    ↓
Layer 2: Specific Relationship Types (in metadata)
    ↓
Layer 3: Rich Context (line numbers, code snippets, etc.)
```

---

## Layer 1: Semantic Categories (Visualization)

**Purpose:** Keep visualization simple and understandable

**Core Categories:**
```python
class EdgeCategory(Enum):
    EXECUTION = "execution"      # Runtime behavior (calls, triggers, handles)
    DEPENDENCY = "dependency"    # Structural (imports, depends_on, requires)
    INHERITANCE = "inheritance"  # OOP (extends, implements, overrides)
    DATA_FLOW = "data_flow"      # Data movement (reads, writes, transforms)
    COMPOSITION = "composition"  # Structure (contains, aggregates)
    REFERENCE = "reference"      # Generic (references, uses)
```

**Why Categories?**
- **Visual clarity:** 6 colors/styles vs 50+
- **User mental model:** Easy to understand at a glance
- **Grouping:** Related types share visual style

**Visualization:**
```typescript
const categoryStyles = {
  'execution': { color: '#f59e0b', style: 'dashed', animated: true },
  'dependency': { color: '#3b82f6', style: 'solid', opacity: 0.7 },
  'inheritance': { color: '#8b5cf6', style: 'solid', width: 3 },
  'data_flow': { color: '#10b981', style: 'dotted' },
  'composition': { color: '#6366f1', style: 'solid' },
  'reference': { color: '#94a3b8', style: 'solid', opacity: 0.5 }
};
```

---

## Layer 2: Specific Relationship Types (Metadata)

**Purpose:** Store the exact relationship for detailed analysis

**All relationship types are valid:**
```python
# In Edge.metadata, store the specific type
edge = Edge(
    source=source_node.id,
    target=target_node.id,
    type=EdgeCategory.EXECUTION,  # For visualization
    metadata={
        'relationship_type': 'calls',  # Specific type
        'relationship_subtype': 'async_calls',  # Even more specific
        'line_number': 42,
        'context': 'function_invocation',
        # ... other details
    }
)
```

**Benefits:**
- ✅ Visualization stays simple (6 categories)
- ✅ Full detail available for analysis
- ✅ Can filter/search by specific type
- ✅ Extensible without breaking UI

---

## Layer 3: Rich Context (Full Details)

**Purpose:** Store everything needed for deep analysis

```python
edge.metadata = {
    # Relationship details
    'relationship_type': 'calls',
    'relationship_subtype': 'async_calls',
    'category': 'execution',
    
    # Location
    'source_line': 42,
    'source_column': 15,
    'target_line': 120,
    'code_snippet': 'await fetchData()',
    
    # Context
    'call_args': ['userId', 'options'],
    'is_async': True,
    'is_dynamic': False,
    
    # Analysis metadata
    'detected_by': 'ast_parser',
    'confidence': 0.95,
    'validation_status': 'verified'
}
```

---

## Implementation Strategy

### Option A: Category-Based (Recommended)

**Backend:**
```python
# backend/analyzer/models.py

class EdgeCategory(Enum):
    """High-level semantic categories for visualization."""
    EXECUTION = "execution"
    DEPENDENCY = "dependency"
    INHERITANCE = "inheritance"
    DATA_FLOW = "data_flow"
    COMPOSITION = "composition"
    REFERENCE = "reference"

class RelationshipType(Enum):
    """Specific relationship types (stored in metadata)."""
    # Execution
    CALLS = "calls"
    ASYNC_CALLS = "async_calls"
    AWAITS = "awaits"
    TRIGGERS = "triggers"
    HANDLES = "handles"
    SCHEDULES = "schedules"
    SPAWNS = "spawns"
    
    # Dependency
    IMPORTS = "imports"
    DEPENDS_ON = "depends_on"
    REQUIRES = "requires"
    LINKS_TO = "links_to"
    INCLUDES = "includes"
    REFERENCES = "references"
    
    # Inheritance
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    OVERRIDES = "overrides"
    
    # Data Flow
    READS = "reads"
    WRITES = "writes"
    UPDATES = "updates"
    TRANSFORMS = "transforms"
    
    # ... all other types
    
    # Generic fallback
    USES = "uses"

@dataclass
class Edge:
    source: str
    target: str
    category: EdgeCategory  # For visualization
    relationship_type: RelationshipType  # Specific type
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # For backward compatibility
    @property
    def type(self) -> EdgeCategory:
        return self.category
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'target': self.target,
            'type': self.category.value,  # For frontend
            'relationship_type': self.relationship_type.value,  # Specific
            'metadata': {
                **self.metadata,
                'category': self.category.value,
                'relationship_type': self.relationship_type.value
            }
        }
```

**Mapping Function:**
```python
def map_relationship_to_category(relationship_type: str) -> EdgeCategory:
    """Map specific relationship type to semantic category."""
    mapping = {
        # Execution
        'calls': EdgeCategory.EXECUTION,
        'async_calls': EdgeCategory.EXECUTION,
        'awaits': EdgeCategory.EXECUTION,
        'triggers': EdgeCategory.EXECUTION,
        'handles': EdgeCategory.EXECUTION,
        'schedules': EdgeCategory.EXECUTION,
        'spawns': EdgeCategory.EXECUTION,
        
        # Dependency
        'imports': EdgeCategory.DEPENDENCY,
        'depends_on': EdgeCategory.DEPENDENCY,
        'requires': EdgeCategory.DEPENDENCY,
        'links_to': EdgeCategory.DEPENDENCY,
        'includes': EdgeCategory.DEPENDENCY,
        'references': EdgeCategory.DEPENDENCY,
        
        # Inheritance
        'extends': EdgeCategory.INHERITANCE,
        'implements': EdgeCategory.INHERITANCE,
        'overrides': EdgeCategory.INHERITANCE,
        
        # Data Flow
        'reads': EdgeCategory.DATA_FLOW,
        'writes': EdgeCategory.DATA_FLOW,
        'updates': EdgeCategory.DATA_FLOW,
        'transforms': EdgeCategory.DATA_FLOW,
        
        # Composition
        'contains': EdgeCategory.COMPOSITION,
        'aggregates': EdgeCategory.COMPOSITION,
        'composes': EdgeCategory.COMPOSITION,
    }
    return mapping.get(relationship_type, EdgeCategory.REFERENCE)
```

**Edge Builder:**
```python
# backend/analyzer/edge_builder.py

def _build_edge_from_reference(self, reference: SymbolReference) -> Optional[Edge]:
    # Determine specific relationship type
    relationship_type = self._determine_relationship_type(reference)
    
    # Map to category for visualization
    category = map_relationship_to_category(relationship_type.value)
    
    # Build edge
    edge = Edge(
        source=source_node.id,
        target=target_node.id,
        category=category,
        relationship_type=relationship_type,
        metadata={
            'line_number': reference.line_number,
            'code_snippet': reference.context_snippet,
            'confidence': reference.confidence,
            # ... all other details
        }
    )
    return edge
```

---

## Frontend Visualization

**Simple by default, detailed on demand:**

```typescript
// dataService.ts - Style by category
function getEdgeStyle(edge: any) {
  const category = edge.metadata?.category || edge.type;
  const relationshipType = edge.metadata?.relationship_type;
  
  const categoryStyles = {
    'execution': { 
      stroke: '#f59e0b', 
      strokeWidth: 3, 
      strokeDasharray: '5,5',
      animated: true 
    },
    'dependency': { 
      stroke: '#3b82f6', 
      strokeWidth: 2, 
      opacity: 0.7 
    },
    'inheritance': { 
      stroke: '#8b5cf6', 
      strokeWidth: 3 
    },
    'data_flow': { 
      stroke: '#10b981', 
      strokeWidth: 2,
      strokeDasharray: '2,2' 
    },
    'composition': { 
      stroke: '#6366f1', 
      strokeWidth: 2 
    },
    'reference': { 
      stroke: '#94a3b8', 
      strokeWidth: 1.5, 
      opacity: 0.5 
    }
  };
  
  const style = categoryStyles[category] || categoryStyles['reference'];
  
  // Add label with specific type on hover
  return {
    ...style,
    label: relationshipType,  // Show specific type on edge
    labelStyle: { fontSize: 10, fill: style.stroke }
  };
}
```

**Filtering UI:**
```typescript
// Filter by category (simple)
const [visibleCategories, setVisibleCategories] = useState<Set<string>>(
  new Set(['execution', 'dependency'])
);

// Filter by specific type (advanced)
const [visibleTypes, setVisibleTypes] = useState<Set<string>>(
  new Set(['calls', 'imports', 'renders'])
);
```

---

## LLM Role: Discovery, Not Typing

### ❌ **Don't Use LLM For:**
- Determining edge types (too slow, unreliable)
- Classifying known relationships (AST can do this)

### ✅ **Use LLM For:**
- **Finding hidden relationships** (semantic similarity, patterns)
- **Understanding intent** (why this relationship exists)
- **Discovering architectural patterns** (MVC, Repository, etc.)

**Example:**
```python
# LLM discovers: "These two functions are semantically similar"
edge = Edge(
    category=EdgeCategory.REFERENCE,
    relationship_type=RelationshipType.SEMANTICALLY_SIMILAR,
    metadata={
        'detected_by': 'llm_semantic_analysis',
        'similarity_score': 0.87,
        'explanation': 'Both functions validate user input using similar patterns',
        'confidence': 0.75  # Lower confidence for LLM-discovered
    }
)
```

**But for AST-detectable relationships:**
```python
# AST detects: "Function A calls Function B"
edge = Edge(
    category=EdgeCategory.EXECUTION,
    relationship_type=RelationshipType.CALLS,
    metadata={
        'detected_by': 'ast_parser',
        'line_number': 42,
        'confidence': 0.95  # High confidence for AST
    }
)
```

---

## Configuration: User-Defined Types

**Allow users to define custom relationship types:**

```yaml
# .flowlens.yml
edge_types:
  custom:
    - name: "authenticates"
      category: "security"
      description: "Component authenticates user"
      style:
        color: "#ef4444"
        width: 2
        dashed: true
    
    - name: "publishes_to"
      category: "messaging"
      description: "Publishes message to queue"
      style:
        color: "#06b6d4"
        width: 2.5
```

**Backend loads and uses:**
```python
# Load user-defined types
user_config = load_user_config()
custom_types = user_config.get('edge_types', {}).get('custom', [])

# Map custom types
for custom_type in custom_types:
    RelationshipType[custom_type['name'].upper()] = custom_type['name']
    category_mapping[custom_type['name']] = custom_type['category']
```

---

## Migration Path

### Phase 1: Add Categories (No Breaking Changes)
```python
# Keep existing EdgeType enum
# Add EdgeCategory enum
# Add category property to Edge
# Map existing types to categories
```

### Phase 2: Enrich Metadata
```python
# Store relationship_type in metadata
# Keep type for backward compatibility
```

### Phase 3: Frontend Updates
```typescript
// Use category for styling
// Show relationship_type in tooltips
// Add filtering by both
```

---

## Answer to Your Questions

### Q: "Are we making things complex?"

**A: No, we're making it flexible without overwhelming users.**

- **Visualization:** Simple (6 categories)
- **Data:** Rich (50+ relationship types)
- **User experience:** Progressive disclosure (simple → detailed)

### Q: "Should we place LLM between each service?"

**A: No, use LLM strategically:**

- ✅ **LLM for:** Discovery, semantic understanding, pattern detection
- ❌ **LLM for:** Typing known relationships (AST is faster, more accurate)

**Better architecture:**
```
AST Parser → Edge Builder → Category Mapper → Visualization
     ↓
LLM Analyzer → Semantic Edges → Category Mapper → Visualization
```

Both feed into the same visualization system, but LLM only runs for enhanced analysis.

### Q: "Will it be good?"

**A: Yes, because:**

1. **Scalable:** Can add new relationship types without UI changes
2. **Understandable:** Categories keep visualization clean
3. **Detailed:** Full information available when needed
4. **Extensible:** Users can define custom types
5. **Performant:** LLM only for discovery, not every edge

---

## Recommended Implementation Order

1. **Week 1:** Add EdgeCategory enum, map existing types
2. **Week 2:** Store relationship_type in metadata
3. **Week 3:** Update frontend to use categories for styling
4. **Week 4:** Add filtering by category and type
5. **Week 5:** Add user-defined types support

**Start simple, add complexity gradually.**

---

## Example: Before vs After

### Before (Current):
```json
{
  "source": "A",
  "target": "B",
  "type": "depends_on",  // Too generic
  "metadata": {}  // Empty
}
```

### After (Proposed):
```json
{
  "source": "A",
  "target": "B",
  "type": "execution",  // Category for visualization
  "metadata": {
    "relationship_type": "calls",  // Specific type
    "relationship_subtype": "async_calls",
    "line_number": 42,
    "code_snippet": "await fetchData()",
    "category": "execution",
    "confidence": 0.95,
    "detected_by": "ast_parser"
  }
}
```

**Visualization:** Orange dashed animated line (execution category)
**Tooltip:** "calls (async_calls) at line 42: await fetchData()"

---

## Conclusion

**Don't hardcode, don't overcomplicate, don't LLM everything.**

**Do:**
- ✅ Use semantic categories for visualization
- ✅ Store specific types in metadata
- ✅ Use LLM for discovery, AST for typing
- ✅ Make it extensible via configuration

This gives you the best of all worlds: simple visualization, rich data, and flexibility.




