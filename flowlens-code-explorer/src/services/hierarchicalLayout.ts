/**
 * Hierarchical Force-Directed Layout for Code Architecture Visualization
 * 
 * This creates a meaningful layout by:
 * 1. Identifying central/hub nodes (files with many connections)
 * 2. Placing important nodes at the center
 * 3. Arranging connected nodes around them
 * 4. Using force-directed algorithm for natural spacing
 */

interface LayoutNode {
    id: string;
    x: number;
    y: number;
}

interface LayoutEdge {
    source: string;
    target: string;
}

/**
 * Calculate node importance based on connectivity
 */
function calculateImportance(nodes: any[], edges: LayoutEdge[]): Map<string, number> {
    const importance = new Map<string, number>();

    // Count incoming and outgoing edges for each node
    const inDegree = new Map<string, number>();
    const outDegree = new Map<string, number>();

    edges.forEach(edge => {
        inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1);
        outDegree.set(edge.source, (outDegree.get(edge.source) || 0) + 1);
    });

    // Calculate importance score
    nodes.forEach(node => {
        const incoming = inDegree.get(node.id) || 0;
        const outgoing = outDegree.get(node.id) || 0;

        // Nodes with many incoming edges are important (dependencies)
        // Nodes with many outgoing edges are also important (coordinators)
        const score = incoming * 2 + outgoing;
        importance.set(node.id, score);
    });

    return importance;
}

/**
 * Create hierarchical force-directed layout
 */
export function calculateHierarchicalLayout(
    nodes: any[],
    edges: LayoutEdge[],
    width: number = 2000,
    height: number = 1500
): Map<string, { x: number; y: number }> {

    if (nodes.length === 0) {
        return new Map();
    }

    const positions = new Map<string, { x: number; y: number }>();
    const importance = calculateImportance(nodes, edges);

    // Sort nodes by importance
    const sortedNodes = [...nodes].sort((a, b) => {
        const scoreA = importance.get(a.id) || 0;
        const scoreB = importance.get(b.id) || 0;
        return scoreB - scoreA;
    });

    // Build adjacency list
    const neighbors = new Map<string, Set<string>>();
    nodes.forEach(node => neighbors.set(node.id, new Set()));

    edges.forEach(edge => {
        neighbors.get(edge.source)?.add(edge.target);
        neighbors.get(edge.target)?.add(edge.source);
    });

    // Place most important nodes in a circle at the center
    const centerNodes = sortedNodes.slice(0, Math.min(10, Math.ceil(nodes.length * 0.2)));
    const centerRadius = 300;
    const centerX = width / 2;
    const centerY = height / 2;

    centerNodes.forEach((node, i) => {
        const angle = (i / centerNodes.length) * 2 * Math.PI;
        positions.set(node.id, {
            x: centerX + centerRadius * Math.cos(angle),
            y: centerY + centerRadius * Math.sin(angle)
        });
    });

    // Place remaining nodes in expanding circles based on their connections
    const placed = new Set(centerNodes.map(n => n.id));
    const layers: string[][] = [centerNodes.map(n => n.id)];

    // Build layers by connection distance from center
    while (placed.size < nodes.length) {
        const currentLayer = layers[layers.length - 1];
        const nextLayer: string[] = [];

        // Find nodes connected to current layer
        currentLayer.forEach(nodeId => {
            const nodeNeighbors = neighbors.get(nodeId) || new Set();
            nodeNeighbors.forEach(neighborId => {
                if (!placed.has(neighborId)) {
                    nextLayer.push(neighborId);
                    placed.add(neighborId);
                }
            });
        });

        // If no connected nodes found, add remaining nodes
        if (nextLayer.length === 0) {
            sortedNodes.forEach(node => {
                if (!placed.has(node.id)) {
                    nextLayer.push(node.id);
                    placed.add(node.id);
                }
            });
        }

        if (nextLayer.length > 0) {
            layers.push(nextLayer);
        } else {
            break;
        }
    }

    // Place nodes in concentric circles
    layers.forEach((layer, layerIndex) => {
        if (layerIndex === 0) return; // Skip center (already placed)

        const radius = centerRadius + (layerIndex * 400);
        const angleStep = (2 * Math.PI) / layer.length;

        layer.forEach((nodeId, i) => {
            const angle = i * angleStep;
            positions.set(nodeId, {
                x: centerX + radius * Math.cos(angle),
                y: centerY + radius * Math.sin(angle)
            });
        });
    });

    // Apply force-directed refinement for better spacing
    const iterations = 50;
    const repulsionStrength = 5000;
    const attractionStrength = 0.01;
    const damping = 0.8;

    for (let iter = 0; iter < iterations; iter++) {
        const forces = new Map<string, { x: number; y: number }>();

        // Initialize forces
        nodes.forEach(node => {
            forces.set(node.id, { x: 0, y: 0 });
        });

        // Repulsion between all nodes
        nodes.forEach(nodeA => {
            nodes.forEach(nodeB => {
                if (nodeA.id === nodeB.id) return;

                const posA = positions.get(nodeA.id);
                const posB = positions.get(nodeB.id);

                // Skip if positions don't exist
                if (!posA || !posB) return;

                const dx = posA.x - posB.x;
                const dy = posA.y - posB.y;
                const distSq = dx * dx + dy * dy + 0.01; // Avoid division by zero
                const dist = Math.sqrt(distSq);

                const force = repulsionStrength / distSq;
                const forceA = forces.get(nodeA.id)!;
                forceA.x += (dx / dist) * force;
                forceA.y += (dy / dist) * force;
            });
        });

        // Attraction along edges
        edges.forEach(edge => {
            const posSource = positions.get(edge.source);
            const posTarget = positions.get(edge.target);

            if (!posSource || !posTarget) return;

            const dx = posTarget.x - posSource.x;
            const dy = posTarget.y - posSource.y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            const force = dist * attractionStrength;

            const forceSource = forces.get(edge.source);
            const forceTarget = forces.get(edge.target);

            // Skip if forces don't exist
            if (!forceSource || !forceTarget) return;

            forceSource.x += (dx / dist) * force;
            forceSource.y += (dy / dist) * force;
            forceTarget.x -= (dx / dist) * force;
            forceTarget.y -= (dy / dist) * force;
        });

        // Apply forces with damping
        nodes.forEach(node => {
            const pos = positions.get(node.id);
            const force = forces.get(node.id);

            // Skip if position or force doesn't exist
            if (!pos || !force) return;

            pos.x += force.x * damping;
            pos.y += force.y * damping;

            // Constrain to bounds
            pos.x = Math.max(100, Math.min(width - 100, pos.x));
            pos.y = Math.max(100, Math.min(height - 100, pos.y));
        });
    }

    return positions;
}

/**
 * Helper to apply layout to nodes
 */
export function applyHierarchicalLayout(nodes: any[], edges: any[]): any[] {
    const layoutEdges: LayoutEdge[] = edges.map(e => ({
        source: e.source,
        target: e.target
    }));

    const positions = calculateHierarchicalLayout(nodes, layoutEdges);

    return nodes.map(node => ({
        ...node,
        position: positions.get(node.id) || { x: 0, y: 0 }
    }));
}
