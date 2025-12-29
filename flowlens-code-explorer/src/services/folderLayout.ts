/**
 * Folder-Based Grid Layout (Inspired by Mockup)
 * 
 * Strategy:
 * 1. Group nodes by their folder path
 * 2. Position each folder group horizontally (side-by-side)
 * 3. Within each folder, arrange nodes in a vertical grid (6 max per column)
 * 4. Create folder background nodes with calculated bounds
 * 5. No parent-child relationships - folders are just visual backgrounds
 */

interface LayoutNode {
    id: string;
    type: string;
    file?: string;
    [key: string]: any;
}

interface FolderGroup {
    folderPath: string;
    nodes: string[];
    bounds: {
        x: number;
        y: number;
        width: number;
        height: number;
    };
}

/**
 * Create folder-based grid layout
 */
export function calculateFolderGridLayout(
    nodes: LayoutNode[],
    nodeWidth: number = 220, // Increased from 170 to account for actual file node widths
    nodeHeight: number = 65
): {
    positions: Map<string, { x: number; y: number }>;
    folderBounds: FolderGroup[];
} {

    // Configuration
    const GAP_X = 30;          // Horizontal gap between nodes
    const GAP_Y = 20;          // Vertical gap between nodes
    const FOLDER_PADDING_X = 40; // Left/right padding in folder (increased from 30)
    const FOLDER_PADDING_Y = 60; // Top padding for folder header (increased from 50)
    const FOLDER_GAP = 60;     // Gap between folders
    const MAX_NODES_PER_COLUMN = 6; // Max nodes in one column

    // Group nodes by folder path
    const folderGroups = new Map<string, LayoutNode[]>();

    nodes.forEach(node => {
        if (!node.file) {
            folderGroups.set('root', [...(folderGroups.get('root') || []), node]);
            return;
        }

        const filePath = node.file.replace(/\\/g, '/');
        const folderPath = filePath.substring(0, filePath.lastIndexOf('/')) || 'root';

        if (!folderGroups.has(folderPath)) {
            folderGroups.set(folderPath, []);
        }
        folderGroups.get(folderPath)!.push(node);
    });

    // Sort folders for consistent layout (prioritize common folders)
    const sortedFolders = Array.from(folderGroups.keys()).sort((a, b) => {
        // Priority order
        const priorities = [
            'src',
            'src/components',
            'src/pages',
            'src/services',
            'src/hooks',
            'src/utils',
            'src/store',
            'src/lib',
        ];

        const aIdx = priorities.findIndex(p => a.startsWith(p));
        const bIdx = priorities.findIndex(p => b.startsWith(p));

        if (aIdx !== -1 && bIdx !== -1) return aIdx - bIdx;
        if (aIdx !== -1) return -1;
        if (bIdx !== -1) return 1;
        return a.localeCompare(b);
    });

    // Position nodes and calculate folder bounds
    const positions = new Map<string, { x: number; y: number }>();
    const folderBounds: FolderGroup[] = [];
    let currentX = 0;

    sortedFolders.forEach(folderPath => {
        const folderNodes = folderGroups.get(folderPath)!;

        // Sort nodes within folder (by type, then name)
        folderNodes.sort((a, b) => {
            const typeOrder = { module: 0, component: 1, function: 2 };
            const aType = a.type as keyof typeof typeOrder;
            const bType = b.type as keyof typeof typeOrder;

            if (a.type !== b.type) {
                return (typeOrder[aType] || 3) - (typeOrder[bType] || 3);
            }
            return (a.name || a.id).localeCompare(b.name || b.id);
        });

        // Calculate grid dimensions for this folder
        const numColumns = Math.ceil(folderNodes.length / MAX_NODES_PER_COLUMN);
        let maxRow = 0;

        // Position each node in grid
        folderNodes.forEach((node, index) => {
            const col = Math.floor(index / MAX_NODES_PER_COLUMN);
            const row = index % MAX_NODES_PER_COLUMN;
            maxRow = Math.max(maxRow, row);

            positions.set(node.id, {
                x: currentX + FOLDER_PADDING_X + col * (nodeWidth + GAP_X),
                y: FOLDER_PADDING_Y + row * (nodeHeight + GAP_Y),
            });
        });

        // Calculate folder bounding box with proper padding
        // Width: account for all columns plus gaps and padding on both sides
        const folderWidth = numColumns * (nodeWidth + GAP_X) - GAP_X + FOLDER_PADDING_X * 2;

        // Height: account for all rows plus gaps, header padding at top, and bottom padding
        const folderHeight = (maxRow + 1) * (nodeHeight + GAP_Y) - GAP_Y + FOLDER_PADDING_Y + 50; // +50 for bottom padding

        folderBounds.push({
            folderPath,
            nodes: folderNodes.map(n => n.id),
            bounds: {
                x: currentX,
                y: 0,
                width: Math.max(folderWidth, 280), // Minimum width to accommodate header
                height: Math.max(folderHeight, 200), // Minimum height
            },
        });
        // Move to next folder position
        currentX += folderWidth + FOLDER_GAP;
    });

    return { positions, folderBounds };
}

/**
 * Helper to apply folder layout to nodes
 */
export function applyFolderLayout(nodes: any[]): {
    nodes: any[];
    folderNodes: any[];
} {
    const { positions, folderBounds } = calculateFolderGridLayout(nodes);

    // Update node positions
    const updatedNodes = nodes.map(node => ({
        ...node,
        position: positions.get(node.id) || node.position || { x: 0, y: 0 },
    }));

    // Create folder background nodes
    const folderNodes = folderBounds.map(folder => ({
        id: `folder-${folder.folderPath.replace(/[\/\\:]/g, '-')}`,
        type: 'folder',
        position: { x: folder.bounds.x, y: folder.bounds.y },
        data: {
            label: folder.folderPath,
            nodeCount: folder.nodes.length,
            folderPath: folder.folderPath,
        },
        style: {
            width: folder.bounds.width,
            height: folder.bounds.height,
            zIndex: -1,
        },
        selectable: false,
        draggable: false,
    }));

    return {
        nodes: updatedNodes,
        folderNodes,
    };
}
