import React, { memo } from 'react';
import { Folder } from 'lucide-react';

interface FolderNodeData {
    label: string;
    nodeCount: number;
    folderPath: string;
}

interface FolderNodeProps {
    data: FolderNodeData;
    width?: number;
    height?: number;
}

const FolderNode: React.FC<FolderNodeProps> = ({ data, width = 250, height = 150 }) => {
    // Extract just the folder name (last part of path)
    const folderName = data.label.split(/[\\/]/).filter(Boolean).pop() || data.label;

    return (
        <div
            className="rounded-xl border-2 border-dashed border-muted-foreground/20 bg-muted/30"
            style={{
                width: `${width}px`,
                height: `${height}px`,
                position: 'relative',
            }}
        >
            {/* Header floats at top, doesn't reduce internal space */}
            <div
                className="absolute top-3 left-3 flex items-center gap-2 px-2 py-1.5 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm"
                style={{ zIndex: 1 }}
            >
                <Folder className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium text-muted-foreground truncate max-w-[180px]">
                    {folderName}
                </span>
                <span className="text-xs text-muted-foreground/60 ml-2 whitespace-nowrap">
                    {data.nodeCount} files
                </span>
            </div>
        </div>
    );
};

export default memo(FolderNode);
