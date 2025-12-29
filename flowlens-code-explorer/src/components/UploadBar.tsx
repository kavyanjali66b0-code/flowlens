import React, { useState } from 'react';
import { Upload, Github, Loader2, FolderOpen, HardDrive } from 'lucide-react';
import { useFlowLensStore } from '../store/useFlowLensStore';
import { fetchAnalyzeRepo, loadProjectData, normalizeResponse } from '../services/dataService';
import { toast } from 'react-toastify';

const UploadBar: React.FC = () => {
  const [mode, setMode] = useState<'upload' | 'github' | 'folder'>('folder');
  const [repoUrl, setRepoUrl] = useState('');
  const [folderPath, setFolderPath] = useState('C:\\Users\\Vandh\\OneDrive\\Desktop\\Flow lens\\flowlens-code-explorer');
  const { setProjectData, setLoading, isLoading } = useFlowLensStore();

  const handleAnalyze = async () => {
    // Validation
    if (mode === 'github' && !repoUrl.trim()) {
      toast.error('Please enter a GitHub repository URL');
      return;
    }
    if (mode === 'folder' && !folderPath.trim()) {
      toast.error('Please enter a folder path');
      return;
    }

    setLoading(true);

    try {
      let data;

      if (mode === 'folder') {
        // Call backend /parse endpoint with folder path
        toast.info('Analyzing folder... This may take a moment.');

        const response = await fetch('http://localhost:5000/parse', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            folder_path: folderPath
          })
        });

        if (!response.ok) {
          throw new Error(`Backend returned ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        console.log('Backend response:', result);

        // Normalize the backend response
        data = normalizeResponse(result);

      } else if (mode === 'github') {
        toast.info('Analyzing repository...');
        data = await fetchAnalyzeRepo(repoUrl);
      } else {
        toast.info('Loading demo project...');
        data = await loadProjectData();
      }

      setProjectData(data);
      toast.success(`Successfully loaded ${data.projectMeta.name}!`);
    } catch (error: any) {
      console.error('Analysis failed:', error);
      toast.error(error.message || 'Failed to analyze project. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      toast.info('File upload functionality coming soon!');
      // For now, just load demo data
      handleAnalyze();
    }
  };

  return (
    <div className="flowlens-panel p-6 m-6 rounded-lg">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-center space-x-4 mb-6">
          <button
            onClick={() => setMode('folder')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-colors ${mode === 'folder' ? 'flowlens-button' : 'flowlens-button-ghost'
              }`}
          >
            <HardDrive className="h-5 w-5" />
            <span>Local Folder</span>
          </button>

          <button
            onClick={() => setMode('upload')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-colors ${mode === 'upload' ? 'flowlens-button' : 'flowlens-button-ghost'
              }`}
          >
            <Upload className="h-5 w-5" />
            <span>Upload Project</span>
          </button>

          <button
            onClick={() => setMode('github')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-colors ${mode === 'github' ? 'flowlens-button' : 'flowlens-button-ghost'
              }`}
          >
            <Github className="h-5 w-5" />
            <span>GitHub Repo</span>
          </button>
        </div>

        {mode === 'folder' ? (
          <div className="space-y-4">
            <div className="flex space-x-4">
              <input
                type="text"
                value={folderPath}
                onChange={(e) => setFolderPath(e.target.value)}
                placeholder="C:\Users\YourName\Desktop\your-project"
                className="flowlens-input flex-1 font-mono text-sm"
                disabled={isLoading}
              />
            </div>
            <p className="text-sm text-muted-foreground">
              Enter the absolute path to a local project folder to analyze
            </p>
          </div>
        ) : mode === 'upload' ? (
          <div className="space-y-4">
            <div className="border-2 border-dashed border-border rounded-lg p-8 text-center bg-accent/10 hover:bg-accent/20 transition-colors">
              <input
                type="file"
                id="file-upload"
                className="hidden"
                onChange={handleFileUpload}
                accept=".zip,.tar.gz"
                multiple
              />
              <label htmlFor="file-upload" className="cursor-pointer">
                <FolderOpen className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-lg font-medium text-foreground mb-2">
                  Drop project files here or click to browse
                </p>
                <p className="text-sm text-muted-foreground">
                  Supports .zip archives and project folders
                </p>
              </label>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex space-x-4">
              <input
                type="url"
                value={repoUrl}
                onChange={(e) => setRepoUrl(e.target.value)}
                placeholder="https://github.com/username/repository"
                className="flowlens-input flex-1"
                disabled={isLoading}
              />
            </div>
            <p className="text-sm text-muted-foreground">
              Enter a GitHub repository URL to analyze its code structure
            </p>
          </div>
        )}

        <div className="flex justify-center mt-6">
          <button
            onClick={handleAnalyze}
            disabled={isLoading}
            className="flowlens-button px-8 py-3 text-base"
          >
            {isLoading ? (
              <div className="flex items-center space-x-2">
                <Loader2 className="h-5 w-5 animate-spin" />
                <span>Analyzing...</span>
              </div>
            ) : (
              <div className="flex items-center space-x-2">
                <span>Analyze Project</span>
              </div>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default UploadBar;