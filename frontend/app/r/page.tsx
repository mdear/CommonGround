'use client';

import { useSearchParams } from 'next/navigation';
import { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import { reaction, runInAction } from 'mobx';
import { observer } from 'mobx-react-lite';
import { useLocalObservable } from 'mobx-react-lite';
import { selectionStore } from '@/app/stores/selectionStore';
import { sessionStore } from '@/app/stores/sessionStore';
import { ChatLayout } from '@/app/chat/components/ChatLayout';
import { projectStore } from '@/app/stores/projectStore';
import LoadingSpinner from '@/components/layout/LoadingSpinner';
import { Node, NodeMouseHandler } from 'reactflow';

// ====================================================================
// 1. Move all existing logic into this new Loader component.
// ====================================================================
const RunPageLoader = observer(() => {
  const searchParams = useSearchParams();
  const runId = searchParams.get('id');

  // If there's no ID in the URL, redirect or show an error.
  // Note: Since this component only renders on the client, we can use `window` directly.
  useEffect(() => {
    if (!runId) {
      window.location.href = '/';
    }
  }, [runId]);

  const { projects, loading: projectsLoading } = projectStore;
  const [isInitialized, setIsInitialized] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [isNodeDetailOpen, setIsNodeDetailOpen] = useState(false);

  const store = useLocalObservable(() => ({
    currentInput: '',
    isLoading: false,
    setCurrentInput(text: string) { this.currentInput = text; },
    setIsLoading(loading: boolean) { this.isLoading = loading; }
  }));

  // When runId changes, reset related state
  useEffect(() => {
    if (runId) {
      setIsInitialized(false);
      runInAction(() => {
        sessionStore.error = null;
        sessionStore.isResuming = false;
      });
    }
  }, [runId]);

  // Initialize run state
  useEffect(() => {
    if (runId && !isInitialized && !projectsLoading && projects.length > 0) {
      const foundFile = findRunInProjects(runId, projects);

      if (foundFile) {
        selectionStore.setSelectedFile(foundFile);
      } else {
        selectionStore.setSelectedFile({
          runId,
          filename: 'Untitled',
          projectId: 'default',
          projectName: 'Default Project'
        });
      }

      setIsInitialized(true);
    }
  }, [runId, projects, isInitialized, projectsLoading]);

  const chatHistoryTurnsForRun = useMemo(() =>
    sessionStore.chatHistoryTurns.filter(t => t.run_id === runId),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [sessionStore.chatHistoryTurns, runId]
  );

  // Resume session
  useEffect(() => {
    const selectedFile = selectionStore.selectedFile;
    if (sessionStore.isConnected && selectedFile && selectedFile.runId === runId && chatHistoryTurnsForRun.length === 0) {
      console.log(`WebSocket connected, requesting to resume run: ${runId} from project: ${selectedFile.projectId}`);
      sessionStore.resumeRun(runId, selectedFile.projectId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, selectionStore.selectedFile, sessionStore.isConnected, chatHistoryTurnsForRun]);

  const lastMessageContent = chatHistoryTurnsForRun.length > 0 ? JSON.stringify(chatHistoryTurnsForRun[chatHistoryTurnsForRun.length - 1].llm_interaction?.final_response?.content) : '';

  // When a run finishes, ensure the input is unlocked.
  useEffect(() => {
    if (!runId) return;

    const disposer = reaction(
      () => sessionStore.getRunStatus(runId).isRunning,
      (isRunning, prevIsRunning) => {
        if (prevIsRunning && !isRunning) {
          store.setIsLoading(false);
        }
      }
    );

    return () => {
      disposer();
    };
  }, [runId, store]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistoryTurnsForRun.length, lastMessageContent]);

  const sendMessage = useCallback(async () => {
    if (!store.currentInput.trim() || store.isLoading || !runId) return;
    const messageToSend = store.currentInput;
    store.setCurrentInput('');
    try {
        store.setIsLoading(true);
        await sessionStore.sendMessage(messageToSend, runId);
    } catch (error) {
        console.error('Failed to send message:', error);
        if (error instanceof Error) {
          sessionStore.error = error.message;
        } else {
          sessionStore.error = 'An unknown error occurred.';
        }
        // If sending the message itself fails, also unlock the input field
        store.setIsLoading(false);
    }
  }, [store, runId]);

  const handleKeyPress = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }, [sendMessage]);

  const handleNodeClick: NodeMouseHandler = useCallback((event, node) => {
    setSelectedNode(node);
    setIsNodeDetailOpen(true);
  }, []);

  const closeNodeDetail = () => {
    setIsNodeDetailOpen(false);
    setSelectedNode(null);
  };

  // 渲染逻辑
  if (!runId) {
    return (
      <div className="h-screen flex items-center justify-center bg-background">
        <LoadingSpinner />
        <p className="ml-2">Redirecting...</p>
      </div>
    );
  }

  if (projectsLoading || !isInitialized) {
    return (
      <div className="h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <LoadingSpinner />
          <p className="mt-4 text-gray-500">
            {projectsLoading ? 'Loading projects...' : 'Initializing...'}
          </p>
        </div>
      </div>
    );
  }

  if (!sessionStore.isConnected) {
    return (
      <div className="h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <LoadingSpinner />
          <p className="mt-4 text-gray-500">Connecting to server...</p>
        </div>
      </div>
    );
  }

  if (sessionStore.isResuming && chatHistoryTurnsForRun.length === 0) {
    return (
      <div className="h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <LoadingSpinner />
          <p className="mt-4 text-gray-500">Loading conversation...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen">
      <ChatLayout
        messages={chatHistoryTurnsForRun}
        currentInput={store.currentInput}
        onInputChange={(val) => store.setCurrentInput(val)}
        onSendMessage={sendMessage}
        onKeyPress={handleKeyPress}
        isLoading={store.isLoading}
        isStreaming={sessionStore.getRunStatus(runId).isStreamStarted}
        onStopExecution={() => sessionStore.stopExecution(runId)}
        messagesEndRef={messagesEndRef}
        runId={runId}
        selectedNode={selectedNode}
        isNodeDetailOpen={isNodeDetailOpen}
        onNodeClick={handleNodeClick}
        onCloseNodeDetail={closeNodeDetail}
      />
    </div>
  );
});

// ====================================================================
// 2. The original Page component is now very simple.
// ====================================================================
export default function RunPage() {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  // Before the client environment is fully ready, just show a loading indicator.
  if (!isClient) {
    return (
      <div className="h-screen flex items-center justify-center bg-background">
        <LoadingSpinner />
      </div>
    );
  }

  // Once the client is ready, render the actual page logic component.
  return <RunPageLoader />;
}

// Helper function remains unchanged
function findRunInProjects(runId: string, projects: { project: { project_id: string; name: string }; runs: { meta?: { run_id?: string; description?: string }; filename?: string }[] }[]) {
  for (const projectData of projects) {
    for (const run of projectData.runs || []) {
      if (run.meta?.run_id === runId) {
        return {
          runId,
          filename: run.filename || run.meta?.description || 'Untitled',
          projectId: projectData.project.project_id,
          projectName: projectData.project.name
        };
      }
    }
  }
  return null;
}
