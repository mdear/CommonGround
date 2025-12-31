import React from 'react';
import { observer } from 'mobx-react-lite';
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { ChatHistory } from './ChatHistory';
import { ChatInput } from './ChatInput';
import { Workspace } from './Workspace';
import { Turn } from '@/app/stores/sessionStore';
import { Node, NodeMouseHandler } from 'reactflow';
import { selectionStore } from '@/app/stores/selectionStore';

interface ChatLayoutProps {
  messages: Turn[];
  currentInput: string;
  onInputChange: (value: string) => void;
  onSendMessage: () => void;
  onKeyPress: (e: React.KeyboardEvent) => void;
  isLoading: boolean;
  isStreaming: boolean;
  onStopExecution: () => void;
  messagesEndRef: React.RefObject<HTMLDivElement>;
  runId: string;
  selectedNode: Node | null;
  isNodeDetailOpen: boolean;
  onNodeClick: NodeMouseHandler;
  onCloseNodeDetail: () => void;
}

export const ChatLayout = observer(function ChatLayout(props: ChatLayoutProps) {
  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <div className="flex-shrink-0 h-12 border-b bg-white flex items-center justify-between px-3">
        <div className="flex items-center gap-3">
          <SidebarTrigger />
          <div className="h-6 w-[1px] bg-gray-200" />
          <div className="flex items-center gap-2 text-sm">
            <span className="font-medium">{selectionStore.displayProjectName}</span>
            <span className="text-gray-400">&gt;</span>
            <span>{selectionStore.displayFileName}</span>
          </div>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        <ResizablePanelGroup direction="horizontal">
          <ResizablePanel 
            defaultSize={30} 
            minSize={25} 
            maxSize={40}
            className="flex flex-col h-[calc(100vh-46px)] border-r overflow-hidden"
          >
            <ChatHistory messages={props.messages} messagesEndRef={props.messagesEndRef} />
            <ChatInput
              currentInput={props.currentInput}
              onInputChange={props.onInputChange}
              onKeyPress={props.onKeyPress}
              onSendMessage={props.onSendMessage}
              isStreaming={props.isStreaming}
              isLoading={props.isLoading}
              onStopExecution={props.onStopExecution}
            />
          </ResizablePanel>

          <ResizableHandle withHandle />

          <ResizablePanel defaultSize={70} className="overflow-hidden">
            <Workspace
              runId={props.runId}
              selectedNode={props.selectedNode}
              isNodeDetailOpen={props.isNodeDetailOpen}
              onNodeClick={props.onNodeClick}
              onCloseNodeDetail={props.onCloseNodeDetail}
            />
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
    </div>
  );
});
