import React from 'react';
import { observer } from 'mobx-react-lite';
import { Turn } from '@/app/stores/sessionStore';
import { TurnBubble } from './details/TurnBubble';

interface ChatHistoryProps {
  messages: Turn[];
  messagesEndRef: React.RefObject<HTMLDivElement>;
}

export const ChatHistory = observer(({ messages, messagesEndRef }: ChatHistoryProps) => {
  return (
    <div className="flex-1 overflow-y-auto overscroll-contain px-3 py-4 space-y-6">
      {messages.map((turn) => (
        <TurnBubble
          key={turn.turn_id}
          turn={turn}
        />
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
});
ChatHistory.displayName = "ChatHistory";
