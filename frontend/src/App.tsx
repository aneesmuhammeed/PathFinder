import { useState } from "react";
import { Composer } from "./components/Composer";
import { EmptyState } from "./components/EmptyState";
import { MessageList } from "./components/MessageList";
import { Sidebar } from "./components/Sidebar";
import { useChat } from "./hooks/useChat";
import { useMemory } from "./hooks/useMemory";
import { useUser } from "./hooks/useUser";

function Shell({ userId, demo }: { userId: string; demo: boolean }) {
  const memory = useMemory(userId);
  const chat = useChat(userId, memory.refresh);
  const [menu, setMenu] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const openConversation = async (id: string) => {
    setLoadError(null);
    try {
      await chat.open(id);
    } catch {
      setLoadError("Couldn't open that conversation.");
    }
  };

  const empty = chat.state.messages.length === 0;
  return (
    <div className="app">
      <Sidebar
        open={menu}
        profile={memory.profile}
        saved={memory.saved}
        conversations={memory.conversations}
        currentConversation={chat.state.conversationId}
        onClose={() => setMenu(false)}
        onNewChat={chat.newChat}
        onOpenConversation={openConversation}
        onRemoveSkill={memory.removeSkill}
        onClearField={memory.clearField}
        onForget={memory.forgetEverything}
        onUnsave={memory.toggleSave}
      />
      {menu && <div className="scrim" onClick={() => setMenu(false)} aria-hidden />}

      <main className="chat">
        <header className="topbar">
          <button type="button" className="btn btn--ghost menu-btn" onClick={() => setMenu(true)} aria-label="Open menu">
            ☰
          </button>
          <span className="topbar__title">Pathfinder</span>
          {demo && (
            <span className="pill" title="No AI key is configured: a rule-based demo assistant is answering. Search and memory are real.">
              Demo assistant
            </span>
          )}
        </header>

        {loadError && (
          <p className="error error--bar" role="alert">
            {loadError}
          </p>
        )}

        {empty ? (
          <div className="messages">
            <div className="messages__inner">
              <EmptyState demo={demo} onPick={chat.send} />
            </div>
          </div>
        ) : (
          <MessageList
            messages={chat.state.messages}
            savedIds={memory.savedIds}
            dismissed={memory.dismissed}
            onToggleSave={memory.toggleSave}
            onDismiss={memory.dismiss}
            onRetry={chat.retry}
          />
        )}

        <Composer busy={chat.busy} onSend={chat.send} onStop={chat.stop} focusKey={chat.state.conversationId ?? "new"} />
      </main>
    </div>
  );
}

export default function App() {
  const user = useUser();
  if (user.status === "loading") {
    return (
      <div className="splash" role="status" aria-live="polite">
        <span className="spinner spinner--lg" aria-hidden /> Connecting…
      </div>
    );
  }
  if (user.status === "error") {
    return (
      <div className="splash splash--error" role="alert">
        <h1>Can't reach the Pathfinder server</h1>
        <p className="muted">{user.message}</p>
        <p className="muted small">
          Start the backend with <code>python scripts/dev_stack.py</code> (from the <code>backend</code> folder), then try again.
        </p>
        <button type="button" className="btn btn--primary" onClick={user.retry}>
          Retry
        </button>
      </div>
    );
  }
  return <Shell userId={user.userId} demo={user.llm === "mock"} />;
}
