import { useEffect, useRef, useState } from "react";
import type { ChatMessage } from "../lib/chatState";
import type { Job } from "../lib/types";
import { JobCard } from "./JobCard";
import { RichText } from "./RichText";

interface Props {
  messages: ChatMessage[];
  savedIds: Set<number>;
  dismissed: Set<number>;
  onToggleSave: (job: Job) => void;
  onDismiss: (job: Job) => void;
  onRetry: (assistantId: string) => void;
}

function Activity({ message }: { message: ChatMessage }) {
  if (!message.activity.length && message.status !== "streaming") return null;
  return (
    <ul className="activity" aria-label="What the assistant is doing">
      {message.activity.map((a) => (
        <li key={a.id} className={`activity__item activity__item--${a.status}`}>
          <span className="activity__icon" aria-hidden>
            {a.status === "running" ? <span className="spinner" /> : a.status === "done" ? "✓" : "✕"}
          </span>
          {a.label}
          {a.status === "failed" && <span className="muted"> (hit a snag, trying another way)</span>}
        </li>
      ))}
      {message.status === "streaming" && !message.activity.some((a) => a.status === "running") && (
        <li className="activity__item">
          <span className="typing" aria-label="Thinking">
            <i />
            <i />
            <i />
          </span>
        </li>
      )}
    </ul>
  );
}

function Bubble({ message, ...p }: { message: ChatMessage } & Omit<Props, "messages">) {
  const [flash, setFlash] = useState<number | null>(null);
  const jobs = message.jobs.filter((j) => !p.dismissed.has(j.id));
  const showJob = (id: number) => {
    const el = document.getElementById(`job-${message.id}-${id}`);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    setFlash(id);
    window.setTimeout(() => setFlash(null), 1600);
  };

  if (message.role === "user") {
    return (
      <div className="msg msg--user">
        <div className="bubble bubble--user">{message.text}</div>
      </div>
    );
  }
  return (
    <div className="msg msg--assistant">
      <div className="avatar" aria-hidden>
        P
      </div>
      <div className="msg__body">
        <Activity message={message} />
        {message.text && (
          <div className="bubble bubble--assistant">
            <RichText text={message.text} onJob={jobs.length ? showJob : undefined} />
          </div>
        )}
        {message.status === "error" && (
          <div className="error" role="alert">
            <span>{message.error}</span>
            {message.retryable && (
              <button type="button" className="btn btn--small" onClick={() => p.onRetry(message.id)}>
                Try again
              </button>
            )}
          </div>
        )}
        {message.unverified && message.unverified.length > 0 && (
          <p className="notice" role="note">
            I couldn't verify {message.unverified.map((i) => `#${i}`).join(", ")} against the job index, so treat it as unconfirmed.
          </p>
        )}
        {jobs.length > 0 && (
          <div className="cards">
            {jobs.map((job) => (
              <JobCard
                key={job.id}
                job={job}
                domId={`job-${message.id}-${job.id}`}
                highlighted={flash === job.id}
                saved={p.savedIds.has(job.id)}
                onToggleSave={p.onToggleSave}
                onDismiss={p.onDismiss}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export function MessageList({ messages, ...rest }: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  // Follow new content, but never yank the view if the reader scrolled up to look at something.
  useEffect(() => {
    if (stick.current) endRef.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  const onScroll = () => {
    const el = scroller.current;
    if (el) stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  };

  return (
    <div className="messages" ref={scroller} onScroll={onScroll} role="log" aria-live="polite" aria-relevant="additions text" aria-label="Conversation">
      <div className="messages__inner">
        {messages.map((m) => (
          <Bubble key={m.id} message={m} {...rest} />
        ))}
        <div ref={endRef} />
      </div>
    </div>
  );
}
