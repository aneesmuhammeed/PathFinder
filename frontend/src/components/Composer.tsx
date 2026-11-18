import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";

const MAX = 4000;

interface Props {
  busy: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  /** Changing this value focuses the box (e.g. after "New chat"). */
  focusKey: string | null;
}

export function Composer({ busy, onSend, onStop, focusKey }: Props) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  // Grow with content up to ~8 lines. An empty box never uses a measured height (CSS min-height applies), so one
  // bad measurement taken during a transient layout can't leave it stuck tall; re-measure on resize too.
  useEffect(() => {
    const fit = () => {
      const el = ref.current;
      if (!el) return;
      if (!el.value) {
        el.style.height = "";
        el.style.overflowY = "hidden";
        return;
      }
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
      el.style.overflowY = el.scrollHeight > 200 ? "auto" : "hidden";
    };
    fit();
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, [text, busy]);

  useEffect(() => ref.current?.focus(), [focusKey]);

  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    const value = text.trim();
    if (!value || busy || value.length > MAX) return;
    onSend(value);
    setText("");
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  const over = text.length > MAX;
  return (
    <form className="composer" onSubmit={submit}>
      <label htmlFor="composer-input" className="sr-only">
        Message Pathfinder
      </label>
      <textarea
        id="composer-input"
        ref={ref}
        rows={1}
        value={text}
        placeholder="What are you looking for?"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        aria-invalid={over}
        aria-describedby="composer-hint"
      />
      {busy ? (
        <button type="button" className="send send--stop" onClick={onStop} aria-label="Stop generating">
          ■
        </button>
      ) : (
        <button type="submit" className="send" disabled={!text.trim() || over} aria-label="Send message">
          ↑
        </button>
      )}
      <p id="composer-hint" className={`composer__hint${over || text.length > MAX - 400 ? " composer__hint--warn" : ""}`}>
        {text.length > MAX - 400 ? `${text.length}/${MAX}` : "Enter to send · Shift+Enter for a new line"}
      </p>
    </form>
  );
}
