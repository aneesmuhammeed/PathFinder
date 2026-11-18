import { useCallback, useReducer, useRef } from "react";
import { api, ApiError, isRetryable, streamChat } from "../lib/api";
import { chatReducer, initialChat, isBusy, type ChatMessage } from "../lib/chatState";
import type { Job } from "../lib/types";

let counter = 0;
const nextId = (p: string) => `${p}-${Date.now().toString(36)}-${counter++}`;

const JOB_REF = /(?<![\w&/])#(\d{1,9})\b/g;

/** Past replies are stored as text only; rebuild job cards from the #ids they cite so reopened chats aren't bare. */
async function hydrateCards(text: string): Promise<Job[]> {
  const ids = [...new Set([...text.matchAll(JOB_REF)].map((m) => Number(m[1])))].slice(0, 6);
  const jobs = await Promise.allSettled(ids.map((id) => api.job(id)));
  return jobs.flatMap((r) => (r.status === "fulfilled" ? [r.value] : []));
}

export function useChat(userId: string, onTurnDone: () => void) {
  const [state, dispatch] = useReducer(chatReducer, initialChat);
  const abort = useRef<AbortController | null>(null);
  const conversationRef = useRef<string | null>(null);
  conversationRef.current = state.conversationId;

  const send = useCallback(
    async (text: string) => {
      const message = text.trim();
      if (!message) return;
      const assistantId = nextId("a");
      dispatch({ type: "send", userMessageId: nextId("u"), assistantId, text: message });
      const controller = new AbortController();
      abort.current = controller;
      try {
        await streamChat({
          userId,
          conversationId: conversationRef.current,
          message,
          signal: controller.signal,
          onEvent: (event) => dispatch({ type: "event", assistantId, event }),
        });
      } catch (e) {
        if (controller.signal.aborted) dispatch({ type: "stopped", assistantId });
        else {
          const msg = e instanceof ApiError ? e.message : "Couldn't reach the server. Check your connection and try again.";
          dispatch({ type: "failed", assistantId, message: msg, retryable: isRetryable(e) });
        }
      } finally {
        abort.current = null;
        onTurnDone();
      }
    },
    [userId, onTurnDone],
  );

  const stop = useCallback(() => abort.current?.abort(), []);

  /** Re-send the question behind a failed reply, replacing the failed attempt instead of stacking a duplicate. */
  const messagesRef = useRef(state.messages);
  messagesRef.current = state.messages;
  const retry = useCallback(
    (assistantId: string) => {
      const list = messagesRef.current;
      const i = list.findIndex((m) => m.id === assistantId);
      const question = i > 0 && list[i - 1].role === "user" ? list[i - 1].text : null;
      if (!question) return;
      dispatch({ type: "discard", assistantId });
      void send(question);
    },
    [send],
  );

  const newChat = useCallback(() => {
    abort.current?.abort();
    dispatch({ type: "reset" });
  }, []);

  const open = useCallback(
    async (conversationId: string) => {
      abort.current?.abort();
      const history = await api.messages(userId, conversationId);
      const messages: ChatMessage[] = await Promise.all(
        history.map(async (m) => ({
          id: `h-${m.id}`,
          role: m.role,
          text: m.text,
          jobs: m.role === "assistant" ? await hydrateCards(m.text) : [],
          activity: [],
          status: "done" as const,
        })),
      );
      dispatch({ type: "loaded", conversationId, messages });
    },
    [userId],
  );

  return { state, busy: isBusy(state), send, stop, retry, newChat, open };
}
