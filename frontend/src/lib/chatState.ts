import { toolLabel } from "./format";
import type { ChatEvent, Job } from "./types";

export interface Activity {
  id: string;
  label: string;
  status: "running" | "done" | "failed";
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  jobs: Job[];
  activity: Activity[];
  status: "streaming" | "done" | "error";
  error?: string;
  retryable?: boolean;
  unverified?: number[];
}

export interface ChatState {
  conversationId: string | null;
  messages: ChatMessage[];
}

export const initialChat: ChatState = { conversationId: null, messages: [] };

export type ChatAction =
  | { type: "reset" }
  | { type: "loaded"; conversationId: string; messages: ChatMessage[] }
  | { type: "send"; userMessageId: string; assistantId: string; text: string }
  | { type: "event"; assistantId: string; event: ChatEvent }
  | { type: "discard"; assistantId: string }
  | { type: "failed"; assistantId: string; message: string; retryable: boolean }
  | { type: "stopped"; assistantId: string };

const patch = (state: ChatState, id: string, fn: (m: ChatMessage) => ChatMessage): ChatState => ({
  ...state,
  messages: state.messages.map((m) => (m.id === id ? fn(m) : m)),
});

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "reset":
      return initialChat;
    case "loaded":
      return { conversationId: action.conversationId, messages: action.messages };
    case "send":
      return {
        ...state,
        messages: [
          ...state.messages,
          { id: action.userMessageId, role: "user", text: action.text, jobs: [], activity: [], status: "done" },
          { id: action.assistantId, role: "assistant", text: "", jobs: [], activity: [], status: "streaming" },
        ],
      };
    case "event": {
      const ev = action.event;
      switch (ev.type) {
        case "conversation":
          return { ...state, conversationId: ev.conversation_id };
        case "tool_call":
          return patch(state, action.assistantId, (m) => ({
            ...m,
            activity: [...m.activity, { id: ev.id, label: toolLabel(ev.name, ev.input), status: "running" }],
          }));
        case "tool_result":
          return patch(state, action.assistantId, (m) => ({
            ...m,
            activity: m.activity.map((a) => (a.id === ev.id ? { ...a, status: ev.ok ? "done" : "failed" } : a)),
          }));
        case "message":
          return patch(state, action.assistantId, (m) => ({
            ...m,
            text: ev.text,
            jobs: ev.jobs,
            status: "done",
            unverified: ev.grounding.unverified,
            activity: m.activity.map((a) => (a.status === "running" ? { ...a, status: "done" } : a)),
          }));
        case "error":
          return patch(state, action.assistantId, (m) => ({ ...m, status: "error", error: ev.message, retryable: ev.retryable }));
      }
      return state;
    }
    case "discard": {
      // Retrying replaces a failed attempt: drop the failed reply and the question that produced it.
      const i = state.messages.findIndex((m) => m.id === action.assistantId);
      if (i < 0) return state;
      const from = state.messages[i - 1]?.role === "user" ? i - 1 : i;
      return { ...state, messages: [...state.messages.slice(0, from), ...state.messages.slice(i + 1)] };
    }
    case "failed":
      return patch(state, action.assistantId, (m) => ({ ...m, status: "error", error: action.message, retryable: action.retryable }));
    case "stopped":
      return patch(state, action.assistantId, (m) => ({
        ...m,
        status: "error",
        error: "Stopped.",
        retryable: false,
        activity: m.activity.map((a) => (a.status === "running" ? { ...a, status: "failed" } : a)),
      }));
  }
}

export const isBusy = (s: ChatState) => s.messages.some((m) => m.status === "streaming");
