import { parseSSE } from "./sse";
import type { ChatEvent, ConversationSummary, HistoryMessage, Job, Profile } from "./types";

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? "/api";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

/**
 * Worth offering "Try again"? Network failures, rate limits (429) and server errors (5xx) are transient.
 * Other 4xx (bad id, validation) will fail identically on retry, so don't invite the user to try.
 */
export function isRetryable(e: unknown): boolean {
  if (!(e instanceof ApiError)) return true;
  return e.status === 0 || e.status === 429 || e.status >= 500;
}

async function errorFrom(res: Response): Promise<ApiError> {
  let detail = `Request failed (${res.status})`;
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") detail = body.detail;
    else if (Array.isArray(body?.detail) && body.detail[0]?.msg) detail = body.detail[0].msg;
  } catch {
    /* non-JSON error body */
  }
  return new ApiError(res.status, detail);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) throw await errorFrom(res);
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

export const api = {
  health: () => request<{ status: string; llm: string }>("/health"),
  createUser: () => request<{ id: string }>("/users", { method: "POST", body: "{}" }),
  profile: (uid: string) => request<Profile>(`/users/${uid}/profile`),
  removeSkill: (uid: string, skill: string) =>
    request<Profile>(`/users/${uid}/profile`, { method: "PATCH", body: JSON.stringify({ remove_skills: [skill] }) }),
  clearField: (uid: string, field: "seniority" | "work_mode" | "job_type" | "min_salary") =>
    request<Profile>(`/users/${uid}/profile`, { method: "PATCH", body: JSON.stringify({ [field]: null }) }),
  forgetEverything: (uid: string) => request<Profile>(`/users/${uid}/profile`, { method: "DELETE" }),
  saved: (uid: string) => request<Job[]>(`/users/${uid}/saved`),
  feedback: (uid: string, jobId: number, action: "saved" | "dismissed" | "clicked") =>
    request<void>(`/users/${uid}/feedback`, { method: "POST", body: JSON.stringify({ job_id: jobId, action }) }),
  unfeedback: (uid: string, jobId: number, action: "saved" | "dismissed") =>
    request<void>(`/users/${uid}/feedback/${action}/${jobId}`, { method: "DELETE" }),
  conversations: (uid: string) => request<ConversationSummary[]>(`/users/${uid}/conversations`),
  messages: (uid: string, conversationId: string) =>
    request<HistoryMessage[]>(`/conversations/${conversationId}/messages?user_id=${uid}`),
  job: (id: number) => request<Job>(`/jobs/${id}`),
};

export interface StreamArgs {
  userId: string;
  conversationId: string | null;
  message: string;
  signal?: AbortSignal;
  onEvent: (event: ChatEvent) => void;
}

/**
 * POST /chat and dispatch each server-sent event as it arrives. (EventSource can't POST, so this reads the
 * response body stream directly.) Resolves when the stream ends; throws if it ended without a terminal event.
 */
export async function streamChat({ userId, conversationId, message, signal, onEvent }: StreamArgs): Promise<void> {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ user_id: userId, conversation_id: conversationId, message, stream: true }),
    signal,
  });
  if (!res.ok) throw await errorFrom(res);
  if (!res.body) throw new ApiError(0, "Streaming isn't supported by this browser.");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let terminal = false;
  for (;;) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const { events, rest } = parseSSE(done ? buffer + "\n\n" : buffer);
    buffer = rest;
    for (const ev of events) {
      if (ev.type === "message" || ev.type === "error") terminal = true;
      onEvent(ev);
    }
    if (done) break;
  }
  if (!terminal) throw new ApiError(0, "The connection closed before the assistant finished. Please try again.");
}
