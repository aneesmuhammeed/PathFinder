import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, isRetryable, streamChat } from "./api";
import type { ChatEvent } from "./types";

const enc = new TextEncoder();
const sse = (type: string, extra: object = {}) => `event: ${type}\ndata: ${JSON.stringify({ type, ...extra })}\n\n`;

function streamOf(chunks: Uint8Array[]): Response {
  let i = 0;
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (i < chunks.length) controller.enqueue(chunks[i++]);
      else controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

afterEach(() => vi.unstubAllGlobals());

async function collect(res: Response | Error, extra: { signal?: AbortSignal } = {}): Promise<ChatEvent[]> {
  vi.stubGlobal("fetch", vi.fn(async () => { if (res instanceof Error) throw res; return res; }));
  const events: ChatEvent[] = [];
  await streamChat({ userId: "u", conversationId: null, message: "hi", onEvent: (e) => events.push(e), ...extra });
  return events;
}

describe("isRetryable", () => {
  it("offers retry for transient failures only", () => {
    expect(isRetryable(new TypeError("Failed to fetch"))).toBe(true); // network
    expect(isRetryable(new ApiError(0, "closed early"))).toBe(true);
    expect(isRetryable(new ApiError(429, "Too many requests. Try again in 20 seconds."))).toBe(true);
    expect(isRetryable(new ApiError(503, "daily limit"))).toBe(true);
    expect(isRetryable(new ApiError(404, "user not found"))).toBe(false); // would fail identically
    expect(isRetryable(new ApiError(422, "message must not be blank"))).toBe(false);
  });
});

describe("streamChat", () => {
  it("delivers events as they arrive, even when chunks split an event and a multi-byte character", async () => {
    const wire = enc.encode(sse("conversation", { conversation_id: "c1" }) + sse("tool_call", { id: "t", name: "search_jobs", input: { q: "₹日本" } }) + sse("message", { text: "ok ₹", jobs: [] }));
    const cut = [7, 61, 62, 130];
    const chunks = cut.map((c, i) => wire.slice(i ? cut[i - 1] : 0, c)).concat(wire.slice(cut.at(-1)));
    const events = await collect(streamOf(chunks));
    expect(events.map((e) => e.type)).toEqual(["conversation", "tool_call", "message"]);
    expect(events[1]).toMatchObject({ type: "tool_call", input: { q: "₹日本" } });
    expect(events[2]).toMatchObject({ type: "message", text: "ok ₹" });
  });

  it("sends the right request", async () => {
    const fetchMock = vi.fn(async () => streamOf([enc.encode(sse("message", { text: "x", jobs: [] }))]));
    vi.stubGlobal("fetch", fetchMock);
    await streamChat({ userId: "u1", conversationId: "c1", message: "hello", onEvent: () => {} });
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/chat");
    expect(JSON.parse(init.body as string)).toEqual({ user_id: "u1", conversation_id: "c1", message: "hello", stream: true });
  });

  it("throws if the stream ends without a terminal event (never a silent half-answer)", async () => {
    await expect(collect(streamOf([enc.encode(sse("conversation", { conversation_id: "c" }))]))).rejects.toThrow(/closed before the assistant finished/);
  });

  it("accepts an error event as terminal", async () => {
    const events = await collect(streamOf([enc.encode(sse("error", { message: "boom", retryable: true }))]));
    expect(events.at(-1)).toMatchObject({ type: "error", message: "boom" });
  });

  it("turns HTTP errors into ApiError with the server's message", async () => {
    const bad = new Response(JSON.stringify({ detail: "user abc not found" }), { status: 404 });
    await expect(collect(bad)).rejects.toMatchObject({ status: 404, message: "user abc not found" });
    const validation = new Response(JSON.stringify({ detail: [{ msg: "message must not be blank" }] }), { status: 422 });
    await expect(collect(validation)).rejects.toBeInstanceOf(ApiError);
  });

  it("propagates aborts so the UI can show 'Stopped'", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(collect(new DOMException("aborted", "AbortError") as unknown as Error, { signal: controller.signal })).rejects.toThrow();
  });
});
