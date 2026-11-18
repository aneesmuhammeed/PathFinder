import { describe, expect, it } from "vitest";
import { parseSSE } from "./sse";

const ev = (type: string, extra: object = {}) => `event: ${type}\ndata: ${JSON.stringify({ type, ...extra })}\n\n`;

describe("parseSSE", () => {
  it("parses complete events and keeps the incomplete tail", () => {
    const { events, rest } = parseSSE(ev("conversation", { conversation_id: "c1" }) + 'event: tool_call\ndata: {"type":"tool_c');
    expect(events).toEqual([{ type: "conversation", conversation_id: "c1" }]);
    expect(rest).toBe('event: tool_call\ndata: {"type":"tool_c');
  });

  it("reassembles an event split at ANY byte boundary", () => {
    const wire = ev("tool_call", { id: "t1", name: "search_jobs", input: { q: "é ₹ 日本" } }) + ev("message", { text: "done", jobs: [] });
    for (let cut = 1; cut < wire.length; cut++) {
      const first = parseSSE(wire.slice(0, cut));
      const second = parseSSE(first.rest + wire.slice(cut));
      expect([...first.events, ...second.events].map((e) => e.type)).toEqual(["tool_call", "message"]);
      expect(second.rest).toBe("");
    }
  });

  it("handles CRLF line endings and multi-line data", () => {
    const { events } = parseSSE('event: error\r\ndata: {"type":"error",\r\ndata: "message":"x","retryable":true}\r\n\r\n');
    expect(events).toEqual([{ type: "error", message: "x", retryable: true }]);
  });

  it("ignores comments, keep-alives and malformed JSON without losing later events", () => {
    const { events } = parseSSE(": keep-alive\n\ndata: {not json}\n\n" + ev("conversation", { conversation_id: "ok" }));
    expect(events).toEqual([{ type: "conversation", conversation_id: "ok" }]);
  });

  it("returns nothing for an empty buffer", () => {
    expect(parseSSE("")).toEqual({ events: [], rest: "" });
  });
});
