import { describe, expect, it } from "vitest";
import { chatReducer, initialChat, isBusy, type ChatState } from "./chatState";
import type { ChatEvent, FinalMessage, Job } from "./types";

const job = (id: number): Job => ({
  id, title: "Backend Intern", company: "Acme", location: "Pune", country: "India", work_mode: "hybrid", job_type: "internship",
  seniority: "intern", salary_min: null, salary_max: null, salary_currency: null, salary_period: null, posted_at: null,
  url: null, source: "seed", skills: ["python"], snippet: "",
});
const final = (over: Partial<FinalMessage> = {}): FinalMessage => ({
  type: "message", text: "Here you go: #1", jobs: [job(1)], grounding: { cited: [1], unverified: [] },
  usage: { input_tokens: 1, output_tokens: 1 }, steps: 2, stop_reason: "end_turn", ...over,
});

function run(events: ChatEvent[], start: ChatState = initialChat): ChatState {
  let s = chatReducer(start, { type: "send", userMessageId: "u1", assistantId: "a1", text: "find jobs" });
  for (const event of events) s = chatReducer(s, { type: "event", assistantId: "a1", event });
  return s;
}
const assistant = (s: ChatState) => s.messages.find((m) => m.id === "a1")!;

describe("chatReducer", () => {
  it("adds the user's message and a streaming assistant placeholder", () => {
    const s = run([]);
    expect(s.messages.map((m) => [m.role, m.status])).toEqual([["user", "done"], ["assistant", "streaming"]]);
    expect(isBusy(s)).toBe(true);
  });

  it("tracks tool activity live, then resolves it", () => {
    const calls: ChatEvent[] = [
      { type: "conversation", conversation_id: "c9" },
      { type: "tool_call", id: "t1", name: "update_profile", input: {} },
      { type: "tool_call", id: "t2", name: "search_jobs", input: { query: "react" } },
    ];
    let s = run(calls);
    expect(s.conversationId).toBe("c9");
    expect(assistant(s).activity.map((a) => [a.label, a.status])).toEqual([
      ["Remembering what you told me", "running"], ["Searching jobs for “react”", "running"],
    ]);
    s = chatReducer(s, { type: "event", assistantId: "a1", event: { type: "tool_result", id: "t1", name: "update_profile", ok: true, summary: "" } });
    s = chatReducer(s, { type: "event", assistantId: "a1", event: { type: "tool_result", id: "t2", name: "search_jobs", ok: false, summary: "" } });
    expect(assistant(s).activity.map((a) => a.status)).toEqual(["done", "failed"]);
  });

  it("completes with text, cards and a clean busy flag", () => {
    const s = run([{ type: "tool_call", id: "t1", name: "search_jobs", input: {} }, final()]);
    const a = assistant(s);
    expect(a.status).toBe("done");
    expect(a.text).toContain("#1");
    expect(a.jobs.map((j) => j.id)).toEqual([1]);
    expect(a.activity[0].status).toBe("done"); // nothing left spinning once the answer arrives
    expect(isBusy(s)).toBe(false);
  });

  it("carries unverified job references through for a warning", () => {
    expect(assistant(run([final({ grounding: { cited: [1, 9], unverified: [9] } })])).unverified).toEqual([9]);
  });

  it.each([
    ["a server error event", { type: "error", message: "rate limited", retryable: true } as ChatEvent],
  ])("surfaces %s with its retry hint", (_n, event) => {
    const a = assistant(run([event]));
    expect(a).toMatchObject({ status: "error", error: "rate limited", retryable: true });
  });

  it("marks a transport failure and a user stop distinctly", () => {
    let s = run([{ type: "tool_call", id: "t1", name: "search_jobs", input: {} }]);
    s = chatReducer(s, { type: "stopped", assistantId: "a1" });
    expect(assistant(s)).toMatchObject({ status: "error", error: "Stopped.", retryable: false });
    expect(assistant(s).activity[0].status).toBe("failed");

    const f = chatReducer(run([]), { type: "failed", assistantId: "a1", message: "offline", retryable: true });
    expect(assistant(f)).toMatchObject({ status: "error", error: "offline", retryable: true });
  });

  it("discard removes a failed reply together with the question behind it, nothing else", () => {
    let s = run([{ type: "error", message: "x", retryable: true }]);
    s = chatReducer(s, { type: "send", userMessageId: "u2", assistantId: "a2", text: "second" });
    const out = chatReducer(s, { type: "discard", assistantId: "a1" });
    expect(out.messages.map((m) => m.id)).toEqual(["u2", "a2"]);
    expect(chatReducer(s, { type: "discard", assistantId: "nope" })).toBe(s); // unknown id: no-op
  });

  it("continues an existing conversation and resets cleanly", () => {
    const loaded = chatReducer(initialChat, { type: "loaded", conversationId: "c1", messages: [] });
    const s = run([], loaded);
    expect(s.conversationId).toBe("c1");
    expect(chatReducer(s, { type: "reset" })).toEqual(initialChat);
  });

  it("never mutates previous state", () => {
    const before = run([]);
    const snapshot = JSON.stringify(before);
    chatReducer(before, { type: "event", assistantId: "a1", event: final() });
    expect(JSON.stringify(before)).toBe(snapshot);
  });
});
