import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { StrictMode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { Job, Profile } from "./lib/types";

const enc = new TextEncoder();
const sse = (type: string, extra: object = {}) => enc.encode(`event: ${type}\ndata: ${JSON.stringify({ type, ...extra })}\n\n`);

const JOB: Job = {
  id: 144, title: "SDE Intern - Backend", company: "Alchemy Games", location: "Gurugram", country: "India", work_mode: "onsite",
  job_type: "internship", seniority: "intern", salary_min: 35000, salary_max: 40000, salary_currency: "INR", salary_period: "month",
  posted_at: null, url: "https://careers.example.com/x", source: "seed", skills: ["python", "sql", "fastapi"], snippet: "",
  match_score: 0.82, why: ["intern level matches yours", "in Gurugram"], watch_out: ["only 2 of 9 listed skills match yours (python, sql)"],
};
const EMPTY: Profile = { skills: [], target_roles: [], preferred_locations: [], seniority: null, work_mode: null, job_type: null, min_salary: null };

/** A fake Pathfinder API. `chat` hands the test a controller so it can push SSE events one at a time. */
function fakeServer(opts: { llm?: string; profile?: Profile } = {}) {
  const calls: { method: string; path: string; body: unknown }[] = [];
  let profile = opts.profile ?? EMPTY;
  let saved: Job[] = [];
  let chat: ReadableStreamDefaultController<Uint8Array> | null = null;
  let chatFails: { status: number; detail: string } | null = null;

  const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    const path = url.replace(/^\/api/, "");
    const method = init?.method ?? "GET";
    const body = init?.body ? JSON.parse(init.body as string) : undefined;
    calls.push({ method, path, body });
    if (path === "/health") return json({ status: "ok", llm: opts.llm ?? "mock" });
    if (path === "/users" && method === "POST") return json({ id: "u1" }, 201);
    if (path === "/users/u1/profile" && method === "GET") return json(profile);
    if (path === "/users/u1/profile" && method === "PATCH") {
      if (body.remove_skills) profile = { ...profile, skills: profile.skills.filter((s) => !body.remove_skills.includes(s)) };
      return json(profile);
    }
    if (path === "/users/u1/profile" && method === "DELETE") return json((profile = EMPTY));
    if (path === "/users/u1/saved") return json(saved);
    if (path === "/users/u1/conversations") return json([]);
    if (path === "/users/u1/feedback" && method === "POST") {
      if (body.action === "saved") saved = [JOB];
      return new Response(null, { status: 204 });
    }
    if (path.startsWith("/users/u1/feedback/saved/") && method === "DELETE") {
      saved = [];
      return new Response(null, { status: 204 });
    }
    if (path === "/chat") {
      if (chatFails) return json({ detail: chatFails.detail }, chatFails.status);
      return new Response(
        new ReadableStream<Uint8Array>({
          start(c) {
            chat = c;
            // Like a real browser: aborting the request errors the body stream.
            init?.signal?.addEventListener("abort", () => c.error(new DOMException("The operation was aborted.", "AbortError")));
          },
        }),
        { status: 200 },
      );
    }
    return json({ detail: "not found" }, 404);
  }));

  return {
    calls,
    setProfile: (p: Profile) => void (profile = p),
    failChat: (status = 500, detail = "boom") => void (chatFails = { status, detail }),
    recover: () => void (chatFails = null),
    push: (type: string, extra?: object) => chat!.enqueue(sse(type, extra)),
    end: () => chat!.close(),
    sentTo: (path: string) => calls.filter((c) => c.path === path),
  };
}

beforeEach(() => vi.useRealTimers());
afterEach(() => vi.unstubAllGlobals());

describe("App", () => {
  it("bootstraps an anonymous user, remembers it, and flags the demo assistant honestly", async () => {
    const server = fakeServer({ llm: "mock" });
    render(<App />);
    expect(await screen.findByText("Demo assistant")).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: /find your next role/i })).toBeInTheDocument();
    expect(localStorage.getItem("pathfinder.userId")).toBe("u1");
    expect(server.sentTo("/users")).toHaveLength(1);
  });

  it("under StrictMode (effects run twice) creates exactly one user and stores that same one", async () => {
    // Regression: both runs used to create a user and write its id, so the stored id could differ from the active one,
    // and the next page load would silently show a different (empty) account.
    const server = fakeServer();
    render(
      <StrictMode>
        <App />
      </StrictMode>,
    );
    await screen.findByRole("heading", { name: /find your next role/i });
    expect(server.sentTo("/users")).toHaveLength(1);
    expect(localStorage.getItem("pathfinder.userId")).toBe("u1");
  });

  it("does not show the demo banner when a real model is configured", async () => {
    fakeServer({ llm: "anthropic" });
    render(<App />);
    await screen.findByRole("heading", { name: /find your next role/i });
    expect(screen.queryByText("Demo assistant")).toBeNull();
  });

  it("reuses a stored user id, and replaces it if the server no longer knows it", async () => {
    localStorage.setItem("pathfinder.userId", "u1");
    const server = fakeServer();
    render(<App />);
    await screen.findByRole("heading", { name: /find your next role/i });
    expect(server.sentTo("/users")).toHaveLength(0); // reused

    vi.unstubAllGlobals();
    localStorage.setItem("pathfinder.userId", "stale");
    const server2 = fakeServer();
    render(<App />);
    await waitFor(() => expect(server2.sentTo("/users")).toHaveLength(1)); // 404 on the stale id → new user
    await waitFor(() => expect(localStorage.getItem("pathfinder.userId")).toBe("u1"));
  });

  it("shows a helpful screen (with the command to run) when the backend is down, and recovers on retry", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/can't reach the pathfinder server/i);
    expect(screen.getByText(/dev_stack\.py/)).toBeInTheDocument();
    vi.unstubAllGlobals();
    fakeServer();
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("heading", { name: /find your next role/i })).toBeInTheDocument();
  });

  it("streams a turn: live activity first, then the answer with an explained job card", async () => {
    const server = fakeServer();
    render(<App />);
    const box = await screen.findByLabelText("Message Pathfinder");
    await userEvent.type(box, "find me a backend internship{Enter}");

    // The user's message appears immediately and the input is locked while the assistant works.
    expect(await screen.findByText("find me a backend internship")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Stop generating" })).toBeInTheDocument();
    expect(server.sentTo("/chat")[0].body).toMatchObject({ user_id: "u1", conversation_id: null, message: "find me a backend internship", stream: true });

    server.push("conversation", { conversation_id: "c1" });
    server.push("tool_call", { id: "t1", name: "search_jobs", input: { query: "backend internship" } });
    expect(await screen.findByText("Searching jobs for “backend internship”")).toBeInTheDocument();

    server.push("tool_result", { id: "t1", name: "search_jobs", ok: true, summary: "1 job(s) found" });
    server.push("message", { text: "Top pick: #144 SDE Intern - Backend.\n- apply soon", jobs: [JOB], grounding: { cited: [144], unverified: [] }, usage: { input_tokens: 1, output_tokens: 1 }, steps: 2, stop_reason: "end_turn" });
    server.end();

    const card = await screen.findByRole("article", { name: "SDE Intern - Backend at Alchemy Games" });
    expect(within(card).getByText("82% match")).toBeInTheDocument();
    expect(within(card).getByText("₹35,000–40,000/month")).toBeInTheDocument();
    expect(within(card).getByText(/intern level matches yours/)).toBeInTheDocument(); // why
    expect(within(card).getByText(/only 2 of 9 listed skills/)).toBeInTheDocument(); // honest watch-out
    expect(within(card).getByText("Demo posting")).toBeInTheDocument(); // fictional seed data is labelled, no dead Apply link
    expect(within(card).queryByRole("link", { name: /apply/i })).toBeNull();
    await waitFor(() => expect(screen.getByRole("button", { name: "Send message" })).toBeInTheDocument()); // unlocked again
  });

  it("save and dismiss call the API, update the sidebar, and 'not interested' removes the card", async () => {
    const server = fakeServer();
    render(<App />);
    await userEvent.type(await screen.findByLabelText("Message Pathfinder"), "jobs{Enter}");
    server.push("message", { text: "Here: #144", jobs: [JOB], grounding: { cited: [144], unverified: [] }, usage: { input_tokens: 1, output_tokens: 1 }, steps: 1, stop_reason: "end_turn" });
    server.end();

    const save = await screen.findByRole("button", { name: /☆ Save/ });
    await userEvent.click(save);
    expect(await screen.findByRole("button", { name: /★ Saved/ })).toHaveAttribute("aria-pressed", "true");
    expect(server.sentTo("/users/u1/feedback")[0].body).toEqual({ job_id: 144, action: "saved" });
    const sidebar = screen.getByRole("complementary", { name: /profile and history/i });
    expect(within(sidebar).getByText("SDE Intern - Backend")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /★ Saved/ })); // toggle off
    await waitFor(() => expect(server.calls.some((c) => c.method === "DELETE" && c.path === "/users/u1/feedback/saved/144")).toBe(true));
    await waitFor(() => expect(within(sidebar).queryByText("SDE Intern - Backend")).toBeNull());

    await userEvent.click(screen.getByRole("button", { name: "Not interested" }));
    await waitFor(() => expect(screen.queryByRole("article")).toBeNull());
    expect(server.sentTo("/users/u1/feedback").at(-1)?.body).toEqual({ job_id: 144, action: "dismissed" });
  });

  it("rolls back an optimistic save when the server rejects it", async () => {
    const server = fakeServer();
    render(<App />);
    await userEvent.type(await screen.findByLabelText("Message Pathfinder"), "jobs{Enter}");
    server.push("message", { text: "#144", jobs: [JOB], grounding: { cited: [144], unverified: [] }, usage: { input_tokens: 1, output_tokens: 1 }, steps: 1, stop_reason: "end_turn" });
    server.end();
    const original = globalThis.fetch;
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) =>
      url.endsWith("/feedback") ? new Response(JSON.stringify({ detail: "no" }), { status: 500 }) : original(url, init)));
    await userEvent.click(await screen.findByRole("button", { name: /☆ Save/ }));
    await waitFor(() => expect(screen.getByRole("button", { name: /☆ Save/ })).toHaveAttribute("aria-pressed", "false"));
  });

  it("shows a server error with a working 'Try again' that resends the same message", async () => {
    const server = fakeServer();
    render(<App />);
    await userEvent.type(await screen.findByLabelText("Message Pathfinder"), "hello{Enter}");
    server.push("error", { message: "The AI service is rate-limited right now. Please retry shortly.", retryable: true });
    server.end();
    expect(await screen.findByRole("alert")).toHaveTextContent(/rate-limited/);
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    await waitFor(() => expect(server.sentTo("/chat")).toHaveLength(2));
    expect(server.sentTo("/chat")[1].body).toMatchObject({ message: "hello" });
  });

  it("surfaces a transport failure instead of hanging", async () => {
    const server = fakeServer();
    server.failChat();
    render(<App />);
    await userEvent.type(await screen.findByLabelText("Message Pathfinder"), "hello{Enter}");
    expect(await screen.findByRole("alert")).toHaveTextContent("boom");
  });

  it("a rate-limited turn shows the server's friendly wait time and lets you retry; a hard 4xx does not", async () => {
    const server = fakeServer();
    server.failChat(429, "You're sending messages too quickly. Try again in 12 seconds.");
    render(<App />);
    await userEvent.type(await screen.findByLabelText("Message Pathfinder"), "hello{Enter}");
    expect(await screen.findByRole("alert")).toHaveTextContent("Try again in 12 seconds");
    server.recover();
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    await waitFor(() => expect(server.sentTo("/chat")).toHaveLength(2));
    server.push("message", { text: "Welcome back", jobs: [], grounding: { cited: [], unverified: [] }, usage: { input_tokens: 1, output_tokens: 1 }, steps: 1, stop_reason: "end_turn" });
    server.end();
    expect(await screen.findByText("Welcome back")).toBeInTheDocument();
    expect(screen.getAllByText("hello")).toHaveLength(1); // the retry REPLACED the failed attempt
    expect(screen.queryByRole("alert")).toBeNull();

    server.failChat(404, "conversation not found");
    await userEvent.type(screen.getByLabelText("Message Pathfinder"), "again{Enter}");
    expect(await screen.findByText("conversation not found")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Try again" })).toBeNull(); // retrying can't help
  });

  it("Stop cancels the stream and unlocks the input", async () => {
    const server = fakeServer();
    render(<App />);
    await userEvent.type(await screen.findByLabelText("Message Pathfinder"), "slow request{Enter}");
    server.push("tool_call", { id: "t1", name: "search_jobs", input: {} });
    await userEvent.click(await screen.findByRole("button", { name: "Stop generating" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Stopped.");
    expect(screen.getByRole("button", { name: "Send message" })).toBeInTheDocument();
  });

  it("Shift+Enter makes a new line instead of sending; blank messages can't be sent", async () => {
    const server = fakeServer();
    render(<App />);
    const box = await screen.findByLabelText("Message Pathfinder");
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled();
    await userEvent.type(box, "line one{Shift>}{Enter}{/Shift}line two");
    expect(box).toHaveValue("line one\nline two");
    expect(server.sentTo("/chat")).toHaveLength(0);
  });

  it("the message box returns to its natural height once emptied (no stale measured height)", async () => {
    fakeServer();
    render(<App />);
    const box = await screen.findByLabelText("Message Pathfinder");
    await userEvent.type(box, "a{Shift>}{Enter}{/Shift}b{Shift>}{Enter}{/Shift}c");
    expect((box as HTMLTextAreaElement).style.height).not.toBe(""); // grows with content
    await userEvent.clear(box);
    expect((box as HTMLTextAreaElement).style.height).toBe(""); // CSS min-height governs an empty box again
  });

  it("memory panel: shows what's remembered, lets you forget one skill or everything", async () => {
    const server = fakeServer({ profile: { ...EMPTY, skills: ["python", "sql"], seniority: "intern", min_salary: 800000 } });
    render(<App />);
    const panel = await screen.findByRole("heading", { name: /what i remember/i });
    expect(panel).toBeInTheDocument();
    expect(await screen.findByText("python")).toBeInTheDocument();
    expect(screen.getByText("Intern")).toBeInTheDocument();
    expect(screen.getByText("₹8 LPA")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Forget python" }));
    await waitFor(() => expect(screen.queryByText("python")).toBeNull());
    expect(server.calls.find((c) => c.method === "PATCH")?.body).toEqual({ remove_skills: ["python"] });

    await userEvent.click(screen.getByRole("button", { name: "Forget everything" }));
    await userEvent.click(screen.getByRole("button", { name: "Yes, forget" }));
    expect(await screen.findByText(/nothing yet/i)).toBeInTheDocument();
    expect(server.calls.some((c) => c.method === "DELETE" && c.path === "/users/u1/profile")).toBe(true);
  });

  it("clicking a suggested prompt sends it", async () => {
    const server = fakeServer();
    render(<App />);
    await userEvent.click(await screen.findByRole("button", { name: /show me remote data engineer roles/i }));
    await waitFor(() => expect(server.sentTo("/chat")).toHaveLength(1));
    expect(server.sentTo("/chat")[0].body).toMatchObject({ message: "Show me remote data engineer roles." });
  });
});
