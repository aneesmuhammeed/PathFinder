import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../lib/api";

const KEY = "pathfinder.userId";

// localStorage can throw (private mode, blocked storage): treat it as optional.
const read = () => {
  try {
    return localStorage.getItem(KEY);
  } catch {
    return null;
  }
};
const write = (v: string) => {
  try {
    localStorage.setItem(KEY, v);
  } catch {
    /* session-only identity */
  }
};

export type UserState =
  | { status: "loading" }
  | { status: "ready"; userId: string; llm: string }
  | { status: "error"; message: string };

/** Anonymous identity: a UUID the browser keeps. Verifies a stored id still exists server-side, else makes a new one. */
export function useUser(): UserState & { retry: () => void } {
  const [state, setState] = useState<UserState>({ status: "loading" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setState({ status: "loading" });
      try {
        const health = await api.health();
        if (cancelled) return; // a superseded run (React StrictMode, a quick retry) must not create or store anything
        let userId = read();
        if (userId) {
          try {
            await api.profile(userId);
          } catch (e) {
            if (e instanceof ApiError && e.status === 404) userId = null;
            else throw e;
          }
          if (cancelled) return;
        }
        if (!userId) {
          const created = await api.createUser();
          if (cancelled) return; // (an orphan user row is harmless; a stale id in storage is not)
          userId = created.id;
          write(userId);
        }
        setState({ status: "ready", userId, llm: health.llm });
      } catch (e) {
        if (!cancelled) setState({ status: "error", message: e instanceof Error ? e.message : "Could not reach the server." });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [attempt]);

  const retry = useCallback(() => setAttempt((a) => a + 1), []);
  return { ...state, retry };
}
