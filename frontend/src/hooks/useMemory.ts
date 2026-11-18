import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";
import type { ConversationSummary, Job, Profile } from "../lib/types";

/** Server-side state the sidebar shows: what the agent remembers, saved jobs, past conversations. */
export function useMemory(userId: string) {
  const [profile, setProfile] = useState<Profile | null>(null);
  const [saved, setSaved] = useState<Job[]>([]);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [dismissed, setDismissed] = useState<Set<number>>(new Set());

  const refresh = useCallback(async () => {
    const [p, s, c] = await Promise.allSettled([api.profile(userId), api.saved(userId), api.conversations(userId)]);
    if (p.status === "fulfilled") setProfile(p.value);
    if (s.status === "fulfilled") setSaved(s.value);
    if (c.status === "fulfilled") setConversations(c.value);
  }, [userId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const savedIds = new Set(saved.map((j) => j.id));

  /** Optimistic: flip locally, call the API, roll back on failure. */
  const toggleSave = useCallback(
    async (job: Job) => {
      const wasSaved = saved.some((j) => j.id === job.id);
      setSaved((cur) => (wasSaved ? cur.filter((j) => j.id !== job.id) : [job, ...cur]));
      try {
        if (wasSaved) await api.unfeedback(userId, job.id, "saved");
        else await api.feedback(userId, job.id, "saved");
      } catch {
        setSaved((cur) => (wasSaved ? [job, ...cur] : cur.filter((j) => j.id !== job.id)));
      }
    },
    [saved, userId],
  );

  const dismiss = useCallback(
    async (job: Job) => {
      setDismissed((cur) => new Set(cur).add(job.id));
      setSaved((cur) => cur.filter((j) => j.id !== job.id));
      try {
        await api.feedback(userId, job.id, "dismissed");
      } catch {
        setDismissed((cur) => {
          const next = new Set(cur);
          next.delete(job.id);
          return next;
        });
      }
    },
    [userId],
  );

  const removeSkill = useCallback(async (skill: string) => setProfile(await api.removeSkill(userId, skill)), [userId]);
  const clearField = useCallback(
    async (field: Parameters<typeof api.clearField>[1]) => setProfile(await api.clearField(userId, field)),
    [userId],
  );
  const forgetEverything = useCallback(async () => setProfile(await api.forgetEverything(userId)), [userId]);

  return { profile, saved, savedIds, dismissed, conversations, refresh, toggleSave, dismiss, removeSkill, clearField, forgetEverything };
}
