export interface Job {
  id: number;
  title: string;
  company: string;
  location: string | null;
  country: string | null;
  work_mode: string;
  job_type: string;
  seniority: string;
  salary_min: number | null;
  salary_max: number | null;
  salary_currency: string | null;
  salary_period: string | null;
  posted_at: string | null;
  url: string | null;
  source: string;
  skills: string[];
  snippet: string;
  // Present on cards the assistant showed (absent when rebuilt from history or fetched directly):
  match_score?: number;
  why?: string[];
  watch_out?: string[];
}

export interface Profile {
  skills: string[];
  target_roles: string[];
  preferred_locations: string[];
  seniority: string | null;
  work_mode: string | null;
  job_type: string | null;
  min_salary: number | null;
}

export interface ConversationSummary {
  id: string;
  title: string | null;
  updated_at: string;
}

export interface HistoryMessage {
  id: number;
  role: "user" | "assistant";
  text: string;
  created_at: string;
}

export interface FinalMessage {
  type: "message";
  text: string;
  jobs: Job[];
  grounding: { cited: number[]; unverified: number[] };
  usage: { input_tokens: number; output_tokens: number };
  steps: number;
  stop_reason: string;
}

/** Everything the /chat endpoint can stream. */
export type ChatEvent =
  | { type: "conversation"; conversation_id: string }
  | { type: "tool_call"; id: string; name: string; input: Record<string, unknown> }
  | { type: "tool_result"; id: string; name: string; ok: boolean; summary: string }
  | FinalMessage
  | { type: "error"; message: string; retryable: boolean };
