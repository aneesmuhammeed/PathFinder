import type { Job, Profile } from "./types";

const WORK_MODE: Record<string, string> = { remote: "Remote", hybrid: "Hybrid", onsite: "On-site" };
const JOB_TYPE: Record<string, string> = {
  full_time: "Full-time",
  internship: "Internship",
  contract: "Contract",
  part_time: "Part-time",
};
const LEVEL: Record<string, string> = { intern: "Intern", junior: "Junior", mid: "Mid-level", senior: "Senior", lead: "Lead" };

export const workModeLabel = (v: string) => WORK_MODE[v] ?? v;
export const jobTypeLabel = (v: string) => JOB_TYPE[v] ?? v;
export const levelLabel = (v: string) => LEVEL[v] ?? v;

const num = new Intl.NumberFormat("en-IN");

/** "₹15–25 LPA", "₹35,000–40,000/month", "USD 80,000–102,000/year", or null when undisclosed. */
export function formatSalary(j: Pick<Job, "salary_min" | "salary_max" | "salary_currency" | "salary_period">): string | null {
  const { salary_min: lo, salary_max: hi, salary_currency: cur, salary_period: period } = j;
  if (!lo || !hi || !cur) return null;
  if (cur === "INR") {
    if (period === "month") return `₹${num.format(lo)}–${num.format(hi)}/month`;
    return `₹${+(lo / 100000).toFixed(2)}–${+(hi / 100000).toFixed(2)} LPA`;
  }
  return `${cur} ${new Intl.NumberFormat("en-US").format(lo)}–${new Intl.NumberFormat("en-US").format(hi)}/${period ?? "year"}`;
}

/** Profile minimum salary is stored as annual INR. */
export const formatMinSalary = (inr: number) => `₹${+(inr / 100000).toFixed(2)} LPA`;

export function relativeDate(iso: string | null, now: Date = new Date()): string | null {
  if (!iso) return null;
  const days = Math.floor((now.getTime() - new Date(iso).getTime()) / 86_400_000);
  if (Number.isNaN(days)) return null;
  if (days <= 0) return "Posted today";
  if (days === 1) return "Posted yesterday";
  if (days < 30) return `Posted ${days} days ago`;
  const months = Math.floor(days / 30);
  return `Posted ${months} month${months > 1 ? "s" : ""} ago`;
}

const TOOL_LABELS: Record<string, (input: Record<string, unknown>) => string> = {
  search_jobs: (i) => (typeof i.query === "string" && i.query.trim() ? `Searching jobs for “${i.query.trim().slice(0, 60)}”` : "Finding jobs that fit your profile"),
  get_job: () => "Reading the job details",
  get_profile: () => "Checking what I know about you",
  update_profile: () => "Remembering what you told me",
  save_job: () => "Saving the job",
  dismiss_job: () => "Hiding the job",
  list_saved_jobs: () => "Looking at your saved jobs",
  compare_jobs: () => "Comparing the jobs",
  explain_match: () => "Checking how well it fits you",
};

export function toolLabel(name: string, input: Record<string, unknown>): string {
  return (TOOL_LABELS[name] ?? (() => `Working on it (${name})`))(input);
}

/** Short human summary of a profile for empty-state checks. */
export function isProfileEmpty(p: Profile | null): boolean {
  return !p || (!p.skills.length && !p.target_roles.length && !p.preferred_locations.length && !p.seniority && !p.work_mode && !p.job_type && !p.min_salary);
}
