import { formatSalary, jobTypeLabel, levelLabel, relativeDate, workModeLabel } from "../lib/format";
import type { Job } from "../lib/types";

interface Props {
  job: Job;
  saved: boolean;
  highlighted?: boolean;
  domId?: string;
  onToggleSave: (job: Job) => void;
  onDismiss: (job: Job) => void;
}

/** Seed jobs point at a placeholder domain: say so instead of offering a dead "Apply" link. */
const isDemoPosting = (job: Job) => job.source === "seed" || !job.url || /(^|\.)example\.com/.test(new URL(job.url, "http://x").hostname);

const ATTRIBUTION: Record<string, string> = { remotive: "via Remotive", arbeitnow: "via Arbeitnow" };

export function JobCard({ job, saved, highlighted, domId, onToggleSave, onDismiss }: Props) {
  const salary = formatSalary(job);
  const posted = relativeDate(job.posted_at);
  const score = job.match_score !== undefined ? Math.round(job.match_score * 100) : null;
  const shownSkills = job.skills.slice(0, 8);
  const demo = isDemoPosting(job);

  return (
    <article id={domId} className={`job${highlighted ? " job--flash" : ""}`} aria-label={`${job.title} at ${job.company}`}>
      <header className="job__head">
        <div>
          <h3 className="job__title">
            <span className="job__id" title="Job reference">
              #{job.id}
            </span>{" "}
            {job.title}
          </h3>
          <p className="job__company">
            {job.company}
            {job.location ? <span className="muted"> · {job.location}</span> : null}
          </p>
        </div>
        {score !== null && (
          <span className={`score score--${score >= 75 ? "high" : score >= 50 ? "mid" : "low"}`} title="How well this fits your saved profile and search">
            {score}% match
          </span>
        )}
      </header>

      <ul className="chips" aria-label="Job details">
        <li className="chip chip--strong">{levelLabel(job.seniority)}</li>
        <li className="chip">{jobTypeLabel(job.job_type)}</li>
        <li className="chip">{workModeLabel(job.work_mode)}</li>
        {salary && <li className="chip chip--money">{salary}</li>}
        {posted && <li className="chip chip--quiet">{posted}</li>}
      </ul>

      {(job.why?.length || job.watch_out?.length) ? (
        <ul className="reasons" aria-label="Why this job was suggested">
          {job.why?.map((w) => (
            <li key={w} className="reason reason--good">
              <span aria-hidden>✓</span> {w}
            </li>
          ))}
          {job.watch_out?.map((w) => (
            <li key={w} className="reason reason--warn">
              <span aria-hidden>!</span> {w}
            </li>
          ))}
        </ul>
      ) : null}

      {shownSkills.length > 0 && (
        <p className="skills" aria-label="Skills">
          {shownSkills.map((s) => (
            <span key={s} className="skill">
              {s}
            </span>
          ))}
          {job.skills.length > shownSkills.length && <span className="skill skill--more">+{job.skills.length - shownSkills.length}</span>}
        </p>
      )}

      <footer className="job__actions">
        <button type="button" className={`btn${saved ? " btn--on" : ""}`} aria-pressed={saved} onClick={() => onToggleSave(job)}>
          {saved ? "★ Saved" : "☆ Save"}
        </button>
        <button type="button" className="btn btn--ghost" onClick={() => onDismiss(job)} title="Hide this and show fewer like it">
          Not interested
        </button>
        {demo ? (
          <span className="demo" title="This listing comes from the built-in sample data, so there is nothing to apply to.">
            Demo posting
          </span>
        ) : (
          <a className="btn btn--link" href={job.url!} target="_blank" rel="noopener noreferrer">
            Apply ↗
          </a>
        )}
        {ATTRIBUTION[job.source] && <span className="muted attribution">{ATTRIBUTION[job.source]}</span>}
      </footer>
    </article>
  );
}
