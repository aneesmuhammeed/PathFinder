import { useState } from "react";
import { formatMinSalary, isProfileEmpty, jobTypeLabel, levelLabel, workModeLabel } from "../lib/format";
import type { ConversationSummary, Job, Profile } from "../lib/types";

interface Props {
  open: boolean;
  profile: Profile | null;
  saved: Job[];
  conversations: ConversationSummary[];
  currentConversation: string | null;
  onClose: () => void;
  onNewChat: () => void;
  onOpenConversation: (id: string) => void;
  onRemoveSkill: (skill: string) => void;
  onClearField: (field: "seniority" | "work_mode" | "job_type" | "min_salary") => void;
  onForget: () => void;
  onUnsave: (job: Job) => void;
}

function Chip({ children, onRemove, label }: { children: React.ReactNode; onRemove?: () => void; label?: string }) {
  return (
    <li className="mem-chip">
      {children}
      {onRemove && (
        <button type="button" className="mem-chip__x" onClick={onRemove} aria-label={`Forget ${label ?? "this"}`}>
          ×
        </button>
      )}
    </li>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mem-row">
      <dt>{label}</dt>
      <dd>
        <ul className="mem-chips">{children}</ul>
      </dd>
    </div>
  );
}

function Memory({ profile, onRemoveSkill, onClearField, onForget }: Pick<Props, "profile" | "onRemoveSkill" | "onClearField" | "onForget">) {
  const [confirming, setConfirming] = useState(false);
  if (isProfileEmpty(profile) || !profile) {
    return <p className="muted small">Nothing yet. Tell me your skills, level and where you'd like to work, and I'll remember it for next time.</p>;
  }
  return (
    <>
      <dl className="mem">
        {profile.skills.length > 0 && (
          <Row label="Skills">
            {profile.skills.map((s) => (
              <Chip key={s} label={s} onRemove={() => onRemoveSkill(s)}>
                {s}
              </Chip>
            ))}
          </Row>
        )}
        {profile.target_roles.length > 0 && <Row label="Looking for">{profile.target_roles.map((r) => <Chip key={r}>{r}</Chip>)}</Row>}
        {profile.seniority && (
          <Row label="Level">
            <Chip label="level" onRemove={() => onClearField("seniority")}>
              {levelLabel(profile.seniority)}
            </Chip>
          </Row>
        )}
        {profile.job_type && (
          <Row label="Wants">
            <Chip label="job type" onRemove={() => onClearField("job_type")}>
              {jobTypeLabel(profile.job_type)}
            </Chip>
          </Row>
        )}
        {profile.work_mode && (
          <Row label="Work mode">
            <Chip label="work mode" onRemove={() => onClearField("work_mode")}>
              {workModeLabel(profile.work_mode)}
            </Chip>
          </Row>
        )}
        {profile.preferred_locations.length > 0 && <Row label="Locations">{profile.preferred_locations.map((l) => <Chip key={l}>{l}</Chip>)}</Row>}
        {profile.min_salary ? (
          <Row label="Minimum pay">
            <Chip label="minimum pay" onRemove={() => onClearField("min_salary")}>
              {formatMinSalary(profile.min_salary)}
            </Chip>
          </Row>
        ) : null}
      </dl>
      {confirming ? (
        <div className="confirm" role="alertdialog" aria-label="Confirm forgetting everything">
          <span>Forget everything I know about you?</span>
          <button type="button" className="btn btn--small btn--danger" onClick={() => { setConfirming(false); onForget(); }}>
            Yes, forget
          </button>
          <button type="button" className="btn btn--small btn--ghost" onClick={() => setConfirming(false)}>
            Cancel
          </button>
        </div>
      ) : (
        <button type="button" className="linklike" onClick={() => setConfirming(true)}>
          Forget everything
        </button>
      )}
    </>
  );
}

export function Sidebar(p: Props) {
  return (
    <aside className={`sidebar${p.open ? " sidebar--open" : ""}`} aria-label="Your profile and history">
      <div className="sidebar__top">
        <div className="brand">
          <span className="brand__mark" aria-hidden>
            ➤
          </span>
          Pathfinder
        </div>
        <button type="button" className="btn btn--ghost sidebar__close" onClick={p.onClose} aria-label="Close menu">
          ✕
        </button>
      </div>

      <button type="button" className="btn btn--primary newchat" onClick={() => { p.onNewChat(); p.onClose(); }}>
        + New chat
      </button>

      <section aria-labelledby="mem-h">
        <h2 id="mem-h" className="side-h">
          What I remember about you
        </h2>
        <Memory profile={p.profile} onRemoveSkill={p.onRemoveSkill} onClearField={p.onClearField} onForget={p.onForget} />
        <p className="muted tiny">Stored against an anonymous ID in this browser. No account needed.</p>
      </section>

      <section aria-labelledby="saved-h">
        <h2 id="saved-h" className="side-h">
          Saved jobs <span className="count">{p.saved.length}</span>
        </h2>
        {p.saved.length === 0 ? (
          <p className="muted small">Jobs you save show up here.</p>
        ) : (
          <ul className="list">
            {p.saved.map((j) => (
              <li key={j.id} className="list__item">
                <div>
                  <strong>{j.title}</strong>
                  <span className="muted small"> {j.company}</span>
                </div>
                <button type="button" className="mem-chip__x" onClick={() => p.onUnsave(j)} aria-label={`Unsave ${j.title}`}>
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section aria-labelledby="hist-h">
        <h2 id="hist-h" className="side-h">
          Conversations
        </h2>
        {p.conversations.length === 0 ? (
          <p className="muted small">Your chats will be listed here.</p>
        ) : (
          <ul className="list">
            {p.conversations.map((c) => (
              <li key={c.id}>
                <button
                  type="button"
                  className={`conv${c.id === p.currentConversation ? " conv--active" : ""}`}
                  aria-current={c.id === p.currentConversation ? "true" : undefined}
                  onClick={() => { p.onOpenConversation(c.id); p.onClose(); }}
                >
                  {c.title || "Untitled chat"}
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </aside>
  );
}
