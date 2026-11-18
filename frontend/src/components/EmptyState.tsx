const PROMPTS = [
  "I'm a final-year student. I know Python, SQL and FastAPI. Find me a backend internship in Bengaluru or Gurugram.",
  "Show me remote data engineer roles.",
  "I have 5 years of experience in frontend development and want a remote role. What's out there?",
  "What do you know about me?",
];

export function EmptyState({ demo, onPick }: { demo: boolean; onPick: (text: string) => void }) {
  return (
    <div className="empty">
      <div className="empty__mark" aria-hidden>
        ➤
      </div>
      <h1>Find your next role by talking about it</h1>
      <p className="muted">
        Tell me what you know and what you want. I search a live jobs index, rank results for <em>you</em>, explain every match, and remember your preferences.
      </p>
      <ul className="prompts" aria-label="Try one of these">
        {PROMPTS.map((p) => (
          <li key={p}>
            <button type="button" className="prompt" onClick={() => onPick(p)}>
              {p}
            </button>
          </li>
        ))}
      </ul>
      {demo && (
        <p className="demo-note" role="note">
          <strong>Demo assistant.</strong> No AI key is configured, so replies come from a simple rule-based assistant. Search, ranking, memory and saving are all real; the
          conversation is scripted. Add an API key to the backend for the full experience.
        </p>
      )}
    </div>
  );
}
