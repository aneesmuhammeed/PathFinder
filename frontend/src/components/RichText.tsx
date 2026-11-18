import type { ReactNode } from "react";

// Assistant replies are plain text with light structure (lists, **bold**, #123 job references, links). This renders
// exactly that, as React elements. There is deliberately no HTML injection path: model output and job text are untrusted.
const INLINE = /(\*\*[^*\n]+\*\*|(?<![\w&/])#\d{1,9}\b|https?:\/\/[^\s)<]+)/g;

function inline(text: string, onJob: ((id: number) => void) | undefined, key: string): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    const k = `${key}-${i}`;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) return <strong key={k}>{part.slice(2, -2)}</strong>;
    if (/^#\d+$/.test(part)) {
      const id = Number(part.slice(1));
      return onJob ? (
        <button key={k} type="button" className="jobref" onClick={() => onJob(id)} aria-label={`Show job ${id}`}>
          {part}
        </button>
      ) : (
        <span key={k} className="jobref">
          {part}
        </span>
      );
    }
    if (/^https?:\/\//.test(part)) {
      const trimmed = part.replace(/[.,;:!?]+$/, "");
      return (
        <span key={k}>
          <a href={trimmed} target="_blank" rel="noopener noreferrer">
            {trimmed}
          </a>
          {part.slice(trimmed.length)}
        </span>
      );
    }
    return part;
  });
}

type Block = { kind: "p" | "ul" | "ol"; lines: string[] };

export function toBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  for (const raw of text.replace(/\r\n/g, "\n").split("\n")) {
    const line = raw.trimEnd();
    if (!line.trim()) {
      blocks.push({ kind: "p", lines: [] }); // paragraph break marker
      continue;
    }
    const bullet = /^\s*[-*•]\s+(.*)$/.exec(line);
    const numbered = /^\s*\d+[.)]\s+(.*)$/.exec(line);
    const kind = bullet ? "ul" : numbered ? "ol" : "p";
    const content = bullet?.[1] ?? numbered?.[1] ?? line;
    const last = blocks[blocks.length - 1];
    if (last && last.kind === kind && last.lines.length) last.lines.push(content);
    else blocks.push({ kind, lines: [content] });
  }
  return blocks.filter((b) => b.lines.length);
}

export function RichText({ text, onJob }: { text: string; onJob?: (id: number) => void }) {
  return (
    <>
      {toBlocks(text).map((b, i) => {
        const items = b.lines.map((l, j) => <li key={j}>{inline(l, onJob, `${i}-${j}`)}</li>);
        if (b.kind === "ul") return <ul key={i}>{items}</ul>;
        if (b.kind === "ol") return <ol key={i}>{items}</ol>;
        return (
          <p key={i}>
            {b.lines.map((l, j) => (
              <span key={j}>
                {j > 0 && <br />}
                {inline(l, onJob, `${i}-${j}`)}
              </span>
            ))}
          </p>
        );
      })}
    </>
  );
}
