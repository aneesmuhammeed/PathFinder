import type { ChatEvent } from "./types";

export interface ParsedChunk {
  events: ChatEvent[];
  /** Incomplete trailing data: prepend it to the next network chunk. */
  rest: string;
}

/**
 * Incremental server-sent-events parser. Network chunks can split an event anywhere (even mid-UTF-8-character
 * boundaries are handled by the decoder upstream), so we only consume complete blocks ("\n\n"-terminated) and hand the
 * remainder back. Malformed JSON blocks are skipped rather than killing the stream.
 */
export function parseSSE(buffer: string): ParsedChunk {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const blocks = normalized.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: ChatEvent[] = [];
  for (const block of blocks) {
    const data = block
      .split("\n")
      .filter((l) => l.startsWith("data:"))
      .map((l) => l.slice(5).replace(/^ /, ""))
      .join("\n");
    if (!data) continue; // comments / keep-alives
    try {
      events.push(JSON.parse(data) as ChatEvent);
    } catch {
      /* ignore a malformed block */
    }
  }
  return { events, rest };
}
