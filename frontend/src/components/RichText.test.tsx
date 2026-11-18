import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { RichText, toBlocks } from "./RichText";

describe("toBlocks", () => {
  it("groups bullets, numbered items and paragraphs", () => {
    expect(toBlocks("Intro\n\n- one\n- two\n\n1. first\n2. second\nThe end")).toEqual([
      { kind: "p", lines: ["Intro"] },
      { kind: "ul", lines: ["one", "two"] },
      { kind: "ol", lines: ["first", "second"] },
      { kind: "p", lines: ["The end"] },
    ]);
  });
});

describe("RichText", () => {
  it("renders structure, bold text and links safely", () => {
    render(<RichText text={"**Great** news\n- item with https://example.com/x."} />);
    expect(screen.getByText("Great").tagName).toBe("STRONG");
    const link = screen.getByRole("link", { name: "https://example.com/x" });
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
    expect(link).toHaveAttribute("target", "_blank");
    expect(screen.getByRole("listitem")).toBeInTheDocument();
  });

  it("makes #id references clickable and reports the id", async () => {
    const onJob = vi.fn();
    render(<RichText text="Try #144 or #75." onJob={onJob} />);
    await userEvent.click(screen.getByRole("button", { name: "Show job 144" }));
    expect(onJob).toHaveBeenCalledWith(144);
    expect(screen.getByRole("button", { name: "Show job 75" })).toBeInTheDocument();
  });

  it("does not treat C#12 or HTML entities as job references", () => {
    render(<RichText text="C#12 and &#39; are not jobs" onJob={() => {}} />);
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("never injects markup from model or job text", () => {
    const { container } = render(<RichText text={'<img src=x onerror="alert(1)"> <script>alert(1)</script> javascript:alert(1)'} />);
    expect(container.querySelector("img, script")).toBeNull();
    expect(container.textContent).toContain("<script>alert(1)</script>"); // shown literally, not executed
    expect(container.querySelector("a")).toBeNull(); // javascript: is not linkified
  });
});
