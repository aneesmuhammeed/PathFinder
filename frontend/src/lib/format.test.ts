import { describe, expect, it } from "vitest";
import { formatMinSalary, formatSalary, isProfileEmpty, relativeDate, toolLabel } from "./format";

const sal = (over: object) => ({ salary_min: null, salary_max: null, salary_currency: null, salary_period: null, ...over });

describe("formatSalary", () => {
  it("formats Indian annual pay as LPA and monthly stipends in full", () => {
    expect(formatSalary(sal({ salary_min: 1_500_000, salary_max: 2_500_000, salary_currency: "INR", salary_period: "year" }))).toBe("₹15–25 LPA");
    expect(formatSalary(sal({ salary_min: 35000, salary_max: 40000, salary_currency: "INR", salary_period: "month" }))).toBe("₹35,000–40,000/month");
    expect(formatSalary(sal({ salary_min: 650_000, salary_max: 1_050_000, salary_currency: "INR", salary_period: "year" }))).toBe("₹6.5–10.5 LPA");
  });
  it("formats other currencies without converting them", () => {
    expect(formatSalary(sal({ salary_min: 80000, salary_max: 102000, salary_currency: "USD", salary_period: "year" }))).toBe("USD 80,000–102,000/year");
  });
  it("returns null when pay is undisclosed", () => {
    expect(formatSalary(sal({}))).toBeNull();
    expect(formatSalary(sal({ salary_min: 1, salary_max: 2 }))).toBeNull(); // no currency
  });
  it("formats the profile's annual-INR minimum", () => expect(formatMinSalary(800_000)).toBe("₹8 LPA"));
});

describe("relativeDate", () => {
  const now = new Date("2026-11-18T12:00:00Z");
  it.each([
    ["2026-11-18T01:00:00Z", "Posted today"], ["2026-11-17T01:00:00Z", "Posted yesterday"],
    ["2026-11-08T12:00:00Z", "Posted 10 days ago"], ["2026-08-18T12:00:00Z", "Posted 3 months ago"],
  ])("%s → %s", (iso, label) => expect(relativeDate(iso, now)).toBe(label));
  it("is null for missing or invalid dates", () => {
    expect(relativeDate(null, now)).toBeNull();
    expect(relativeDate("not a date", now)).toBeNull();
  });
});

describe("toolLabel", () => {
  it("describes what the agent is doing in plain language", () => {
    expect(toolLabel("search_jobs", { query: "python backend" })).toBe("Searching jobs for “python backend”");
    expect(toolLabel("search_jobs", {})).toBe("Finding jobs that fit your profile");
    expect(toolLabel("update_profile", {})).toBe("Remembering what you told me");
    expect(toolLabel("something_new", {})).toContain("something_new"); // unknown tools degrade gracefully
  });
  it("truncates very long queries", () => expect(toolLabel("search_jobs", { query: "x".repeat(200) }).length).toBeLessThan(90));
});

describe("isProfileEmpty", () => {
  const empty = { skills: [], target_roles: [], preferred_locations: [], seniority: null, work_mode: null, job_type: null, min_salary: null };
  it("detects empty and non-empty profiles", () => {
    expect(isProfileEmpty(null)).toBe(true);
    expect(isProfileEmpty(empty)).toBe(true);
    expect(isProfileEmpty({ ...empty, skills: ["python"] })).toBe(false);
  });
});
