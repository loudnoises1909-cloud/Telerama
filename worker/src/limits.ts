import { DurableObject } from "cloudflare:workers";

// Daily search caps, so a shared link can't run up the Claude bill. One instance holds every
// counter, which keeps the check-and-increment atomic; at a few hundred searches a day that's
// far within what a single Durable Object handles.
export const DAILY_PER_PERSON = 15;
export const DAILY_TOTAL = 100;

export type LimitResult = { ok: true } | { ok: false; reason: "person" | "total" };

export class Limits extends DurableObject {
  constructor(ctx: DurableObjectState, env: unknown) {
    super(ctx, env as never);
    ctx.storage.sql.exec("CREATE TABLE IF NOT EXISTS counts (day TEXT NOT NULL, who TEXT NOT NULL, n INTEGER NOT NULL, PRIMARY KEY (day, who))");
  }

  // Counts one search for `who` today (UTC), unless that would pass either cap.
  take(who: string, perPerson = DAILY_PER_PERSON, total = DAILY_TOTAL): LimitResult {
    const sql = this.ctx.storage.sql;
    const day = new Date().toISOString().slice(0, 10);
    const count = (key: string) => (sql.exec("SELECT n FROM counts WHERE day = ? AND who = ?", day, key).toArray()[0]?.n as number | undefined) ?? 0;
    if (count("*") >= total) return { ok: false, reason: "total" };
    if (count(who) >= perPerson) return { ok: false, reason: "person" };
    for (const key of [who, "*"]) {
      sql.exec("INSERT INTO counts (day, who, n) VALUES (?, ?, 1) ON CONFLICT (day, who) DO UPDATE SET n = n + 1", day, key);
    }
    // Earlier days are no longer needed.
    sql.exec("DELETE FROM counts WHERE day < ?", day);
    return { ok: true };
  }
}
