import Anthropic from "@anthropic-ai/sdk";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import { z } from "zod";

interface Env {
  ANTHROPIC_API_KEY: string;
  TMDB_TOKEN: string;
  ALLOWED_ORIGINS: string;
}

// Haiku keeps each recommendation at roughly half a cent.
const MODEL = "claude-haiku-4-5";
const TMDB = "https://api.themoviedb.org/3";
const POSTER_BASE = "https://image.tmdb.org/t/p/w342";
const MAX_TEXT = 600;
const MAX_IMAGE_BYTES = 3_500_000; // base64 length; the frontend resizes before upload
const MAX_CANDIDATES = 40;
const MAX_PICKS = 6;

// TMDB's genre IDs are stable, so they're hardcoded rather than fetched on every request.
const GENRES = {
  Action: 28, Adventure: 12, Animation: 16, Comedy: 35, Crime: 80,
  Documentary: 99, Drama: 18, Family: 10751, Fantasy: 14, History: 36,
  Horror: 27, Music: 10402, Mystery: 9648, Romance: 10749,
  "Science Fiction": 878, Thriller: 53, War: 10752, Western: 37,
} as const;
type GenreName = keyof typeof GENRES;
const GENRE_NAMES = Object.keys(GENRES) as [GenreName, ...GenreName[]];

// ---------- Request shape ----------

interface RecommendRequest {
  vibe?: string;
  music?: string;
  plot?: string;
  eras?: string[]; // decades like "1970s"; empty or missing means any
  region?: string; // ISO country code for streaming providers
  image?: { media_type: "image/jpeg" | "image/png" | "image/webp"; data: string };
}

// ---------- Claude step 1: turn fuzzy input into a search profile ----------

const ProfileSchema = z.object({
  summary: z.string().describe("One evocative sentence describing the viewing mood the user is after."),
  genres: z.array(z.enum(GENRE_NAMES)).describe("1-3 TMDB genres that best fit."),
  keywords: z.array(z.string()).describe("3-6 short TMDB-style keywords, e.g. 'heist', 'road trip', 'small town', 'neo-noir'."),
  year_from: z.number().int().nullable().describe("Earliest release year, or null for no limit."),
  year_to: z.number().int().nullable().describe("Latest release year, or null for no limit."),
  reference_titles: z.array(z.string()).describe("4-6 real, well-known films that strongly match. Titles only."),
});
type Profile = z.infer<typeof ProfileSchema>;

const PROFILE_SYSTEM = `You are a film curator. A user describes what they want to watch using any mix of: a mood or vibe, music they're into, a room or setting (sometimes as a photo), an era, or a plot idea.
Translate that into a search profile for a movie database.
- Music: infer the emotional tone, energy, and aesthetic of the artist or song, and map that to film tone.
- Photos: read the palette, decor, era cues, lighting, and atmosphere; treat it as the setting the user wants to feel.
- Eras: if the user picked decades, draw reference titles from those decades (spread across them if there are several). year_from/year_to are only used when no decades are picked: then use null unless the input clearly implies a period.
- Reference titles must be real films you are confident exist.`;

async function buildProfile(client: Anthropic, req: RecommendRequest): Promise<Profile> {
  const parts: string[] = [];
  if (req.vibe) parts.push(`Vibe / mood: ${req.vibe}`);
  if (req.music) parts.push(`Music they're into: ${req.music}`);
  if (req.plot) parts.push(`Plot idea: ${req.plot}`);
  if (req.eras?.length) parts.push(`Eras: ${req.eras.join(", ")}`);
  if (req.image) parts.push("Setting / decor: see the attached photo.");

  const content: Anthropic.ContentBlockParam[] = [];
  if (req.image) {
    content.push({ type: "image", source: { type: "base64", media_type: req.image.media_type, data: req.image.data } });
  }
  content.push({ type: "text", text: parts.join("\n") });

  const response = await client.messages.parse({
    model: MODEL,
    max_tokens: 1024,
    system: PROFILE_SYSTEM,
    messages: [{ role: "user", content }],
    output_config: { format: zodOutputFormat(ProfileSchema) },
  });
  if (!response.parsed_output) throw new Error(`Could not read the request (stop_reason: ${response.stop_reason})`);
  return response.parsed_output;
}

// ---------- TMDB: gather real candidates ----------

interface TmdbMovie {
  id: number;
  title: string;
  release_date?: string;
  overview?: string;
  poster_path?: string | null;
  vote_average?: number;
  vote_count?: number;
  adult?: boolean;
}

async function tmdb<T>(env: Env, path: string, params: Record<string, string> = {}): Promise<T> {
  const url = new URL(TMDB + path);
  for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
  const res = await fetch(url, { headers: { Authorization: `Bearer ${env.TMDB_TOKEN}`, Accept: "application/json" } });
  if (res.status === 401) throw new TmdbAuthError();
  if (!res.ok) throw new Error(`TMDB ${path} failed: ${res.status}`);
  return res.json() as Promise<T>;
}

class TmdbAuthError extends Error {}

// One failed lookup (an obscure title, a missing keyword) shouldn't sink the whole request,
// but a rejected token should, so it isn't mistaken for "no matches".
async function soft<T>(p: Promise<T>, fallback: T): Promise<T> {
  try {
    return await p;
  } catch (err) {
    if (err instanceof TmdbAuthError) throw err;
    return fallback;
  }
}

type YearRange = { from: number | null; to: number | null };

// Picked decades are exact, so they come from the request rather than the model's guess.
function yearRanges(req: RecommendRequest, profile: Profile): YearRange[] {
  if (req.eras?.length) return req.eras.map((e) => ({ from: Number(e.slice(0, 4)), to: Number(e.slice(0, 4)) + 9 }));
  return [{ from: profile.year_from, to: profile.year_to }];
}

async function gatherCandidates(env: Env, profile: Profile, ranges: YearRange[]): Promise<TmdbMovie[]> {
  type Page = { results: TmdbMovie[] };
  const empty: Page = { results: [] };

  // Reference titles → the film itself plus TMDB's "recommendations" for it.
  const fromReferences = Promise.all(profile.reference_titles.slice(0, 5).map(async (title) => {
    const search = await soft(tmdb<Page>(env, "/search/movie", { query: title, include_adult: "false" }), empty);
    const hit = search.results[0];
    if (!hit) return [];
    const recs = await soft(tmdb<Page>(env, `/movie/${hit.id}/recommendations`), empty);
    return [hit, ...recs.results.slice(0, 8)];
  }));
  // Runs alongside the keyword lookups below; if those throw first, don't leave this rejection unhandled.
  fromReferences.catch(() => {});

  // Keywords → TMDB keyword IDs, then a discover query filtered by genre, keyword and era.
  const keywordIds = (
    await Promise.all(
      profile.keywords.slice(0, 6).map(async (k) => {
        const r = await soft(tmdb<{ results: { id: number }[] }>(env, "/search/keyword", { query: k }), { results: [] });
        return r.results[0]?.id;
      }),
    )
  ).filter((id): id is number => id !== undefined);

  const discoverParams: Record<string, string> = {
    include_adult: "false",
    sort_by: "vote_average.desc",
    "vote_count.gte": "150",
  };
  const genreIds = profile.genres.map((g) => GENRES[g]);
  if (genreIds.length) discoverParams.with_genres = genreIds.join("|"); // "|" = any of

  // One discover query per decade so each picked era gets its own results; past four, one query spans them all.
  const queryRanges =
    ranges.length <= 4
      ? ranges
      : [{ from: Math.min(...ranges.map((r) => r.from ?? 0)), to: Math.max(...ranges.map((r) => r.to ?? 9999)) }];
  const discover = (range: YearRange, extra: Record<string, string> = {}) => {
    const params = { ...discoverParams, ...extra };
    if (range.from) params["primary_release_date.gte"] = `${range.from}-01-01`;
    if (range.to) params["primary_release_date.lte"] = `${range.to}-12-31`;
    return soft(tmdb<Page>(env, "/discover/movie", params), empty);
  };
  // Keyword-matched queries first: they're the most specific signal.
  const discovers = [
    ...(keywordIds.length ? queryRanges.map((r) => discover(r, { with_keywords: keywordIds.join("|") })) : []),
    ...queryRanges.map((r) => discover(r)),
  ];

  const [refLists, discoverPages] = await Promise.all([fromReferences, Promise.all(discovers)]);

  const inRange = (year: number) => !year || ranges.some((r) => (!r.from || year >= r.from) && (!r.to || year <= r.to));
  const seen = new Map<number, TmdbMovie>();
  for (const m of [...discoverPages.flatMap((p) => p.results), ...refLists.flat()]) {
    if (m.adult || !m.overview || seen.has(m.id)) continue;
    if (!inRange(Number(m.release_date?.slice(0, 4)))) continue;
    seen.set(m.id, m);
  }
  return [...seen.values()].slice(0, MAX_CANDIDATES);
}

// ---------- Claude step 2: pick and explain, from real candidates only ----------

const PicksSchema = z.object({
  picks: z.array(
    z.object({
      id: z.number().int().describe("The TMDB id from the candidate list."),
      reason: z.string().describe("One sentence, under 25 words, on why it fits this specific request."),
    }),
  ),
});

async function rankCandidates(client: Anthropic, req: RecommendRequest, profile: Profile, candidates: TmdbMovie[]) {
  const list = candidates
    .map((m) => `[${m.id}] ${m.title} (${m.release_date?.slice(0, 4) || "?"}) — ${m.overview!.slice(0, 220)}`)
    .join("\n");

  const ask = [req.vibe && `Vibe: ${req.vibe}`, req.music && `Music: ${req.music}`, req.plot && `Plot: ${req.plot}`, req.eras?.length && `Eras: ${req.eras.join(", ")} (spread picks across them)`, req.image && "They also shared a photo of a setting."]
    .filter(Boolean)
    .join("\n");

  const response = await client.messages.parse({
    model: MODEL,
    max_tokens: 1024,
    system: `You are a film curator picking what someone should watch tonight. Choose only from the candidate list, by id. Prefer a varied set (not all the same director or decade unless asked). Write reasons that speak to the user's own words.`,
    messages: [
      {
        role: "user",
        content: `What they asked for:\n${ask}\n\nYour read of it: ${profile.summary}\n\nCandidates:\n${list}\n\nPick the best ${MAX_PICKS}.`,
      },
    ],
    output_config: { format: zodOutputFormat(PicksSchema) },
  });
  if (!response.parsed_output) throw new Error(`Could not rank results (stop_reason: ${response.stop_reason})`);

  const byId = new Map(candidates.map((m) => [m.id, m]));
  // Drop any id the model didn't actually get from us.
  return response.parsed_output.picks.filter((p) => byId.has(p.id)).slice(0, MAX_PICKS).map((p) => ({ movie: byId.get(p.id)!, reason: p.reason }));
}

async function watchProviders(env: Env, id: number, region: string): Promise<string[]> {
  type Providers = { results: Record<string, { flatrate?: { provider_name: string }[] }> };
  const r = await soft(tmdb<Providers>(env, `/movie/${id}/watch/providers`), { results: {} });
  return (r.results[region]?.flatrate ?? []).map((p) => p.provider_name).slice(0, 4);
}

// ---------- HTTP plumbing ----------

function corsHeaders(request: Request, env: Env): Record<string, string> {
  const origin = request.headers.get("Origin") ?? "";
  const allowed = env.ALLOWED_ORIGINS.split(",").map((s) => s.trim());
  return allowed.includes(origin)
    ? { "Access-Control-Allow-Origin": origin, "Access-Control-Allow-Methods": "POST, OPTIONS", "Access-Control-Allow-Headers": "Content-Type", Vary: "Origin" }
    : {};
}

function json(body: unknown, status: number, headers: Record<string, string>) {
  return new Response(JSON.stringify(body), { status, headers: { ...headers, "Content-Type": "application/json" } });
}

function clean(s: unknown): string | undefined {
  return typeof s === "string" && s.trim() ? s.trim().slice(0, MAX_TEXT) : undefined;
}

function parseEras(body: any): string[] {
  // Older pages send a single `era` string.
  const raw: unknown[] = Array.isArray(body?.eras) ? body.eras : body?.era ? [body.era] : [];
  const valid = raw.filter((e): e is string => typeof e === "string" && /^(19|20)\d0s$/.test(e));
  return [...new Set(valid)].sort().slice(0, 10);
}

function parseRequest(body: any): RecommendRequest | string {
  const req: RecommendRequest = {
    vibe: clean(body?.vibe),
    music: clean(body?.music),
    plot: clean(body?.plot),
    eras: parseEras(body),
    region: /^[A-Z]{2}$/.test(body?.region ?? "") ? body.region : "US",
  };
  const img = body?.image;
  if (img) {
    if (!["image/jpeg", "image/png", "image/webp"].includes(img.media_type) || typeof img.data !== "string") return "Unsupported image.";
    if (img.data.length > MAX_IMAGE_BYTES) return "Image is too large.";
    req.image = { media_type: img.media_type, data: img.data };
  }
  if (!req.vibe && !req.music && !req.plot && !req.image && !req.eras?.length) return "Tell me something: a vibe, music, a plot, an era or a photo.";
  return req;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const cors = corsHeaders(request, env);
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: cors });

    const url = new URL(request.url);
    if (request.method !== "POST" || url.pathname !== "/api/recommend") return json({ error: "Not found" }, 404, cors);
    // Only the allowed sites may spend your API credits.
    if (!cors["Access-Control-Allow-Origin"]) return json({ error: "Origin not allowed" }, 403, cors);

    let body: unknown;
    try {
      body = await request.json();
    } catch {
      return json({ error: "Invalid JSON" }, 400, cors);
    }
    const req = parseRequest(body);
    if (typeof req === "string") return json({ error: req }, 400, cors);

    if (!env.ANTHROPIC_API_KEY) return json({ error: "Server is missing its ANTHROPIC_API_KEY secret." }, 500, cors);
    if (!env.TMDB_TOKEN) return json({ error: "Server is missing its TMDB_TOKEN secret." }, 500, cors);

    const client = new Anthropic({ apiKey: env.ANTHROPIC_API_KEY });
    try {
      const profile = await buildProfile(client, req);
      const candidates = await gatherCandidates(env, profile, yearRanges(req, profile));
      if (!candidates.length) return json({ profile, results: [] }, 200, cors);

      const picks = await rankCandidates(client, req, profile, candidates);
      const results = await Promise.all(
        picks.map(async ({ movie, reason }) => ({
          id: movie.id,
          title: movie.title,
          year: movie.release_date?.slice(0, 4) ?? null,
          rating: movie.vote_average ? Math.round(movie.vote_average * 10) / 10 : null,
          overview: movie.overview,
          poster: movie.poster_path ? POSTER_BASE + movie.poster_path : null,
          url: `https://www.themoviedb.org/movie/${movie.id}`,
          reason,
          streaming: await watchProviders(env, movie.id, req.region!),
        })),
      );
      return json({ profile, results }, 200, cors);
    } catch (err) {
      console.error(err);
      if (err instanceof Anthropic.RateLimitError) return json({ error: "Busy right now, try again in a moment." }, 429, cors);
      if (err instanceof Anthropic.AuthenticationError) return json({ error: "The Anthropic API key was rejected. Check ANTHROPIC_API_KEY." }, 500, cors);
      if (err instanceof TmdbAuthError) return json({ error: "The TMDB token was rejected. Check TMDB_TOKEN." }, 500, cors);
      if (err instanceof Anthropic.APIError) {
        // e.g. "credit balance is too low" — worth showing as-is.
        return json({ error: `The AI service had a problem (${err.status ?? "network"}): ${err.message.slice(0, 200)}` }, 502, cors);
      }
      return json({ error: `Something went wrong finding movies: ${err instanceof Error ? err.message.slice(0, 200) : "unknown error"}` }, 500, cors);
    }
  },
};
