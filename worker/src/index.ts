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
const LOGO_BASE = "https://image.tmdb.org/t/p/w92";
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
// TMDB's TV genres are coarser; map each film genre to its closest one (null = no TV equivalent).
const TV_GENRES: Record<GenreName, number | null> = {
  Action: 10759, Adventure: 10759, Animation: 16, Comedy: 35, Crime: 80,
  Documentary: 99, Drama: 18, Family: 10751, Fantasy: 10765, History: null,
  Horror: null, Music: null, Mystery: 9648, Romance: null,
  "Science Fiction": 10765, Thriller: 9648, War: 10768, Western: 37,
};

type Kind = "movie" | "tv";
const KIND_LABEL: Record<Kind, string> = { movie: "film", tv: "series" };

// ---------- Request shape ----------

interface RecommendRequest {
  vibe?: string;
  music?: string;
  plot?: string;
  eras?: string[]; // decades like "1970s"; empty or missing means any
  kinds: Kind[]; // films, series, or both
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
  reference_titles: z
    .array(z.object({ title: z.string(), kind: z.enum(["movie", "tv"]) }))
    .describe("4-6 real, well-known titles that strongly match, each marked as a movie or a tv series."),
});
type Profile = z.infer<typeof ProfileSchema>;

const PROFILE_SYSTEM = `You are a film and TV curator. A user describes what they want to watch using any mix of: a mood or vibe, music they're into, a room or setting (sometimes as a photo), an era, or a plot idea.
Translate that into a search profile for a movie and TV database.
- Music: infer the emotional tone, energy, and aesthetic of the artist or song, and map that to film tone.
- Photos: read the palette, decor, era cues, lighting, and atmosphere; treat it as the setting the user wants to feel.
- Eras: if the user picked decades, draw reference titles from those decades (spread across them if there are several). year_from/year_to are only used when no decades are picked: then use null unless the input clearly implies a period.
- Reference titles must be real films or series you are confident exist, and only of the kinds the user wants.`;

async function buildProfile(client: Anthropic, req: RecommendRequest): Promise<Profile> {
  const parts: string[] = [];
  if (req.vibe) parts.push(`Vibe / mood: ${req.vibe}`);
  if (req.music) parts.push(`Music they're into: ${req.music}`);
  if (req.plot) parts.push(`Plot idea: ${req.plot}`);
  if (req.eras?.length) parts.push(`Eras: ${req.eras.join(", ")}`);
  if (req.image) parts.push("Setting / decor: see the attached photo.");
  parts.push(`They want: ${req.kinds.map((k) => (k === "movie" ? "films" : "series")).join(" and ")}`);

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

// TMDB names the same fields differently for films and series.
interface TmdbItem {
  id: number;
  title?: string;
  name?: string;
  release_date?: string;
  first_air_date?: string;
  overview?: string;
  poster_path?: string | null;
  vote_average?: number;
  original_language?: string;
  adult?: boolean;
}

interface Candidate {
  key: string; // "movie-123" / "tv-123": TMDB ids are only unique within a kind
  kind: Kind;
  id: number;
  title: string;
  year: number | null;
  overview: string;
  posterPath: string | null;
  rating: number | null;
  language: string | null;
}

function toCandidate(kind: Kind, m: TmdbItem): Candidate | null {
  const title = m.title ?? m.name;
  if (m.adult || !m.overview || !title) return null;
  const year = Number((m.release_date ?? m.first_air_date)?.slice(0, 4)) || null;
  return {
    key: `${kind}-${m.id}`, kind, id: m.id, title, year, overview: m.overview,
    posterPath: m.poster_path ?? null, rating: m.vote_average ? Math.round(m.vote_average * 10) / 10 : null,
    language: m.original_language ?? null,
  };
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

async function gatherCandidates(env: Env, profile: Profile, ranges: YearRange[], kinds: Kind[]): Promise<Candidate[]> {
  type Page = { results: TmdbItem[] };
  const empty: Page = { results: [] };

  // Keyword IDs are shared between films and series, so look them up once.
  const keywordIds = (
    await Promise.all(
      profile.keywords.slice(0, 6).map(async (k) => {
        const r = await soft(tmdb<{ results: { id: number }[] }>(env, "/search/keyword", { query: k }), { results: [] });
        return r.results[0]?.id;
      }),
    )
  ).filter((id): id is number => id !== undefined);

  // One discover query per decade so each picked era gets its own results; past four, one query spans them all.
  const queryRanges =
    ranges.length <= 4
      ? ranges
      : [{ from: Math.min(...ranges.map((r) => r.from ?? 0)), to: Math.max(...ranges.map((r) => r.to ?? 9999)) }];
  const inRange = (year: number | null) => !year || ranges.some((r) => (!r.from || year >= r.from) && (!r.to || year <= r.to));

  async function forKind(kind: Kind): Promise<Candidate[]> {
    // Reference titles → the title itself plus TMDB's "recommendations" for it.
    const refs = profile.reference_titles.filter((r) => r.kind === kind).slice(0, 4);
    const fromReferences = Promise.all(refs.map(async ({ title }) => {
      const search = await soft(tmdb<Page>(env, `/search/${kind}`, { query: title, include_adult: "false" }), empty);
      const hit = search.results[0];
      if (!hit) return [];
      const recs = await soft(tmdb<Page>(env, `/${kind}/${hit.id}/recommendations`), empty);
      return [hit, ...recs.results.slice(0, 8)];
    }));
    // Runs alongside the discover queries below; if those throw first, don't leave this rejection unhandled.
    fromReferences.catch(() => {});

    const genreIds = profile.genres.map((g) => (kind === "movie" ? GENRES[g] : TV_GENRES[g])).filter((id): id is number => id !== null);
    const base: Record<string, string> = {
      include_adult: "false",
      sort_by: "vote_average.desc",
      "vote_count.gte": kind === "movie" ? "150" : "80", // series collect fewer votes
    };
    if (genreIds.length) base.with_genres = [...new Set(genreIds)].join("|"); // "|" = any of
    const dateField = kind === "movie" ? "primary_release_date" : "first_air_date";
    const discover = (range: YearRange, extra: Record<string, string> = {}) => {
      const params = { ...base, ...extra };
      if (range.from) params[`${dateField}.gte`] = `${range.from}-01-01`;
      if (range.to) params[`${dateField}.lte`] = `${range.to}-12-31`;
      return soft(tmdb<Page>(env, `/discover/${kind}`, params), empty);
    };
    // Keyword-matched queries first: they're the most specific signal.
    const discovers = [
      ...(keywordIds.length ? queryRanges.map((r) => discover(r, { with_keywords: keywordIds.join("|") })) : []),
      ...queryRanges.map((r) => discover(r)),
    ];

    const [refLists, discoverPages] = await Promise.all([fromReferences, Promise.all(discovers)]);
    const seen = new Map<string, Candidate>();
    for (const item of [...discoverPages.flatMap((p) => p.results), ...refLists.flat()]) {
      const c = toCandidate(kind, item);
      if (c && !seen.has(c.key) && inRange(c.year)) seen.set(c.key, c);
    }
    return [...seen.values()].slice(0, Math.floor(MAX_CANDIDATES / kinds.length));
  }

  const perKind = await Promise.all(kinds.map(forKind));
  // Interleave so a "both" request shows the picker films and series side by side.
  const out: Candidate[] = [];
  for (let i = 0; i < Math.max(...perKind.map((l) => l.length)); i++) for (const l of perKind) if (l[i]) out.push(l[i]);
  return out;
}

// ---------- Claude step 2: pick and explain, from real candidates only ----------

const PicksSchema = z.object({
  picks: z.array(
    z.object({
      id: z.string().describe("The id in square brackets from the candidate list, e.g. movie-123."),
      reason: z.string().describe("One sentence, under 25 words, on why it fits this specific request."),
    }),
  ),
});

async function rankCandidates(client: Anthropic, req: RecommendRequest, profile: Profile, candidates: Candidate[]) {
  const list = candidates
    .map((c) => `[${c.key}] ${c.title} (${KIND_LABEL[c.kind]}, ${c.year ?? "?"}, lang: ${c.language ?? "?"}) — ${c.overview.slice(0, 220)}`)
    .join("\n");

  const ask = [req.vibe && `Vibe: ${req.vibe}`, req.music && `Music: ${req.music}`, req.plot && `Plot: ${req.plot}`, req.eras?.length && `Eras: ${req.eras.join(", ")} (spread picks across them)`, req.image && "They also shared a photo of a setting."]
    .filter(Boolean)
    .join("\n");

  const response = await client.messages.parse({
    model: MODEL,
    max_tokens: 1024,
    system: `You are a film and TV curator picking what someone should watch tonight. Choose only from the candidate list, by id. When both films and series are wanted, mix them unless the user's words lean one way. Honour anything the user rules out, such as a country or language (each candidate's original language is given as an ISO code). Prefer a varied set (not all the same director or decade unless asked). Write reasons that speak to the user's own words.`,
    messages: [
      {
        role: "user",
        content: `What they asked for:\n${ask}\n\nYour read of it: ${profile.summary}\n\nCandidates:\n${list}\n\nPick the best ${MAX_PICKS}.`,
      },
    ],
    output_config: { format: zodOutputFormat(PicksSchema) },
  });
  if (!response.parsed_output) throw new Error(`Could not rank results (stop_reason: ${response.stop_reason})`);

  const byKey = new Map(candidates.map((c) => [c.key, c]));
  // Drop any id the model didn't actually get from us.
  return response.parsed_output.picks
    .map((p) => ({ item: byKey.get(p.id.replace(/^\[|\]$/g, "")), reason: p.reason }))
    .filter((p): p is { item: Candidate; reason: string } => !!p.item)
    .slice(0, MAX_PICKS);
}

// Subscription, free and ad-supported services. `link` is TMDB's "where to watch" page for the
// film in that country (powered by JustWatch), which links out to each service.
async function watchProviders(env: Env, kind: Kind, id: number, region: string) {
  type Provider = { provider_id: number; provider_name: string; logo_path?: string };
  type Providers = { results: Record<string, { link?: string; flatrate?: Provider[]; free?: Provider[]; ads?: Provider[] }> };
  const r = await soft(tmdb<Providers>(env, `/${kind}/${id}/watch/providers`), { results: {} });
  const here = r.results[region];
  const seen = new Set<number>();
  const services = [...(here?.flatrate ?? []), ...(here?.free ?? []), ...(here?.ads ?? [])]
    .filter((p) => !seen.has(p.provider_id) && seen.add(p.provider_id))
    .slice(0, 5)
    .map((p) => ({ name: p.provider_name, logo: p.logo_path ? LOGO_BASE + p.logo_path : null }));
  return { services, link: here?.link ?? null };
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

function parseKinds(body: any): Kind[] {
  // Older pages don't send this and only ever meant films.
  const raw: unknown[] = Array.isArray(body?.kinds) ? body.kinds : ["movie"];
  const kinds = (["movie", "tv"] as const).filter((k) => raw.includes(k));
  return kinds.length ? kinds : ["movie"];
}

function parseRequest(body: any): RecommendRequest | string {
  const req: RecommendRequest = {
    vibe: clean(body?.vibe),
    music: clean(body?.music),
    plot: clean(body?.plot),
    eras: parseEras(body),
    kinds: parseKinds(body),
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
      const candidates = await gatherCandidates(env, profile, yearRanges(req, profile), req.kinds);
      if (!candidates.length) return json({ profile, results: [] }, 200, cors);

      const picks = await rankCandidates(client, req, profile, candidates);
      const results = await Promise.all(
        picks.map(async ({ item, reason }) => {
          const watch = await watchProviders(env, item.kind, item.id, req.region!);
          return {
            id: item.id,
            kind: item.kind,
            title: item.title,
            year: item.year ? String(item.year) : null,
            rating: item.rating,
            overview: item.overview,
            poster: item.posterPath ? POSTER_BASE + item.posterPath : null,
            url: `https://www.themoviedb.org/${item.kind}/${item.id}`,
            reason,
            streaming: watch.services,
            watchUrl: watch.link,
          };
        }),
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
      return json({ error: `Something went wrong finding picks: ${err instanceof Error ? err.message.slice(0, 200) : "unknown error"}` }, 500, cors);
    }
  },
};
