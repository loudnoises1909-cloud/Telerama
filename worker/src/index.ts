import Anthropic from "@anthropic-ai/sdk";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import { z } from "zod";
import { DAILY_PER_PERSON, DAILY_TOTAL, Limits } from "./limits";

// The class must be exported from the main module for Cloudflare to run it.
export { Limits };

interface Env {
  ANTHROPIC_API_KEY: string;
  TMDB_TOKEN: string;
  ALLOWED_ORIGINS: string;
  SEARCH_BURST: RateLimit; // 5 searches a minute per person (wrangler.toml)
  LIMITS: DurableObjectNamespace<Limits>; // daily caps
  DAILY_SEARCHES_PER_PERSON?: string;
  DAILY_SEARCHES_TOTAL?: string;
}

// Haiku keeps each recommendation at roughly half a cent.
const MODEL = "claude-haiku-4-5";
const TMDB = "https://api.themoviedb.org/3";
const POSTER_BASE = "https://image.tmdb.org/t/p/w342";
const LOGO_BASE = "https://image.tmdb.org/t/p/w92";
const MAX_TEXT = 600;
const MAX_IMAGE_BYTES = 3_500_000; // base64 length; the frontend resizes before upload
const MAX_CANDIDATES = 40;
const SHOWN_PICKS = 10; // shown at once, with streaming info
const MAX_PICKS = 20; // ranked per search; the rest wait behind the Shuffle button
const MAX_EXCLUDE = 120;

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
  exclude: Set<string>; // candidate keys already shown, when shuffling
  region?: string; // ISO country code for streaming providers
  services: Set<string>; // service keys (see serviceKey); empty = any service
  partner?: { vibe?: string; music?: string }; // the other person, in "two of us" mode
  image?: { media_type: "image/jpeg" | "image/png" | "image/webp"; data: string };
}

// ---------- Claude step 1: turn fuzzy input into a search profile ----------

function partnerText(p: { vibe?: string; music?: string }): string {
  return [p.vibe && `vibe: ${p.vibe}`, p.music && `music: ${p.music}`].filter(Boolean).join("; ");
}

const ProfileSchema = z.object({
  summary: z.string().describe("One evocative sentence describing the viewing mood the user is after."),
  folder_name: z.string().describe("A 2-4 word, title-case name for a folder of picks like these, evocative rather than literal, e.g. 'Florida Noir', 'Rainy Sunday Chills', 'Lana Del Rey Summer'."),
  music_feel: z.string().describe("If music was given: what it feels like, not facts about the artist. Mood, energy, themes and imagery, e.g. Eddie Vedder → 'raw, earnest, open roads, wilderness, solitude, searching for freedom'. Empty if no music."),
  must_match: z
    .array(z.string())
    .describe("Hard requirements the user stated or clearly implied, phrased as checkable facts: 'set in Florida', 'a heist plot', 'not an Indian film'. Moods and tones ('sunshine', 'cosy') are not requirements. Empty if there are none."),
  genres: z.array(z.enum(GENRE_NAMES)).describe("1-3 TMDB genres that best fit."),
  keywords: z.array(z.string()).describe("3-6 short TMDB-style keywords, e.g. 'heist', 'road trip', 'small town', 'neo-noir'."),
  year_from: z.number().int().nullable().describe("Earliest release year, or null for no limit."),
  year_to: z.number().int().nullable().describe("Latest release year, or null for no limit."),
  reference_titles: z
    .array(z.object({ title: z.string(), kind: z.enum(["movie", "tv"]) }))
    .describe("Up to 6 real titles per wanted kind that strongly match, each marked as a movie or a tv series. Lead with the most obvious, on-the-nose matches."),
});
type Profile = z.infer<typeof ProfileSchema>;

const PROFILE_SYSTEM = `You are a film and TV curator. A user describes what they want to watch using any mix of: a mood or vibe, music they're into, a room or setting (sometimes as a photo), an era, or a plot idea.
Translate that into a search profile for a movie and TV database.
- Music describes a feeling, not a filmography. First write music_feel (mood, energy, themes, imagery of the sound), then choose titles that feel like that. At most one reference title may be directly tied to the artist (a film they scored or soundtracked, appear in, or that is about them); the rest must be other films and series with the same feel. Example: Eddie Vedder → maybe Into the Wild, then Wild, Nomadland, The Way, not a list of his soundtracks.
- Photos: read the palette, decor, era cues, lighting, and atmosphere; treat it as the setting the user wants to feel.
- Eras: if the user picked decades, draw reference titles from those decades (spread across them if there are several). year_from/year_to are only used when no decades are picked: then use null unless the input clearly implies a period.
- Reference titles must be real films or series you are confident exist, and only of the kinds the user wants. Start with the titles a knowledgeable friend would name first (a show literally set where and how the user describes), then less obvious ones.
- Requirements: every reference title must meet every item in must_match (e.g. actually be set in the place named).
- Keywords: put the most specific first (a place, a profession, a setting), as each one is searched on its own.
- Two people: if a second person is described, find the overlap both will enjoy (blend the moods, don't just pick one side). Hard requirements from either person still apply, and the summary should speak to both.`;

async function buildProfile(client: Anthropic, req: RecommendRequest): Promise<Profile> {
  const parts: string[] = [];
  if (req.vibe) parts.push(`Vibe / mood: ${req.vibe}`);
  if (req.music) parts.push(`Music they're into: ${req.music}`);
  if (req.plot) parts.push(`Plot idea: ${req.plot}`);
  if (req.eras?.length) parts.push(`Eras: ${req.eras.join(", ")}`);
  if (req.image) parts.push("Setting / decor: see the attached photo.");
  if (req.partner) parts.push(`A second person is watching too. They want: ${partnerText(req.partner)}`);
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

// Cloudflare's free plan allows 50 outbound requests per search, so the lookups are budgeted.
// Any service, films and series, worst case: 4 keyword lookups + 2 × (6 title searches + 3 "similar"
// lists + 3 keyword queries + 2 genre queries) + 2 Claude calls + 10 streaming lookups = 44.
// "My services" also checks streaming for every ranked pick (20) and fetches the region's service
// list (2), so it searches a little less: 3 + 2 × (4 + 2 + 2 + 2) + 2 + 2 + 20 = 47.
type Budget = { keywords: number; refs: number; refRecs: number; keywordQueries: number };
const FULL_BUDGET: Budget = { keywords: 4, refs: 6, refRecs: 3, keywordQueries: 3 };
const FILTERED_BUDGET: Budget = { keywords: 3, refs: 4, refRecs: 2, keywordQueries: 2 };

// Discover filters that keep TMDB's broad queries to the user's services.
type ServiceFilter = { ids: number[]; region: string } | null;

async function gatherCandidates(env: Env, profile: Profile, ranges: YearRange[], kinds: Kind[], services: ServiceFilter): Promise<Candidate[]> {
  const budget = services ? FILTERED_BUDGET : FULL_BUDGET;
  type Page = { results: TmdbItem[] };
  const empty: Page = { results: [] };

  // Keyword IDs are shared between films and series, so look them up once.
  const keywordIds = (
    await Promise.all(
      profile.keywords.slice(0, budget.keywords).map(async (k) => {
        const r = await soft(tmdb<{ results: { id: number }[] }>(env, "/search/keyword", { query: k }), { results: [] });
        return r.results[0]?.id;
      }),
    )
  ).filter((id): id is number => id !== undefined);

  const inRange = (year: number | null) => !year || ranges.some((r) => (!r.from || year >= r.from) && (!r.to || year <= r.to));
  // One window covering every picked era; results are then filtered to the eras themselves.
  const span: YearRange = {
    from: ranges.some((r) => !r.from) ? null : Math.min(...ranges.map((r) => r.from!)),
    to: ranges.some((r) => !r.to) ? null : Math.max(...ranges.map((r) => r.to!)),
  };
  // Genre queries are the broadest, so give each era its own when there are only a couple.
  const genreRanges = ranges.length <= 2 ? ranges : [span];

  async function forKind(kind: Kind): Promise<Candidate[]> {
    const dateField = kind === "movie" ? "primary_release_date" : "first_air_date";
    const discover = (range: YearRange, params: Record<string, string>) => {
      const p: Record<string, string> = { include_adult: "false", ...params };
      if (services) {
        p.with_watch_providers = services.ids.join("|");
        p.watch_region = services.region;
        p.with_watch_monetization_types = "flatrate|free|ads";
      }
      if (range.from) p[`${dateField}.gte`] = `${range.from}-01-01`;
      if (range.to) p[`${dateField}.lte`] = `${range.to}-12-31`;
      return soft(tmdb<Page>(env, `/discover/${kind}`, p), empty).then((r) => r.results);
    };

    // 1. Claude's own suggestions, looked up by name: the most on-point source, so they always make the cut.
    const refs = profile.reference_titles.filter((r) => r.kind === kind).slice(0, budget.refs);
    const refHits = Promise.all(
      refs.map(({ title }) => soft(tmdb<Page>(env, `/search/${kind}`, { query: title, include_adult: "false" }), empty).then((r) => r.results[0])),
    );
    // 2. TMDB's "similar" lists for the first few of them.
    const refRecs = refHits.then((hits) =>
      Promise.all(hits.filter(Boolean).slice(0, budget.refRecs).map((h) =>
        soft(tmdb<Page>(env, `/${kind}/${h!.id}/recommendations`), empty).then((r) => r.results.slice(0, 10)))),
    );
    // Runs alongside the discover queries below; if those throw first, don't leave this rejection unhandled.
    refRecs.catch(() => {});

    // 3. Each keyword on its own, best-known titles first ("florida" alone, not "florida" OR "murder").
    const keywordLists = Promise.all(
      keywordIds.slice(0, budget.keywordQueries).map((id) =>
        discover(span, { with_keywords: String(id), sort_by: "vote_count.desc" }).then((r) => r.slice(0, 12))),
    );

    // 4. Highly rated titles in the right genres: the broadest net, so it fills in last.
    const genreIds = [...new Set(profile.genres.map((g) => (kind === "movie" ? GENRES[g] : TV_GENRES[g])).filter((id): id is number => id !== null))];
    const genreLists = genreIds.length
      ? Promise.all(genreRanges.map((r) =>
          discover(r, { with_genres: genreIds.join("|"), sort_by: "vote_average.desc", "vote_count.gte": kind === "movie" ? "150" : "80" })))
      : Promise.resolve([] as TmdbItem[][]);

    const [hits, recs, byKeyword, byGenre] = await Promise.all([refHits, refRecs, keywordLists, genreLists]);

    const seen = new Map<string, Candidate>();
    const add = (item: TmdbItem | undefined) => {
      const c = item && toCandidate(kind, item);
      if (c && !seen.has(c.key) && inRange(c.year)) seen.set(c.key, c);
    };
    hits.forEach(add);
    // Then take from the other lists in turn, so no single broad query crowds the rest out.
    const lists = [...byKeyword, ...recs, ...byGenre];
    for (let i = 0; i < Math.max(0, ...lists.map((l) => l.length)); i++) for (const l of lists) add(l[i]);
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
      check: z.string().describe("Before anything else: how this title meets each requirement, from what you know of it, e.g. 'Set in Miami, Florida'. Write 'none' if there are no requirements."),
      meets_requirements: z.boolean().describe("True only if you are confident it meets every requirement."),
      reason: z.string().describe("One sentence, under 25 words, on why it fits this specific request."),
    }),
  ),
});

async function rankCandidates(client: Anthropic, req: RecommendRequest, profile: Profile, candidates: Candidate[]) {
  const list = candidates
    .map((c) => `[${c.key}] ${c.title} (${KIND_LABEL[c.kind]}, ${c.year ?? "?"}, lang: ${c.language ?? "?"}) — ${c.overview.slice(0, 220)}`)
    .join("\n");

  const ask = [req.vibe && `Vibe: ${req.vibe}`, req.music && `Music: ${req.music}`, req.plot && `Plot: ${req.plot}`, req.eras?.length && `Eras: ${req.eras.join(", ")} (spread picks across them)`, req.image && "They also shared a photo of a setting.",
    req.partner && `Watching with a second person, who wants: ${partnerText(req.partner)}. Pick titles both will enjoy, and say why it works for both of them.`]
    .filter(Boolean)
    .join("\n");

  const response = await client.messages.parse({
    model: MODEL,
    max_tokens: 4000,
    system: `You are a film and TV curator picking what someone should watch tonight. Choose only from the candidate list, by id. When both films and series are wanted, mix them unless the user's words lean one way. Honour anything the user rules out, such as a country or language (each candidate's original language is given as an ISO code). Prefer a varied set (not all the same director or decade unless asked). Music describes a feeling: pick titles that feel like it, and include at most one title directly tied to the artist (one they made music for, appear in, or that is about them). Write reasons that speak to the user's own words.
Requirements are firm. Use your own knowledge of each title, not just the short summary given: a show about murders that is set in Philadelphia does not fit "set in Florida". Leave out anything you are not confident meets every requirement. A shorter list of accurate picks is far better than a full list with a wrong one.`,
    messages: [
      {
        role: "user",
        content: `What they asked for:\n${ask}\n\nYour read of it: ${profile.summary}${profile.music_feel ? `\nThe music feels like: ${profile.music_feel}` : ""}\nRequirements: ${profile.must_match.length ? profile.must_match.join("; ") : "none"}\n\nCandidates:\n${list}\n\nRank up to ${Math.min(MAX_PICKS, candidates.length)} that meet the requirements, best first. The first ${SHOWN_PICKS} are shown together, so make them a strong, varied set on their own.`,
      },
    ],
    output_config: { format: zodOutputFormat(PicksSchema) },
  });
  if (!response.parsed_output) throw new Error(`Could not rank results (stop_reason: ${response.stop_reason})`);

  const byKey = new Map(candidates.map((c) => [c.key, c]));
  // Drop any id the model didn't actually get from us.
  return response.parsed_output.picks
    .filter((p) => p.meets_requirements) // the model flagged its own misfit
    .map((p) => ({ item: byKey.get(p.id.replace(/^\[|\]$/g, "")), reason: p.reason }))
    .filter((p): p is { item: Candidate; reason: string } => !!p.item)
    .slice(0, MAX_PICKS);
}

// TMDB lists several entries for one service ("Netflix", "Netflix Standard with Ads", "Paramount+
// Amazon Channel"). Users pick the service once, so entries are grouped by their base name.
const SERVICE_SUFFIX = / (Amazon Channel|Apple TV Channel|Roku Premium Channel|Standard with Ads|Basic with Ads|with Ads|Premium Plus|Premium|Essential|Ad-Free)$/i;
function baseService(name: string): string {
  let n = name.trim();
  for (let prev = ""; prev !== n; ) { prev = n; n = n.replace(SERVICE_SUFFIX, "").trim(); }
  return n.replace(/ Plus$/i, "+"); // the brands' own spelling: "Paramount+", "Disney+"
}
// Matching key: TMDB spells some services both ways ("Paramount+" and "Paramount Plus").
function serviceKey(name: string): string {
  return baseService(name).toLowerCase().replace(/\+/g, " plus").replace(/\s+/g, " ").trim();
}

type ProviderEntry = { provider_id: number; provider_name: string; logo_path?: string; display_priority?: number; display_priorities?: Record<string, number> };

// Every service available in a region, films and series combined, most popular first.
async function regionProviders(env: Env, region: string) {
  const lists = await Promise.all((["movie", "tv"] as const).map((k) =>
    soft(tmdb<{ results: ProviderEntry[] }>(env, `/watch/providers/${k}`, { watch_region: region }), { results: [] })));
  const byBase = new Map<string, { key: string; name: string; logo: string | null; ids: Set<number>; priority: number }>();
  const entries = lists.flatMap((l) => l.results)
    .map((p) => ({ ...p, priority: p.display_priorities?.[region] ?? p.display_priority ?? 999 }))
    .sort((a, b) => a.priority - b.priority);
  for (const p of entries) {
    const key = serviceKey(p.provider_name);
    const name = baseService(p.provider_name);
    const priority = p.priority;
    // The most popular spelling names the group.
    const entry = byBase.get(key) ?? { key, name, logo: null, ids: new Set<number>(), priority };
    entry.ids.add(p.provider_id);
    entry.priority = Math.min(entry.priority, priority);
    // Prefer the plain entry's logo ("Netflix", not "Netflix Standard with Ads").
    if (p.logo_path && (!entry.logo || p.provider_name === name)) entry.logo = LOGO_BASE + p.logo_path;
    byBase.set(key, entry);
  }
  return [...byBase.values()].sort((a, b) => a.priority - b.priority);
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
    .map((p) => ({ name: baseService(p.provider_name), base: serviceKey(p.provider_name), logo: p.logo_path ? LOGO_BASE + p.logo_path : null }));
  return { services, link: here?.link ?? null };
}

type Watch = Awaited<ReturnType<typeof watchProviders>>;

// The user's own services first, then the rest; five at most on a card.
function cardServices(watch: Watch, mine: Set<string>) {
  const ordered = [...watch.services.filter((s) => mine.has(s.base)), ...watch.services.filter((s) => !mine.has(s.base))];
  const seen = new Set<string>();
  return ordered.filter((s) => !seen.has(s.base) && seen.add(s.base)).slice(0, 5).map(({ name, logo }) => ({ name, logo }));
}

function toResult(item: Candidate, reason: string, watch?: Watch, mine: Set<string> = new Set()) {
  return {
    key: item.key,
    id: item.id,
    kind: item.kind,
    title: item.title,
    year: item.year ? String(item.year) : null,
    rating: item.rating,
    overview: item.overview,
    poster: item.posterPath ? POSTER_BASE + item.posterPath : null,
    url: `https://www.themoviedb.org/${item.kind}/${item.id}`,
    reason,
    // null = not looked up yet; the page asks /api/watch when it shows the card.
    streaming: watch ? cardServices(watch, mine) : null,
    watchUrl: watch ? watch.link : null,
  };
}

// ---------- Share card posters ----------

// The share card draws the poster onto a canvas, which browsers only allow for images served with
// CORS headers. Only TMDB poster paths are accepted, so this can't be used as an open proxy.
async function posterProxy(route: string, cors: Record<string, string>): Promise<Response> {
  const m = route.match(/^\/api\/poster\/(w342|w500|w780)\/([A-Za-z0-9_-]{8,64}\.(?:jpg|png))$/);
  if (!m) return json({ error: "Not found" }, 404, cors);
  const res = await fetch(`https://image.tmdb.org/t/p/${m[1]}/${m[2]}`, { cf: { cacheTtl: 604800, cacheEverything: true } });
  if (!res.ok) return json({ error: "Poster not found" }, 404, cors);
  return new Response(res.body, {
    headers: { ...cors, "Content-Type": res.headers.get("Content-Type") ?? "image/jpeg", "Cache-Control": "public, max-age=604800" },
  });
}

// ---------- Rate limits ----------

// Returns the message to show when this search is over a limit, or null if it may go ahead.
async function checkLimits(request: Request, env: Env): Promise<string | null> {
  const who = request.headers.get("CF-Connecting-IP") ?? "unknown";
  // Missing bindings (e.g. an old deploy) shouldn't take the site down: skip that check.
  if (env.SEARCH_BURST && !(await env.SEARCH_BURST.limit({ key: who })).success) {
    return "Easy there, projectionist. That's a lot of shuffling: give it a minute and try again.";
  }
  if (env.LIMITS) {
    const setting = (v: string | undefined, fallback: number) => (Number(v) > 0 ? Number(v) : fallback);
    const result = await env.LIMITS.get(env.LIMITS.idFromName("daily"))
      .take(who, setting(env.DAILY_SEARCHES_PER_PERSON, DAILY_PER_PERSON), setting(env.DAILY_SEARCHES_TOTAL, DAILY_TOTAL));
    if (!result.ok) {
      return result.reason === "person"
        ? "That's all your searches for today. Your bubble refills tomorrow."
        : "Shuffle Bubble is resting for today. Come back tomorrow.";
    }
  }
  return null;
}

// ---------- HTTP plumbing ----------

function corsHeaders(request: Request, env: Env): Record<string, string> {
  const origin = request.headers.get("Origin") ?? "";
  const allowed = env.ALLOWED_ORIGINS.split(",").map((s) => s.trim());
  return allowed.includes(origin)
    ? { "Access-Control-Allow-Origin": origin, "Access-Control-Allow-Methods": "GET, POST, OPTIONS", "Access-Control-Allow-Headers": "Content-Type", Vary: "Origin" }
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

function parseRegion(body: any): string {
  return /^[A-Z]{2}$/.test(body?.region ?? "") ? body.region : "US";
}

// Streaming info for cards the page shows later (the Shuffle reserve), kept out of the main
// search so that one stays under the free plan's 50 outbound requests.
async function handleWatch(env: Env, body: any) {
  const items = (Array.isArray(body?.items) ? body.items : [])
    .filter((i: any) => (i?.kind === "movie" || i?.kind === "tv") && Number.isInteger(i?.id) && i.id > 0)
    .slice(0, SHOWN_PICKS) as { kind: Kind; id: number }[];
  const region = parseRegion(body);
  const entries = await Promise.all(
    items.map(async (i) => {
      const w = await watchProviders(env, i.kind, i.id, region);
      return [`${i.kind}-${i.id}`, { streaming: cardServices(w, new Set()), watchUrl: w.link }] as const;
    }),
  );
  return { watch: Object.fromEntries(entries) };
}

function parseRequest(body: any): RecommendRequest | string {
  const req: RecommendRequest = {
    vibe: clean(body?.vibe),
    music: clean(body?.music),
    plot: clean(body?.plot),
    eras: parseEras(body),
    kinds: parseKinds(body),
    region: parseRegion(body),
    services: new Set(
      (Array.isArray(body?.services) ? body.services : [])
        .filter((n: unknown): n is string => typeof n === "string" && n.trim().length > 0 && n.length <= 60)
        .map((n: string) => serviceKey(n))
        .slice(0, 30),
    ),
    partner: (() => {
      const p = { vibe: clean(body?.partner?.vibe), music: clean(body?.partner?.music) };
      return p.vibe || p.music ? p : undefined;
    })(),
    exclude: new Set(
      (Array.isArray(body?.exclude) ? body.exclude : [])
        .filter((k: unknown): k is string => typeof k === "string" && /^(movie|tv)-\d+$/.test(k))
        .slice(0, MAX_EXCLUDE),
    ),
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
    const route = url.pathname;
    if (request.method === "GET" && route.startsWith("/api/poster/")) {
      // The share card loads posters with crossorigin, so the browser sends the page's Origin.
      if (!cors["Access-Control-Allow-Origin"]) return json({ error: "Origin not allowed" }, 403, cors);
      return posterProxy(route, cors);
    }
    if (request.method !== "POST" || !["/api/recommend", "/api/watch", "/api/providers"].includes(route)) return json({ error: "Not found" }, 404, cors);
    // Only the allowed sites may spend your API credits.
    if (!cors["Access-Control-Allow-Origin"]) return json({ error: "Origin not allowed" }, 403, cors);

    let body: unknown;
    try {
      body = await request.json();
    } catch {
      return json({ error: "Invalid JSON" }, 400, cors);
    }
    if (!env.TMDB_TOKEN) return json({ error: "Server is missing its TMDB_TOKEN secret." }, 500, cors);
    if (route === "/api/providers") {
      try {
        const list = await regionProviders(env, parseRegion(body));
        return json({ providers: list.slice(0, 30).map(({ name, logo }) => ({ name, logo })) }, 200, cors);
      } catch (err) {
        if (err instanceof TmdbAuthError) return json({ error: "The TMDB token was rejected. Check TMDB_TOKEN." }, 500, cors);
        throw err;
      }
    }
    if (route === "/api/watch") {
      try {
        return json(await handleWatch(env, body), 200, cors);
      } catch (err) {
        if (err instanceof TmdbAuthError) return json({ error: "The TMDB token was rejected. Check TMDB_TOKEN." }, 500, cors);
        throw err;
      }
    }

    const req = parseRequest(body);
    if (typeof req === "string") return json({ error: req }, 400, cors);

    // Only searches cost money, so only they are limited.
    const limited = await checkLimits(request, env);
    if (limited) return json({ error: limited }, 429, cors);

    if (!env.ANTHROPIC_API_KEY) return json({ error: "Server is missing its ANTHROPIC_API_KEY secret." }, 500, cors);

    const client = new Anthropic({ apiKey: env.ANTHROPIC_API_KEY });
    try {
      const profile = await buildProfile(client, req);
      // "My services": map the picked names to TMDB's provider ids for this region.
      let filter: ServiceFilter = null;
      if (req.services.size) {
        const ids = (await regionProviders(env, req.region!)).filter((p) => req.services.has(p.key)).flatMap((p) => [...p.ids]);
        if (ids.length) filter = { ids, region: req.region! };
      }

      const candidates = (await gatherCandidates(env, profile, yearRanges(req, profile), req.kinds, filter))
        .filter((c) => !req.exclude.has(c.key)); // already shown before a shuffle
      if (!candidates.length) return json({ profile, results: [], reserve: [] }, 200, cors);

      const picks = await rankCandidates(client, req, profile, candidates);
      if (filter) {
        // Title searches and "similar" lists aren't filtered by service, so check every pick and keep
        // only what's on the user's services. The reserve comes back with streaming info filled in.
        const checked = await Promise.all(picks.map(async (p) => ({ ...p, watch: await watchProviders(env, p.item.kind, p.item.id, req.region!) })));
        const available = checked.filter((p) => p.watch.services.some((s) => req.services.has(s.base)));
        const all = available.map(({ item, reason, watch }) => toResult(item, reason, watch, req.services));
        return json({ profile, results: all.slice(0, SHOWN_PICKS), reserve: all.slice(SHOWN_PICKS) }, 200, cors);
      }
      const results = await Promise.all(
        picks.slice(0, SHOWN_PICKS).map(async ({ item, reason }) => toResult(item, reason, await watchProviders(env, item.kind, item.id, req.region!))),
      );
      const reserve = picks.slice(SHOWN_PICKS).map(({ item, reason }) => toResult(item, reason));
      return json({ profile, results, reserve }, 200, cors);
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
