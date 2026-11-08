# Shuffle Bubble

Find your flick. Get in your bubble. Film and series recommendations from a vibe, the music you're into, a photo of your room or a place, an era, or a plot idea.

Voice, colours, type and logo rules: see [BRAND.md](BRAND.md).

## How it works

```
docs/index.html (GitHub Pages)  ──POST /api/recommend──▶  worker/ (Cloudflare Worker)
                                                            1. Claude Haiku: input (+ photo) → search profile
                                                               (genres, keywords, years, reference films)
                                                            2. TMDB: search, recommendations, discover → ~40 real films
                                                            3. Claude Haiku: pick the best 6 from that list + say why
                                                            4. TMDB: where each one is streaming
```

Claude only picks from films TMDB actually returned, so it can't invent titles. Your API keys live only in the Worker, never in the page.

## Cost

- TMDB, Cloudflare Workers (100k requests/day) and GitHub Pages: free for non-commercial use.
- Claude API: pay-as-you-go, roughly half a cent per recommendation with Haiku. Set a monthly spend limit in the Anthropic console.

## Setup

### 1. Get keys
- **Anthropic:** create an API key at https://console.anthropic.com, add a few dollars of credit, and set a spend limit.
- **TMDB:** create an account, then request an API key at https://www.themoviedb.org/settings/api. Copy the **API Read Access Token** (the long one).

### 2. Deploy the backend
```bash
cd worker
npm install
npx wrangler login                        # free Cloudflare account
npx wrangler secret put ANTHROPIC_API_KEY
npx wrangler secret put TMDB_TOKEN
npx wrangler deploy                        # prints your URL, e.g. https://telerama-api.<you>.workers.dev
```

If you host the page anywhere other than `https://loudnoises1909-cloud.github.io`, add that origin to `ALLOWED_ORIGINS` in `worker/wrangler.toml` and redeploy. Only listed sites can call the API (and spend your credits).

### 3. Point the page at it
In `docs/index.html`, set `API_URL` to the Worker URL from step 2.

### 4. Publish the page
On GitHub: **Settings → Pages → Deploy from a branch →** pick the branch and the `/docs` folder. The site will be at `https://loudnoises1909-cloud.github.io/Telerama/`.

## Limits

Searches are the only part that costs money (Claude Opus reads the request and Claude Haiku ranks the picks: roughly 3-4 cents a search), so they're capped:

- **5 per minute** per visitor (Cloudflare's rate limiter, `SEARCH_BURST` in `worker/wrangler.toml`).
- **15 per visitor per day** and **100 per day in total** (`DAILY_SEARCHES_PER_PERSON` and `DAILY_SEARCHES_TOTAL` in `worker/wrangler.toml`, counted in a Durable Object). Days reset at midnight UTC.

Raise the numbers in `wrangler.toml` and push to change them. Keep a spend limit (or auto-reload off) in the Anthropic console as the final backstop.

## Local development
```bash
cd worker
printf 'ANTHROPIC_API_KEY=sk-ant-...\nTMDB_TOKEN=eyJ...\n' > .dev.vars   # git-ignored
npx wrangler dev                                                       # http://localhost:8787
# in another terminal, from the repo root:
python3 -m http.server 8000 --directory docs
# open http://localhost:8000/?api=http://localhost:8787
```

## Ideas for next steps
- Plot search with embeddings over TMDB overviews
- "More like this" / thumbs up–down feedback
- Saved watchlists
