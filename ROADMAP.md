# Roadmap

## Now: friends test

- Share the link with a few friends and collect their thoughts on pick quality and speed.
- Watch real cost per search on the Anthropic console's Usage page (estimate: 3–4 cents with Opus + Haiku 5.5).
- Tune `PROFILE_EFFORT` in `worker/wrangler.toml` if picks feel shallow (raise) or searches feel slow (keep at low).

## Next: Coming soon

A page that keeps people ahead of new releases, sorted by how much *they* will care. The search answers "what now?"; this answers "what's coming?", and gives people a reason to come back every week.

### What it does

- **Upcoming films and series** for the user's country, refreshed daily.
- **Sorted for the user:** *Must-see*, *Worth a look* and *If you're curious*, rated by Claude against their taste (the genre and tone of picks they've saved in their bubbles), then grouped by genre.
- **♡ to save** into an automatic **"Coming soon"** bubble, ordered by must-see level.
- **"Now streaming" badge** on saved titles once they arrive on the user's services, with where to watch.

### Data

- TMDB, which we already use:
  - upcoming films and their digital release dates per country (`/movie/upcoming`, `release_dates` type 4);
  - new series by premiere date (`/discover/tv` with `first_air_date.gte`);
  - streaming originals before launch via `with_networks` (e.g. Netflix 213, Apple TV+ 2552, Prime Video 1024, Disney+ 2739, Max 3186, Hulu 453; verify these IDs when building).
- Gap: TMDB doesn't reliably say when an *older* title arrives on a service. That needs licensed data (e.g. JustWatch). Originals and new releases cover most of what people want.

### Cost and build

- Fetch the list once a day per country and share it with everyone (store it in the existing Durable Object or a small KV namespace).
- Personal sorting with Claude Haiku 5.5: a fraction of a cent per person per week. Cheaper than a search; doesn't count toward search limits.
- About the size of the folders feature. Works without accounts (saved on the device, like bubbles).

## Later

- **Share a bubble** as a link or story card ("my Florida Noir list").
- **Accounts**, so bubbles and Coming soon follow people across devices; then **reminders** by email or notification ("Bad Monkey season 2 drops Friday").
- **"More like this"** on any pick.
- **Own domain** (check shufflebubble.com) and move the site from GitHub Pages to Cloudflare Pages.
- **Before charging:** USPTO trademark search for "Shuffle Bubble", TMDB commercial licence, privacy policy and terms, pricing (subscription or free tier plus paid; see the cost notes in the README).
