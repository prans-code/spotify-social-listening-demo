
---

### `docs/DEMO.md`
```md
# Demo Walkthrough (3–5 minutes)

## Goal
Show that the dashboard works *per client* and feels interactive.

## Script
1) Open the app and select **Client = Spotify**
2) Go to **Overview**
   - Highlight KPIs
   - Show sentiment mix + platform mix
3) Go to **Trends**
   - Switch aggregation Daily → Weekly
   - Show net sentiment line
4) Go to **Drivers**
   - Show negative drivers (TF-IDF)
   - Select 2–3 keywords to activate “driver keyword filter”
5) Go to **Explorer**
   - Confirm only matching posts appear
   - Download filtered CSV
6) Go to **Compare periods**
   - Pick two date ranges and show sentiment shift
7) Switch **Client = Netflix** and repeat one quick insight (30s)

## Backup Plan
If the live demo is slow:
- Use Overview + Explorer only
- Show exports and keyword filter to prove interactivity

# Roadmap (Robust Version)

## Data & Coverage
- Add more platforms: X/Twitter (API-based), Facebook public pages (where allowed), news, forums
- Better YouTube metadata: video publish date, channel, like counts
- Robust Reddit collection: better rate limiting, retries, and richer metadata

## Analytics
- Aspect-based sentiment (pricing, ads, UX, content, reliability)
- Topic modeling upgrades (BERTopic / embeddings)
- Spike detection + alerting (sudden negative spikes)
- Share of voice (brand vs competitors)

## Product UX
- Saved views per client
- “Insights” auto-summary per week/month
- Report export: PDF / PPT
