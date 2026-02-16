# Data Schema (posts.csv)

Minimum required columns:
- `created_utc` (datetime, unix seconds/ms, or ISO string)
- `text` (string)

Recommended columns:
- `platform` (e.g., `reddit`, `youtube`)
- `subreddit` (for reddit rows; `unknown` otherwise)
- `query` (for youtube rows, search query used)
- `url` (source URL if available)
- `sentiment_label` (`positive` / `neutral` / `negative`)
- `sentiment_score` (float, optional)
- `sentiment_confidence` (float 0–1, optional)
- `clean_text` (preprocessed text used for keyword analysis)
- `source` (derived field used in UI; subreddit for reddit, query for youtube)

Notes:
- The app normalizes timestamps and fills missing optional fields.
- If YouTube timestamps were collected as “now”, you can spread them for demo purposes using your fixer script.
