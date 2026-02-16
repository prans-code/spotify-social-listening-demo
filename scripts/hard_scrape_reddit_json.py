from __future__ import annotations

import argparse
import time
import re
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


BASE_DIR = Path(__file__).resolve().parents[1]
CLIENTS_DIR = BASE_DIR / "data" / "clients"

UA = "social-listening-demo/1.0 (no-auth json) by chatgpt-user"


def vader_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def clean_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fetch_posts(subreddit: str, query: str, limit: int, t: str) -> list[dict]:
    """
    Uses Reddit's public JSON endpoint (no client_id).
    Might rate-limit (429). We handle basic retries.
    """
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = {
        "q": query,
        "restrict_sr": 1,
        "sort": "new",
        "t": t,          # hour/day/week/month/year/all
        "limit": min(limit, 100),
    }

    headers = {"User-Agent": UA}
    for attempt in range(6):
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code == 200:
            data = r.json()
            children = data.get("data", {}).get("children", []) or []
            out = []
            for ch in children:
                d = ch.get("data", {}) or {}
                out.append(d)
            return out

        # backoff on rate limit / forbidden
        if r.status_code in (429, 403):
            time.sleep(2 + attempt * 2)
            continue

        # other errors: wait a bit and retry
        time.sleep(1 + attempt)
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", required=True, help="Client folder under data/clients/")
    ap.add_argument("--subreddits", required=True, help='Comma list, e.g. "spotify,technology"')
    ap.add_argument("--queries", required=True, help='Comma list, e.g. "spotify app,spotify premium"')
    ap.add_argument("--limit", type=int, default=80, help="Posts per subreddit per query (max 100 per call)")
    ap.add_argument("--time", default="year", choices=["hour","day","week","month","year","all"])
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--sleep", type=float, default=1.2, help="Seconds between requests (avoid rate limiting)")
    args = ap.parse_args()

    client_dir = CLIENTS_DIR / args.client
    client_dir.mkdir(parents=True, exist_ok=True)

    subreddits = [s.strip() for s in args.subreddits.split(",") if s.strip()]
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]

    analyzer = SentimentIntensityAnalyzer()
    rows = []
    seen = set()

    for sr in subreddits:
        for q in queries:
            posts = fetch_posts(sr, q, limit=args.limit, t=args.time)
            time.sleep(args.sleep)

            for p in posts:
                pid = p.get("id")
                if not pid or pid in seen:
                    continue
                seen.add(pid)

                title = (p.get("title") or "").strip()
                body = (p.get("selftext") or "").strip()
                text = (title + "\n\n" + body).strip() if body else title
                if not text:
                    continue

                created = p.get("created_utc")
                score = analyzer.polarity_scores(text)["compound"]

                rows.append({
                    "id": pid,
                    "client": args.client,
                    "platform": "reddit",
                    "subreddit": p.get("subreddit") or sr,
                    "created_utc": pd.to_datetime(created, unit="s", errors="coerce") if created else pd.Timestamp.utcnow(),
                    "text": text,
                    "clean_text": clean_text(text),
                    "sentiment_score": float(score),
                    "sentiment_label": vader_label(float(score)),
                    "sentiment_confidence": float(abs(score)),
                    "url": "https://www.reddit.com" + (p.get("permalink") or ""),
                    "query": q,
                })

    df_new = pd.DataFrame(rows)
    if len(df_new) == 0:
        raise RuntimeError("No Reddit posts collected. Try broader queries/subreddits or increase --time all.")

    df_new = df_new.dropna(subset=["created_utc"]).sort_values("created_utc").reset_index(drop=True)

    out_path = client_dir / "posts.csv"
    if args.append and out_path.exists():
        df_old = pd.read_csv(out_path)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
        # de-dupe by platform+id if present
        if "id" in df_out.columns:
            df_out = df_out.drop_duplicates(subset=["platform", "id"], keep="first")
    else:
        df_out = df_new

    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Wrote {len(df_new):,} new rows. Total in {out_path}: {len(df_out):,}")


if __name__ == "__main__":
    main()
