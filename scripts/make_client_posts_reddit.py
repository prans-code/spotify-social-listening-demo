from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

import praw


BASE_DIR = Path(__file__).resolve().parents[1]
CLIENTS_DIR = BASE_DIR / "data" / "clients"


def load_sources(client_dir: Path) -> dict[str, Any]:
    p = client_dir / "sources.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Create it first.")
    return json.loads(p.read_text(encoding="utf-8"))


def get_reddit() -> praw.Reddit:
    cid = os.getenv("REDDIT_CLIENT_ID")
    csec = os.getenv("REDDIT_CLIENT_SECRET")
    ua = os.getenv("REDDIT_USER_AGENT", "social-listening-demo/1.0 by u/yourname")
    if not cid or not csec:
        raise RuntimeError("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in your environment.")
    return praw.Reddit(client_id=cid, client_secret=csec, user_agent=ua, check_for_async=False)


def vader_label(compound: float) -> str:
    # classic VADER thresholds
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def collect_reddit(client: str, cfg: dict[str, Any]) -> pd.DataFrame:
    reddit = get_reddit()
    analyzer = SentimentIntensityAnalyzer()

    subreddits = cfg.get("subreddits", ["all"])
    queries = cfg.get("queries", [])
    limit_per_query = int(cfg.get("limit_per_query", 200))
    time_filter = cfg.get("time_filter", "year")  # hour/day/week/month/year/all

    rows = []
    seen = set()

    for sr in subreddits:
        subreddit = reddit.subreddit(sr)
        for q in queries:
            # PRAW search supports time_filter + sorting
            try:
                it = subreddit.search(q, sort="new", time_filter=time_filter, limit=limit_per_query)
                for s in it:
                    sid = getattr(s, "id", None)
                    if not sid or sid in seen:
                        continue
                    seen.add(sid)

                    title = getattr(s, "title", "") or ""
                    body = getattr(s, "selftext", "") or ""
                    text = (title + "\n\n" + body).strip() if body else title.strip()
                    if not text:
                        continue

                    score = analyzer.polarity_scores(text)["compound"]
                    rows.append(
                        {
                            "id": sid,
                            "client": client,
                            "platform": "reddit",
                            "subreddit": getattr(s, "subreddit", sr).__str__(),
                            "created_utc": int(getattr(s, "created_utc", 0)),
                            "text": text,
                            "sentiment_score": float(score),
                            "sentiment_label": vader_label(float(score)),
                            "sentiment_confidence": float(abs(score)),
                            "url": f"https://www.reddit.com{getattr(s, 'permalink', '')}",
                            "query": q,
                        }
                    )
            except Exception as e:
                print(f"[WARN] subreddit={sr} query={q} failed: {e}")

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    # Convert created_utc to datetime for your dashboard (it can handle unix too, but this is nicer)
    df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    df = df.dropna(subset=["created_utc"])
    df = df.sort_values("created_utc").reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", required=True, help="Client folder name under data/clients/")
    args = ap.parse_args()

    client_dir = CLIENTS_DIR / args.client
    if not client_dir.exists():
        raise FileNotFoundError(f"Client folder not found: {client_dir}")

    sources = load_sources(client_dir)
    platforms = sources.get("platforms", ["reddit"])

    frames = []
    if "reddit" in platforms:
        frames.append(collect_reddit(args.client, sources.get("reddit", {})))

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out) == 0:
        raise RuntimeError("No data collected. Check queries/subreddits and API credentials.")

    # Minimal clean_text column (optional but helpful)
    out["clean_text"] = (
        out["text"].astype(str).str.lower()
        .str.replace(r"http\S+", " ", regex=True)
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    out_path = client_dir / "posts.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Wrote {len(out):,} rows to {out_path}")


if __name__ == "__main__":
    main()