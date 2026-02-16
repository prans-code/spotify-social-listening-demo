from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
CLIENTS_DIR = BASE_DIR / "data" / "clients"


def parse_mixed_datetime(s: pd.Series) -> pd.Series:
    # Handles unix seconds/ms even if stored as strings, plus ISO strings.
    s0 = s.copy()
    sn = pd.to_numeric(s0.astype(str).str.strip(), errors="coerce")
    dt_num = pd.Series(pd.NaT, index=s0.index)

    if sn.notna().any():
        vals = sn.dropna().astype(float)
        med = float(np.median(vals)) if len(vals) else 0.0
        if med > 1e12:
            dt_num = pd.to_datetime(sn, unit="ms", errors="coerce")
        elif med > 1e9:
            dt_num = pd.to_datetime(sn, unit="s", errors="coerce")
        else:
            dt_num = pd.to_datetime(sn, errors="coerce")

    dt_str = pd.to_datetime(s0, errors="coerce")
    return dt_num.fillna(dt_str)


def infer_platform(df: pd.DataFrame) -> pd.DataFrame:
    if "platform" not in df.columns:
        df["platform"] = ""

    df["platform"] = df["platform"].fillna("").astype(str).str.strip().str.lower()

    url = df["url"].fillna("").astype(str).str.lower() if "url" in df.columns else pd.Series("", index=df.index)
    sub = df["subreddit"].fillna("").astype(str).str.lower() if "subreddit" in df.columns else pd.Series("", index=df.index)

    blank = df["platform"].isin(["", "nan", "none", "unknown"])

    df.loc[blank & url.str.contains("reddit.com", na=False), "platform"] = "reddit"
    df.loc[blank & url.str.contains("youtube.com|youtu.be", na=False), "platform"] = "youtube"
    df.loc[blank & (sub == "youtube"), "platform"] = "youtube"

    df.loc[df["platform"].isin(["", "nan", "none"]), "platform"] = "unknown"
    return df


def normalize_subreddit(df: pd.DataFrame) -> pd.DataFrame:
    if "subreddit" not in df.columns:
        df["subreddit"] = "unknown"
    df["subreddit"] = (
        df["subreddit"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.replace(r"^r/", "", regex=True)
        .str.lower()
    )
    df.loc[df["subreddit"].isin(["", "nan", "none"]), "subreddit"] = "unknown"
    return df


def build_source(df: pd.DataFrame) -> pd.DataFrame:
    # Stable "source": subreddit for reddit, query for youtube (if query exists), else fallback.
    df["platform"] = df["platform"].astype(str).str.strip().str.lower()

    if "query" in df.columns:
        q = df["query"].fillna("").astype(str).str.strip().str.lower()
        q = q.replace({"nan": "", "none": ""})
        df["source"] = np.where(df["platform"] == "youtube", q, df["subreddit"])
        df.loc[(df["platform"] == "youtube") & (df["source"] == ""), "source"] = "youtube"
    else:
        df["source"] = df["subreddit"]

    df.loc[df["source"].isin(["", "nan", "none"]), "source"] = "unknown"
    return df


def normalize_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if "sentiment_label" not in df.columns:
        df["sentiment_label"] = "unlabeled"

    df["sentiment_label"] = (
        df["sentiment_label"]
        .fillna("unlabeled")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # map common variants
    df["sentiment_label"] = df["sentiment_label"].replace(
        {
            "pos": "positive",
            "neg": "negative",
            "neu": "neutral",
            "positive ": "positive",
            "negative ": "negative",
            "neutral ": "neutral",
        }
    )

    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = np.nan
    if "sentiment_confidence" not in df.columns:
        df["sentiment_confidence"] = np.nan

    return df


def clean_text_col(df: pd.DataFrame) -> pd.DataFrame:
    if "clean_text" not in df.columns:
        df["clean_text"] = (
            df["text"].astype(str).str.lower()
            .str.replace(r"http\S+", " ", regex=True)
            .str.replace(r"[^a-z0-9\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    return df


def optionally_spread_dates_for_demo(df: pd.DataFrame) -> pd.DataFrame:
    """
    If almost everything is on a single day (common with hard-scraped YouTube),
    spread across last 180 days so Trends works like legacy.
    """
    if len(df) < 50:
        return df

    d = df["created_utc"].dt.date
    if d.nunique() <= 2:
        base = pd.Timestamp.utcnow().normalize()
        # spread deterministically (no randomness needed)
        offsets = np.linspace(0, 180, num=len(df), dtype=int)
        df = df.sort_values("created_utc").reset_index(drop=True)
        df["created_utc"] = [base - pd.Timedelta(days=int(x)) for x in offsets]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", required=True)
    ap.add_argument("--demo_spread_dates", action="store_true", help="Make Trends useful if dates are 1-day only")
    args = ap.parse_args()

    path = CLIENTS_DIR / args.client / "posts.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    df = pd.read_csv(path)

    # required columns
    if "text" not in df.columns:
        for alt in ["post", "content", "body", "comment"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "text"})
                break
    if "text" not in df.columns:
        raise RuntimeError("posts.csv must contain a 'text' column.")

    if "created_utc" not in df.columns:
        for alt in ["created", "date", "timestamp", "time"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "created_utc"})
                break
    if "created_utc" not in df.columns:
        df["created_utc"] = pd.Timestamp.utcnow()

    # datetime
    df["created_utc"] = parse_mixed_datetime(df["created_utc"])
    # don’t drop; fill missing so rows don’t vanish
    df.loc[df["created_utc"].isna(), "created_utc"] = pd.Timestamp.utcnow()

    # platform/subreddit/source/sentiment/clean_text
    df = infer_platform(df)
    df = normalize_subreddit(df)
    df = build_source(df)
    df = normalize_sentiment(df)
    df = clean_text_col(df)

    if args.demo_spread_dates:
        df = optionally_spread_dates_for_demo(df)

    df = df.sort_values("created_utc").reset_index(drop=True)
    df.to_csv(path, index=False, encoding="utf-8")

    # summary
    print(f"✅ Fixed {args.client}: {path}")
    print("Rows:", len(df))
    print("Platforms:\n", df["platform"].value_counts().head(10))
    print("Subreddits (top):\n", df["subreddit"].value_counts().head(10))
    print("Sources (top):\n", df["source"].value_counts().head(10))
    print("Date range:", df["created_utc"].min(), "→", df["created_utc"].max())


if __name__ == "__main__":
    main()
