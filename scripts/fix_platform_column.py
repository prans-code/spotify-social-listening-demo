from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
CLIENTS_DIR = BASE_DIR / "data" / "clients"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", required=True)
    args = ap.parse_args()

    path = CLIENTS_DIR / args.client / "posts.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    # Ensure columns exist
    if "platform" not in df.columns:
        df["platform"] = ""
    if "url" not in df.columns:
        df["url"] = ""
    if "subreddit" not in df.columns:
        df["subreddit"] = ""

    url = df["url"].astype(str).str.lower()
    src = df["subreddit"].astype(str).str.lower()

    mask_reddit = url.str.contains("reddit.com", na=False) | src.str.contains(r"^r/", na=False)
    mask_youtube = url.str.contains("youtube.com|youtu.be", na=False) | (src == "youtube")

    # Assign platforms
    df.loc[mask_reddit, "platform"] = "reddit"
    df.loc[mask_youtube, "platform"] = "youtube"

    # Clean platform strings
    df["platform"] = df["platform"].fillna("unknown").astype(str).str.strip().str.lower()
    df.to_csv(path, index=False, encoding="utf-8")

    print("✅ Updated:", path)
    print(df["platform"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
