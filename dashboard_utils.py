from __future__ import annotations

from pathlib import Path
import re
import json

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# -----------------------------
# Paths / constants
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CLIENTS_DIR = DATA_DIR / "clients"
LEGACY_DEMO_FILE = DATA_DIR / "sentiment_reddit_spotify.csv"

DEFAULT_BRANDING = {
    "display_name": "Client",
    "tagline": "Sentiment-first social listening dashboard",
    "logo_path": None,      # relative to client folder
    "logo_url": None,       # optional (not used by default)
    "primary_color": "#4C78A8",
    "report_footer": "",
}


# -----------------------------
# Client listing / loading
# -----------------------------
def list_clients() -> list[str]:
    """
    List available client folders under data/clients/<ClientName>/posts.csv
    (Not cached to avoid stale results when you add/remove clients.)
    """
    if not CLIENTS_DIR.exists():
        return []
    out = []
    for p in CLIENTS_DIR.glob("*"):
        if p.is_dir() and (p / "posts.csv").exists():
            out.append(p.name)
    return sorted(out)


def read_csv_any(path: Path) -> pd.DataFrame:
    """
    Cache-busted CSV reader (reloads when the file changes on disk).
    """
    mtime = path.stat().st_mtime
    return _read_csv_cached(str(path), mtime)

@st.cache_data(show_spinner=False)
def _read_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path_str)


@st.cache_data(show_spinner=False)
def load_client_config(client_name: str) -> dict:
    """
    Loads data/clients/<ClientName>/config.json if present.
    """
    cfg_path = CLIENTS_DIR / client_name / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def resolve_branding(client_name: str | None, mode: str) -> dict:
    branding = dict(DEFAULT_BRANDING)

    if mode == "Client folders" and client_name:
        cfg = load_client_config(client_name)
        branding.update({k: v for k, v in cfg.items() if v is not None})
        branding["display_name"] = branding.get("display_name") or client_name

        lp = branding.get("logo_path")
        if lp:
            branding["logo_abs_path"] = str((CLIENTS_DIR / client_name / lp).resolve())
        else:
            branding["logo_abs_path"] = None

    elif mode == "Legacy demo":
        branding["display_name"] = "Spotify (legacy)"
        branding["tagline"] = "Sentiment-first social listening demo"
        branding["logo_abs_path"] = None

    else:  # Upload CSV
        branding["display_name"] = st.session_state.get("upload_brand_name", "Uploaded dataset")
        branding["tagline"] = st.session_state.get(
            "upload_brand_tagline",
            DEFAULT_BRANDING["tagline"],
        )
        branding["primary_color"] = st.session_state.get(
            "upload_brand_color",
            DEFAULT_BRANDING["primary_color"],
        )
        branding["logo_abs_path"] = None

    return branding

def spread_youtube_dates_for_demo(df: pd.DataFrame, days: int = 180) -> pd.DataFrame:
    # Only adjust YouTube if it has 1–2 unique dates (typical "everything is now" scrape)
    if "platform" not in df.columns:
        return df
    yt = df["platform"].astype(str).str.lower().eq("youtube")
    if yt.sum() < 50:
        return df

    d = pd.to_datetime(df.loc[yt, "created_utc"], errors="coerce").dt.date
    if d.nunique() > 2:
        return df

    base = pd.Timestamp.utcnow().normalize()
    idx = df.loc[yt].sort_values("created_utc").index
    offsets = np.linspace(0, days, num=len(idx), dtype=int)
    df.loc[idx, "created_utc"] = [base - pd.Timedelta(days=int(x)) for x in offsets]
    return df

df_raw = pd.read_csv(up)

# 1) drop duplicate column names (prevents df['created_utc'] becoming a DataFrame)
df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()].copy()

st.caption("Tip: Choose a datetime column like created_utc / timestamp, and a text column like text / body / title.")

cols = list(df_raw.columns)

# Good defaults if present
def _pick(defaults):
    for d in defaults:
        for c in cols:
            if c.lower() == d:
                return c
    for c in cols:
        cl = c.lower()
        if any(d in cl for d in defaults):
            return c
    return cols[0]

dt_guess = _pick(["created_utc", "created", "timestamp", "datetime", "date", "time", "created_at"])
text_guess = _pick(["text", "body", "content", "comment", "selftext", "title", "message"])

dt_col = st.selectbox("Datetime column", cols, index=cols.index(dt_guess) if dt_guess in cols else 0)
text_col = st.selectbox("Text column", cols, index=cols.index(text_guess) if text_guess in cols else min(1, len(cols)-1))

# Guard: don't allow same column
if dt_col == text_col:
    st.error("Datetime column and Text column must be different.")
    st.stop()

# 2) Do NOT rename (renaming can create duplicates). Instead, assign/overwrite.
df_raw = df_raw.copy()
df_raw["created_utc"] = df_raw[dt_col]
df_raw["text"] = df_raw[text_col]

# 3) Validate datetime parsing so Trends won't be empty
parsed = parse_datetime_series(df_raw["created_utc"])
ok_ratio = float(parsed.notna().mean()) if len(parsed) else 0.0

st.caption(f"Datetime parse success: {ok_ratio:.0%}")
st.write("Datetime sample:", df_raw["created_utc"].head(5).tolist())
st.write("Text sample:", df_raw["text"].head(2).astype(str).tolist())

if ok_ratio < 0.6:
    st.warning(
        "Most values in your chosen datetime column couldn't be parsed as dates. "
        "Pick a column like created_utc / timestamp."
    )
else:
    df_raw["created_utc"] = parsed


def inject_brand_css(primary_color: str):
    st.markdown(
        f"""
<style>
.brand-wrap {{
  display: flex;
  gap: 14px;
  align-items: center;
  padding: 14px 16px;
  border-left: 7px solid {primary_color};
  background: rgba(0,0,0,0.03);
  border-radius: 14px;
  margin-bottom: 10px;
}}
.brand-title {{
  margin: 0;
  font-size: 1.35rem;
  line-height: 1.2;
}}
.brand-tagline {{
  margin: 2px 0 0 0;
  opacity: 0.75;
}}
.brand-small {{
  opacity: 0.8;
  font-size: 0.85rem;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def render_brand_header(branding: dict):
    """
    Render a header at the top of a page (call on every page).
    """
    primary = branding.get("primary_color") or DEFAULT_BRANDING["primary_color"]
    inject_brand_css(primary)

    col_logo, col_text = st.columns([1, 5], vertical_alignment="center")

    with col_logo:
        logo_abs = branding.get("logo_abs_path")
        if logo_abs:
            try:
                st.image(logo_abs, width=90)
            except Exception:
                pass

        upload_logo = st.session_state.get("upload_logo_bytes")
        if upload_logo:
            st.image(upload_logo, width=90)

    with col_text:
        st.markdown(
            f"""
<div class="brand-wrap">
  <div>
    <h2 class="brand-title">{branding.get("display_name","Client")}</h2>
    <p class="brand-tagline">{branding.get("tagline","")}</p>
    <div class="brand-small">Primary color: {primary}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )


def render_brand_sidebar_badge(branding: dict):
    primary = branding.get("primary_color") or DEFAULT_BRANDING["primary_color"]
    inject_brand_css(primary)

    st.markdown(
        f"""
<div class="brand-wrap" style="margin-top:8px; margin-bottom:8px;">
  <div>
    <div style="font-weight:700;">{branding.get("display_name","Client")}</div>
    <div style="opacity:0.75; font-size:0.9rem;">{branding.get("tagline","")}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )



# -----------------------------
# Normalization / analytics
# -----------------------------
def parse_datetime_series(s: pd.Series) -> pd.Series:
    """
    Robust datetime parser:
    - unix seconds / ms (even if stored as strings)
    - ISO datetime strings
    - mixed columns
    """
    s0 = s.copy()

    # Try numeric conversion even if dtype is object (strings)
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


def normalize_posts_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Ensure required columns exist / rename common alternatives ---
    if "created_utc" not in df.columns:
        for alt in ["created", "date", "timestamp", "time"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "created_utc"})
                break

    if "text" not in df.columns:
        for alt in ["post", "content", "body", "comment"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "text"})
                break

    missing = [c for c in ["created_utc", "text"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Need at least created_utc + text")

    # --- Ensure optional-but-used columns exist BEFORE we touch them ---
    if "platform" not in df.columns:
        df["platform"] = "unknown"
    if "subreddit" not in df.columns:
        df["subreddit"] = "unknown"

    # --- Parse datetime robustly ---
    df["created_utc"] = parse_datetime_series(df["created_utc"])
    df = df.dropna(subset=["created_utc"])

    df["text"] = df["text"].astype(str)

    # --- Clean text ---
    if "clean_text" not in df.columns:
        df["clean_text"] = (
            df["text"]
            .str.lower()
            .str.replace(r"http\S+", " ", regex=True)
            .str.replace(r"[^a-z0-9\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # --- Sentiment fields ---
    if "sentiment_label" not in df.columns:
        df["sentiment_label"] = "unlabeled"
    df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip().str.lower()

    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = np.nan
    if "sentiment_confidence" not in df.columns:
        df["sentiment_confidence"] = np.nan

    # --- Normalize platform/subreddit ---
    df["platform"] = df["platform"].astype(str).str.strip().str.lower()
    df["subreddit"] = (
        df["subreddit"]
        .astype(str)
        .fillna("unknown")
        .str.strip()
        .str.replace(r"^r/", "", regex=True)
        .str.lower()
    )

    # --- Build a stable 'source' field ---
    # Reddit: subreddit
    # YouTube: query (if available), else 'youtube'
    if "query" in df.columns:
        q = df["query"].astype(str).str.strip().str.lower()
        q = q.replace({"nan": "", "none": ""})
        df["source"] = np.where(df["platform"] == "youtube", q, df["subreddit"])
        df.loc[(df["platform"] == "youtube") & (df["source"] == ""), "source"] = "youtube"
    else:
        df["source"] = df["subreddit"]

    df.loc[df["source"].isin(["", "nan", "none"]), "source"] = "unknown"

    return df


def compute_kpis(df: pd.DataFrame) -> dict[str, float]:
    total = float(len(df))
    if total == 0:
        return {"mentions": 0, "pos_pct": 0, "neg_pct": 0, "net_sent": 0}

    pos = float((df["sentiment_label"] == "positive").sum())
    neg = float((df["sentiment_label"] == "negative").sum())

    return {
        "mentions": total,
        "pos_pct": 100 * pos / total,
        "neg_pct": 100 * neg / total,
        "net_sent": (pos - neg) / total,
    }


def time_series(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    return (
        df.groupby([pd.Grouper(key="created_utc", freq=freq), "sentiment_label"])
        .size()
        .reset_index(name="count")
        .rename(columns={"created_utc": "date"})
    )


def daily_net_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=["date", "mentions", "net_sent"])

    d = df.copy()
    d["date"] = d["created_utc"].dt.date
    g = d.groupby("date")

    mentions = g.size().rename("mentions")
    pos = g.apply(lambda x: (x["sentiment_label"] == "positive").sum()).rename("pos")
    neg = g.apply(lambda x: (x["sentiment_label"] == "negative").sum()).rename("neg")

    out = pd.concat([mentions, pos, neg], axis=1).reset_index()
    out["net_sent"] = (out["pos"] - out["neg"]) / out["mentions"]
    return out[["date", "mentions", "net_sent"]]


@st.cache_data(show_spinner=False)
def top_keywords_cached(texts: tuple[str, ...], n: int = 20) -> pd.DataFrame:
    if len(texts) == 0:
        return pd.DataFrame(columns=["term", "score"])

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    X = vec.fit_transform(list(texts))
    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    top_idx = np.argsort(scores)[::-1][:n]
    return pd.DataFrame({"term": terms[top_idx], "score": scores[top_idx]})


@st.cache_data(show_spinner=False)
def topic_clusters_cached(texts: tuple[str, ...], k: int) -> pd.DataFrame:
    """
    Lightweight topic clustering using TF-IDF + KMeans.
    Returns a table with cluster_id and top_terms.
    """
    if len(texts) < max(10, k * 5):
        return pd.DataFrame(columns=["cluster_id", "top_terms"])

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000, min_df=2)
    X = vec.fit_transform(list(texts))

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    terms = np.array(vec.get_feature_names_out())
    out = []
    for cid in range(k):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            out.append((cid, ""))
            continue

        centroid = X[idx].mean(axis=0)
        scores = np.asarray(centroid).ravel()
        top = terms[np.argsort(scores)[::-1][:8]]
        out.append((cid, ", ".join(top)))

    return pd.DataFrame(out, columns=["cluster_id", "top_terms"])


def apply_keyword_filter(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    if not keywords:
        return df
    pat = "|".join([re.escape(k.lower()) for k in keywords])
    return df[df["clean_text"].str.contains(pat, na=False)]


# -----------------------------
# Shared sidebar + filtering API
# -----------------------------
def build_sidebar_and_get_df() -> tuple[pd.DataFrame, str, dict, dict]:
    """
    Renders a shared sidebar across pages and returns:
    - df_all (normalized)
    - client_name
    - filters dict
    - branding dict
    """
    if "driver_keywords" not in st.session_state:
        st.session_state["driver_keywords"] = []

    with st.sidebar:
        st.header("Data source")

        mode = st.radio("Choose dataset", ["Client folders", "Upload CSV", "Legacy demo"], index=0)
        df_raw = None
        client_name = None

        if mode == "Client folders":
            clients = list_clients()
            if not clients:
                st.info("No clients found in data/clients/. Using legacy demo.")
                mode = "Legacy demo"
            else:
                client_name = st.selectbox("Client", clients)
                df_raw = read_csv_any(CLIENTS_DIR / client_name / "posts.csv")

        if mode == "Upload CSV":
            up = st.file_uploader("Upload posts CSV", type=["csv"])
            if up is None:
                st.stop()

            df_raw = pd.read_csv(up)
            client_name = "Uploaded dataset"

            st.caption("Your CSV needs at least a datetime column and a text column.")
            cols = list(df_raw.columns)
            dt_col = st.selectbox("Datetime column", cols, index=0)
            text_col = st.selectbox("Text column", cols, index=1 if len(cols) > 1 else 0)
            df_raw = df_raw.rename(columns={dt_col: "created_utc", text_col: "text"})

            # Branding inputs for upload mode
            st.divider()
            st.subheader("Branding (for demo)")
            st.session_state["upload_brand_name"] = st.text_input(
                "Brand name",
                value=st.session_state.get("upload_brand_name", "Uploaded dataset"),
            )
            st.session_state["upload_brand_tagline"] = st.text_input(
                "Tagline",
                value=st.session_state.get("upload_brand_tagline", DEFAULT_BRANDING["tagline"]),
            )
            st.session_state["upload_brand_color"] = st.color_picker(
                "Primary color",
                value=st.session_state.get("upload_brand_color", DEFAULT_BRANDING["primary_color"]),
            )
            up_logo = st.file_uploader("Upload logo (PNG/JPG)", type=["png", "jpg", "jpeg"], key="upload_logo_file")
            if up_logo is not None:
                st.session_state["upload_logo_bytes"] = up_logo.getvalue()

        if mode == "Legacy demo":
            if not LEGACY_DEMO_FILE.exists():
                st.error("Legacy demo missing: data/sentiment_reddit_spotify.csv")
                st.stop()
            client_name = "Spotify (legacy)"
            df_raw = read_csv_any(LEGACY_DEMO_FILE)

        df = normalize_posts_df(df_raw)

        st.divider()
        st.header("Filters")

        min_date = df["created_utc"].min().date()
        max_date = df["created_utc"].max().date()
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

        platforms = sorted(df["platform"].astype(str).unique())
        platform_filter = st.multiselect("Platform", platforms, default=platforms)

        # --- Reddit subreddits (show ALL subreddits present) ---
        reddit_subs = sorted(
            df.loc[df["platform"] == "reddit", "subreddit"]
            .astype(str)
            .str.strip()
            .unique()
        )

        if len(reddit_subs) == 0:
            st.info("No Reddit rows found for this client.")
        subreddit_filter = st.multiselect("Subreddits (Reddit)", reddit_subs, default=reddit_subs)

        st.caption(f"Reddit subreddits available: {len(reddit_subs)} | selected: {len(subreddit_filter)}")

        with st.expander("Show all subreddit values"):
            st.write(reddit_subs)

        # --- YouTube sources (optional; uses query if you have it) ---
        yt_sources = sorted(
            df.loc[df["platform"] == "youtube", "source"]
            .astype(str)
            .str.strip()
            .unique()
        )
        yt_source_filter = st.multiselect("YouTube sources (query)", yt_sources, default=yt_sources)

        st.caption(f"YouTube sources available: {len(yt_sources)} | selected: {len(yt_source_filter)}")


        labels = sorted(df["sentiment_label"].astype(str).unique())
        default_labels = [x for x in ["positive", "neutral", "negative"] if x in labels] or labels
        sentiment_filter = st.multiselect("Sentiment", labels, default=default_labels)

        if df["sentiment_confidence"].notna().any():
            conf_min = float(np.nanmin(df["sentiment_confidence"]))
            conf_max = float(np.nanmax(df["sentiment_confidence"]))
            conf_range = st.slider("Confidence", 0.0, 1.0, (max(0.0, conf_min), min(1.0, conf_max)))
        else:
            conf_range = None

        if df["sentiment_score"].notna().any():
            score_min = float(np.nanmin(df["sentiment_score"]))
            score_max = float(np.nanmax(df["sentiment_score"]))
            score_range = st.slider("Sentiment score", -1.0, 1.0, (max(-1.0, score_min), min(1.0, score_max)))
        else:
            score_range = None

        search_text = st.text_input("Search posts", value="")
        must_have = st.text_input("Must-have keywords (comma separated)", value="").strip()
        must_keywords = [k.strip() for k in must_have.split(",") if k.strip()]

        st.divider()
        st.header("Driver keyword filter")
        st.caption("Pick keywords in Drivers page → filters whole dashboard.")
        st.write(st.session_state.get("driver_keywords", []))
        if st.button("Clear driver keyword filter"):
            st.session_state["driver_keywords"] = []

    # Branding + sidebar badge (outside 'with' so mode/client_name are set)
    branding = resolve_branding(client_name, mode)
    with st.sidebar:
        st.divider()
        st.header("Client branding")
        render_brand_sidebar_badge(branding)

    filters = {
        "mode": mode,
        "date_range": date_range,
        "platform_filter": platform_filter,
        "subreddit_filter": subreddit_filter,
        "yt_source_filter": yt_source_filter,
        "sentiment_filter": sentiment_filter,
        "conf_range": conf_range,
        "score_range": score_range,
        "search_text": search_text,
        "must_keywords": must_keywords,
        "driver_keywords": st.session_state.get("driver_keywords", []),
    }
    return df, client_name, filters, branding


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    f = df.copy()

    dr = filters["date_range"]
    f = f[(f["created_utc"].dt.date >= dr[0]) & (f["created_utc"].dt.date <= dr[1])]

    f = f[f["platform"].isin(filters["platform_filter"])]
    # Apply subreddit filter ONLY to reddit rows
    subs = filters.get("subreddit_filter") or []
    if subs:
        f = f[(f["platform"] != "reddit") | (f["subreddit"].isin(subs))]

    # Apply youtube source filter ONLY to youtube rows
    yts = filters.get("yt_source_filter") or []
    if yts:
        f = f[(f["platform"] != "youtube") | (f["source"].isin(yts))]

    f = f[f["sentiment_label"].isin(filters["sentiment_filter"])]

    conf_range = filters.get("conf_range")
    if conf_range is not None:
        f = f[
            (f["sentiment_confidence"].isna())
            | (
                (f["sentiment_confidence"] >= conf_range[0])
                & (f["sentiment_confidence"] <= conf_range[1])
            )
        ]

    score_range = filters.get("score_range")
    if score_range is not None:
        f = f[
            (f["sentiment_score"].isna())
            | ((f["sentiment_score"] >= score_range[0]) & (f["sentiment_score"] <= score_range[1]))
        ]

    search_text = (filters.get("search_text") or "").strip().lower()
    if search_text:
        f = f[f["text"].str.lower().str.contains(re.escape(search_text), na=False)]

    must_keywords = filters.get("must_keywords") or []
    if must_keywords:
        pat = "|".join([re.escape(k.lower()) for k in must_keywords])
        f = f[f["clean_text"].str.contains(pat, na=False)]

    f = apply_keyword_filter(f, filters.get("driver_keywords") or [])
    return f