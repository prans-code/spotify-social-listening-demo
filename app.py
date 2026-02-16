from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


st.set_page_config(page_title="Social Listening Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CLIENTS_DIR = DATA_DIR / "clients"
LEGACY_DEMO_FILE = DATA_DIR / "sentiment_reddit_spotify.csv"

st.markdown(
    "<style>[data-testid='stSidebarNav'] {display:none;}</style>",
    unsafe_allow_html=True
)

# ---------------------------
# Utilities
# ---------------------------
@st.cache_data(show_spinner=False)
def list_clients() -> list[str]:
    if not CLIENTS_DIR.exists():
        return []
    out = []
    for p in CLIENTS_DIR.glob("*"):
        if p.is_dir() and (p / "posts.csv").exists():
            out.append(p.name)
    return sorted(out)


@st.cache_data(show_spinner=False)
def read_csv_any(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_datetime_series(s: pd.Series) -> pd.Series:
    # Handles unix seconds, unix ms, or ISO strings
    if pd.api.types.is_numeric_dtype(s):
        vals = s.dropna().astype(float)
        if len(vals) == 0:
            return pd.to_datetime(s, errors="coerce")
        med = float(np.median(vals))
        if med > 1e12:  # likely ms
            return pd.to_datetime(s, unit="ms", errors="coerce")
        if med > 1e9:   # likely seconds
            return pd.to_datetime(s, unit="s", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def normalize_posts_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Minimal requirements
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

    df["created_utc"] = parse_datetime_series(df["created_utc"])
    df = df.dropna(subset=["created_utc"])
    df["text"] = df["text"].astype(str)

    if "clean_text" not in df.columns:
        df["clean_text"] = (
            df["text"]
            .str.lower()
            .str.replace(r"http\S+", " ", regex=True)
            .str.replace(r"[^a-z0-9\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    if "platform" not in df.columns:
        df["platform"] = "unknown"
    if "subreddit" not in df.columns:
        df["subreddit"] = "unknown"

    if "sentiment_label" not in df.columns:
        df["sentiment_label"] = "unlabeled"
    df["sentiment_label"] = df["sentiment_label"].astype(str).str.lower()

    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = np.nan
    if "sentiment_confidence" not in df.columns:
        df["sentiment_confidence"] = np.nan

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
    ts = (
        df.groupby([pd.Grouper(key="created_utc", freq=freq), "sentiment_label"])
        .size()
        .reset_index(name="count")
        .rename(columns={"created_utc": "date"})
    )
    return ts


@st.cache_data(show_spinner=False)
def top_keywords_cached(texts: tuple[str, ...], n: int = 15) -> pd.DataFrame:
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


def apply_keyword_filter(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    if not keywords:
        return df
    pat = "|".join([re.escape(k.lower()) for k in keywords])
    return df[df["clean_text"].str.contains(pat, na=False)]


# ---------------------------
# Sidebar: data source + filters
# ---------------------------
st.title("Sentiment-First Social Listening Dashboard")
st.caption("Interactive Streamlit demo — switch clients, filter, drill down, export.")

if "driver_keywords" not in st.session_state:
    st.session_state["driver_keywords"] = []  # selected keywords from Drivers tab

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

        st.caption("Tip: Your CSV needs at least a datetime column and a text column.")
        cols = list(df_raw.columns)
        dt_col = st.selectbox("Datetime column", cols, index=0)
        text_col = st.selectbox("Text column", cols, index=1 if len(cols) > 1 else 0)
        df_raw = df_raw.rename(columns={dt_col: "created_utc", text_col: "text"})

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

    sources = sorted(df["subreddit"].astype(str).unique())
    source_filter = st.multiselect("Source (subreddit/site)", sources, default=sources)

    labels = sorted(df["sentiment_label"].astype(str).unique())
    default_labels = [x for x in ["positive", "neutral", "negative"] if x in labels] or labels
    sentiment_filter = st.multiselect("Sentiment", labels, default=default_labels)

    # Optional sliders (if present)
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
    st.caption("Select keywords in Drivers tab to filter the whole dashboard.")
    if st.button("Clear driver keyword filter"):
        st.session_state["driver_keywords"] = []


# ---------------------------
# Apply filters
# ---------------------------
f = df.copy()
f = f[(f["created_utc"].dt.date >= date_range[0]) & (f["created_utc"].dt.date <= date_range[1])]
f = f[f["platform"].isin(platform_filter)]
f = f[f["subreddit"].isin(source_filter)]
f = f[f["sentiment_label"].isin(sentiment_filter)]

if conf_range is not None:
    f = f[(f["sentiment_confidence"].isna()) | ((f["sentiment_confidence"] >= conf_range[0]) & (f["sentiment_confidence"] <= conf_range[1]))]

if score_range is not None:
    f = f[(f["sentiment_score"].isna()) | ((f["sentiment_score"] >= score_range[0]) & (f["sentiment_score"] <= score_range[1]))]

if search_text.strip():
    s = search_text.strip().lower()
    f = f[f["text"].str.lower().str.contains(re.escape(s), na=False)]

if must_keywords:
    pat = "|".join([re.escape(k.lower()) for k in must_keywords])
    f = f[f["clean_text"].str.contains(pat, na=False)]

# Apply driver keyword filter from session state (set in Drivers tab)
f = apply_keyword_filter(f, st.session_state.get("driver_keywords", []))


# ---------------------------
# Header + KPI + Highlights
# ---------------------------
st.subheader(f"Client: **{client_name}**")
k = compute_kpis(f)

a, b, c, d = st.columns(4)
a.metric("Mentions", f"{int(k['mentions']):,}")
b.metric("% Positive", f"{k['pos_pct']:.1f}%")
c.metric("% Negative", f"{k['neg_pct']:.1f}%")
d.metric("Net Sentiment", f"{k['net_sent']:+.2f}")

# Highlights (quick “wow”)
if len(f) > 0:
    dn = daily_net_sentiment(f)
    if len(dn) > 0:
        worst = dn.sort_values("net_sent").iloc[0]
        worst_date = worst["date"]
        worst_net = float(worst["net_sent"])
        worst_mentions = int(worst["mentions"])

        st.markdown("### Highlights")
        st.info(f"**Most negative period:** {worst_date} — Net sentiment {worst_net:+.2f} across {worst_mentions:,} mentions.")

        # show top negative drivers for that day
        day_df = f[f["created_utc"].dt.date == worst_date]
        neg_texts = tuple(day_df.loc[day_df["sentiment_label"] == "negative", "clean_text"].astype(str).tolist())
        neg_terms_day = top_keywords_cached(neg_texts, n=10)

        cols = st.columns([1, 1])
        with cols[0]:
            if len(neg_terms_day) > 0:
                st.caption("Top negative keywords (that day)")
                st.plotly_chart(px.bar(neg_terms_day, x="score", y="term", orientation="h"), use_container_width=True)
            else:
                st.caption("No negative keywords available for that day.")

        with cols[1]:
            st.caption("Example negative posts (that day)")
            ex = day_df[day_df["sentiment_label"] == "negative"].sort_values("sentiment_confidence", ascending=False)
            ex = ex.head(3) if len(ex) > 0 else day_df.head(3)
            for i, row in ex.iterrows():
                with st.expander(row["text"][:120] + ("..." if len(row["text"]) > 120 else "")):
                    st.write(row["text"])


# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Trends", "Drivers", "Explorer", "Compare periods"])


with tab1:
    if len(f) == 0:
        st.warning("No data for the current filters.")
    else:
        left, right = st.columns(2)

        with left:
            dist = f["sentiment_label"].value_counts().reset_index()
            dist.columns = ["sentiment_label", "count"]
            st.plotly_chart(px.pie(dist, names="sentiment_label", values="count", hole=0.45), use_container_width=True)

            # Heatmap: weekday vs hour (nice interactive drilldown)
            h = f.copy()
            h["weekday"] = h["created_utc"].dt.day_name()
            h["hour"] = h["created_utc"].dt.hour
            # keep weekday order
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            h["weekday"] = pd.Categorical(h["weekday"], categories=weekday_order, ordered=True)
            heat = h.groupby(["weekday", "hour"]).size().reset_index(name="mentions")
            st.markdown("#### Activity heatmap (weekday × hour)")
            st.plotly_chart(px.density_heatmap(heat, x="hour", y="weekday", z="mentions"), use_container_width=True)

        with right:
            by_plat = f.groupby(["platform", "sentiment_label"]).size().reset_index(name="count")
            st.plotly_chart(px.bar(by_plat, x="platform", y="count", color="sentiment_label", barmode="stack"), use_container_width=True)

            # Top sources by negative volume
            neg_sources = (
                f[f["sentiment_label"] == "negative"]
                .groupby("subreddit")
                .size()
                .sort_values(ascending=False)
                .head(10)
                .reset_index(name="negative_mentions")
            )
            st.markdown("#### Top sources by negative mentions")
            if len(neg_sources) == 0:
                st.caption("No negative posts in this filter set.")
            else:
                st.plotly_chart(px.bar(neg_sources, x="negative_mentions", y="subreddit", orientation="h"), use_container_width=True)


with tab2:
    if len(f) == 0:
        st.warning("No data for the current filters.")
    else:
        freq_label = st.selectbox("Aggregation", ["Daily", "Weekly", "Monthly"], index=0)
        freq = {"Daily": "D", "Weekly": "W", "Monthly": "M"}[freq_label]

        ts = time_series(f, freq=freq)
        st.markdown("#### Volume over time (by sentiment)")
        if len(ts) == 0:
            st.info("No data for this filter set.")
        else:
            st.plotly_chart(px.area(ts, x="date", y="count", color="sentiment_label"), use_container_width=True)

        # Net sentiment over time
        st.markdown("#### Net sentiment over time")
        dn = daily_net_sentiment(f)
        if len(dn) > 0:
            st.plotly_chart(px.line(dn, x="date", y="net_sent"), use_container_width=True)


with tab3:
    if len(f) == 0:
        st.warning("No data for the current filters.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Negative drivers (TF-IDF keywords)")
            neg_texts = tuple(f.loc[f["sentiment_label"] == "negative", "clean_text"].astype(str).tolist())
            neg_terms = top_keywords_cached(neg_texts, n=25)
            if len(neg_terms) == 0:
                st.caption("No negative posts in this filter set.")
            else:
                st.plotly_chart(px.bar(neg_terms, x="score", y="term", orientation="h"), use_container_width=True)

                picked = st.multiselect(
                    "Select keywords to filter the whole dashboard",
                    neg_terms["term"].tolist(),
                    default=st.session_state.get("driver_keywords", []),
                    key="neg_kw_pick",
                )
                if picked != st.session_state.get("driver_keywords", []):
                    st.session_state["driver_keywords"] = picked

        with col2:
            st.markdown("#### Positive drivers (TF-IDF keywords)")
            pos_texts = tuple(f.loc[f["sentiment_label"] == "positive", "clean_text"].astype(str).tolist())
            pos_terms = top_keywords_cached(pos_texts, n=25)
            if len(pos_terms) == 0:
                st.caption("No positive posts in this filter set.")
            else:
                st.plotly_chart(px.bar(pos_terms, x="score", y="term", orientation="h"), use_container_width=True)

        st.divider()
        st.markdown("### Topic clusters (themes)")
        st.caption("Experimental: TF-IDF + KMeans. Useful for quick client demos.")

        enable_clusters = st.checkbox("Enable topic clustering", value=False)
        if enable_clusters:
            k_topics = st.slider("Number of topics", 2, 10, 5)
            texts_all = tuple(f["clean_text"].astype(str).tolist())
            clusters = topic_clusters_cached(texts_all, k=k_topics)

            if len(clusters) == 0:
                st.warning("Not enough data for clustering with current filters.")
            else:
                st.dataframe(clusters, use_container_width=True)


with tab4:
    if len(f) == 0:
        st.warning("No data for the current filters.")
    else:
        st.download_button(
            "Download filtered posts (CSV)",
            data=f.to_csv(index=False).encode("utf-8"),
            file_name="filtered_posts.csv",
            mime="text/csv",
        )

        st.caption(f"Driver keyword filter active: {st.session_state.get('driver_keywords', [])}")

        show_cols = [c for c in ["created_utc", "platform", "subreddit", "sentiment_label", "sentiment_score", "sentiment_confidence", "text"] if c in f.columns]
        f_show = f.sort_values("created_utc", ascending=False).reset_index(drop=True)

        sort_mode = st.selectbox("Sort by", ["Newest", "Oldest", "Most negative score", "Most confident"], index=0)
        if sort_mode == "Oldest":
            f_show = f_show.sort_values("created_utc", ascending=True).reset_index(drop=True)
        elif sort_mode == "Most negative score" and "sentiment_score" in f_show.columns:
            f_show = f_show.sort_values("sentiment_score", ascending=True).reset_index(drop=True)
        elif sort_mode == "Most confident" and "sentiment_confidence" in f_show.columns:
            f_show = f_show.sort_values("sentiment_confidence", ascending=False).reset_index(drop=True)

        st.dataframe(f_show[show_cols], use_container_width=True, height=450)

        if len(f_show) > 0:
            idx = st.number_input("Open full post (row #)", 0, len(f_show) - 1, 0, 1)
            row = f_show.loc[int(idx)]
            st.markdown("#### Full text")
            st.write(row["text"])


with tab5:
    if len(df) == 0:
        st.warning("No data loaded.")
    else:
        st.markdown("### Compare two time windows (client-friendly)")
        st.caption("Pick two periods to show changes in sentiment mix + net sentiment.")

        full_min = df["created_utc"].min().date()
        full_max = df["created_utc"].max().date()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Period A**")
            a_range = st.date_input("A date range", value=(full_min, full_max), min_value=full_min, max_value=full_max, key="a_range")
        with c2:
            st.markdown("**Period B**")
            b_range = st.date_input("B date range", value=(full_min, full_max), min_value=full_min, max_value=full_max, key="b_range")

        def slice_period(dfr: pd.DataFrame, dr):
            out = dfr[(dfr["created_utc"].dt.date >= dr[0]) & (dfr["created_utc"].dt.date <= dr[1])]
            # keep same non-date filters to make comparison fair
            out = out[out["platform"].isin(platform_filter)]
            out = out[out["subreddit"].isin(source_filter)]
            out = out[out["sentiment_label"].isin(sentiment_filter)]
            return out

        A = slice_period(df, a_range)
        B = slice_period(df, b_range)

        kA = compute_kpis(A)
        kB = compute_kpis(B)

        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Mentions (A)", f"{int(kA['mentions']):,}")
        x2.metric("Mentions (B)", f"{int(kB['mentions']):,}")
        x3.metric("Net Sent (A)", f"{kA['net_sent']:+.2f}")
        x4.metric("Net Sent (B)", f"{kB['net_sent']:+.2f}", delta=f"{(kB['net_sent']-kA['net_sent']):+.2f}")

        distA = A["sentiment_label"].value_counts(normalize=True).reset_index()
        distA.columns = ["sentiment_label", "pct"]
        distA["period"] = "A"

        distB = B["sentiment_label"].value_counts(normalize=True).reset_index()
        distB.columns = ["sentiment_label", "pct"]
        distB["period"] = "B"

        dist = pd.concat([distA, distB], ignore_index=True)
        dist["pct"] = dist["pct"] * 100

        st.markdown("#### Sentiment mix comparison")
        st.plotly_chart(px.bar(dist, x="sentiment_label", y="pct", color="period", barmode="group"), use_container_width=True)