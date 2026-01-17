import streamlit as st
import pandas as pd
from pathlib import Path

st.write("✅ App started successfully")

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Spotify Social Listening Dashboard",
    layout="wide"
)
# ----------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Load data
try:
    df = pd.read_csv(DATA_DIR / "sentiment_reddit_spotify.csv")
    summary = pd.read_csv(DATA_DIR / "sentiment_summary.csv")
    time_series = pd.read_csv(DATA_DIR / "sentiment_time_series.csv")
except Exception as e:
    st.error("Failed to load data files")
    st.exception(e)
    st.stop()

df["created_utc"] = pd.to_datetime(df["created_utc"])
time_series["date"] = pd.to_datetime(time_series["date"])

st.title("🎧 Spotify Social Listening Dashboard")
st.markdown(
    "Sentiment-first analysis of public Reddit discussions related to Spotify."
)

# ---- Sidebar Filters ----
st.sidebar.header("Filters")

min_date = df["created_utc"].min().date()
max_date = df["created_utc"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

sentiment_filter = st.sidebar.multiselect(
    "Select sentiment",
    df["sentiment_label"].unique(),
    default=df["sentiment_label"].unique()
)

filtered_df = df[
    (df["created_utc"].dt.date >= date_range[0]) &
    (df["created_utc"].dt.date <= date_range[1]) &
    (df["sentiment_label"].isin(sentiment_filter))
]

# ---- Sentiment Overview ----
st.subheader("Sentiment Overview")
st.bar_chart(
    summary.set_index("sentiment")["count"]
)

# ---- Sentiment Over Time ----
st.subheader("Sentiment Over Time")
pivot_ts = (
    time_series
    .pivot(index="date", columns="sentiment_label", values="count")
    .fillna(0)
)
st.line_chart(pivot_ts)

# ---- Sample Posts ----
st.subheader("Sample Posts")
st.dataframe(
    filtered_df[["created_utc", "sentiment_label", "text"]]
    .sort_values("created_utc", ascending=False)
    .head(10),
    use_container_width=True
)