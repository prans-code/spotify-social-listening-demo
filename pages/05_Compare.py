import streamlit as st
import plotly.express as px

from dashboard_utils import build_sidebar_and_get_df, compute_kpis

st.set_page_config(page_title="Compare periods", layout="wide")

df, client_name, filters = build_sidebar_and_get_df()

st.title("Compare periods")
st.subheader(f"Client: **{client_name}**")

if len(df) == 0:
    st.warning("No data loaded.")
    st.stop()

full_min = df["created_utc"].min().date()
full_max = df["created_utc"].max().date()

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Period A**")
    a_range = st.date_input("A date range", value=(full_min, full_max), min_value=full_min, max_value=full_max, key="a_range")
with c2:
    st.markdown("**Period B**")
    b_range = st.date_input("B date range", value=(full_min, full_max), min_value=full_min, max_value=full_max, key="b_range")

def slice_period(dfr, dr):
    out = dfr[(dfr["created_utc"].dt.date >= dr[0]) & (dfr["created_utc"].dt.date <= dr[1])]
    out = out[out["platform"].isin(filters["platform_filter"])]
    out = out[out["subreddit"].isin(filters["source_filter"])]
    out = out[out["sentiment_label"].isin(filters["sentiment_filter"])]
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

dist = distA._append(distB, ignore_index=True)
dist["pct"] = dist["pct"] * 100

st.markdown("### Sentiment mix comparison")
st.plotly_chart(px.bar(dist, x="sentiment_label", y="pct", color="period", barmode="group"), use_container_width=True)