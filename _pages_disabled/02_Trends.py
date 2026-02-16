import streamlit as st
import plotly.express as px

from dashboard_utils import (
    build_sidebar_and_get_df,
    apply_filters,
    time_series,
    daily_net_sentiment,
    render_brand_header,
)

st.set_page_config(page_title="Trends", layout="wide")

df, client_name, filters, branding = build_sidebar_and_get_df()
f = apply_filters(df, filters)

render_brand_header(branding)
st.title("Trends")
st.subheader(f"Client: **{client_name}**")

with st.expander("Debug"):
    st.write("Rows after filters:", len(f))
    if len(f) > 0:
        st.write("Date range:", f["created_utc"].min(), "→", f["created_utc"].max())
        st.write("Platform counts:", f["platform"].value_counts().head(10))
        st.write("Top sources:", f["source"].value_counts().head(10))

if len(f) == 0:
    st.warning("No data for the current filters. Expand the date range and select more platforms/sources.")
    st.stop()

freq_label = st.selectbox("Aggregation", ["Daily", "Weekly", "Monthly"], index=0)
freq = {"Daily": "D", "Weekly": "W", "Monthly": "M"}[freq_label]

st.markdown("### Volume over time (by sentiment)")
ts = time_series(f, freq=freq)
st.plotly_chart(px.area(ts, x="date", y="count", color="sentiment_label"), use_container_width=True)

st.markdown("### Net sentiment over time")
dn = daily_net_sentiment(f)
st.plotly_chart(px.line(dn, x="date", y="net_sent"), use_container_width=True)