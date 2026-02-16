import streamlit as st

from dashboard_utils import (
    build_sidebar_and_get_df,
    apply_filters,
    render_brand_header,
)

st.set_page_config(page_title="Explorer", layout="wide")

df, client_name, filters, branding = build_sidebar_and_get_df()
f = apply_filters(df, filters)

st.title("Explorer")
render_brand_header(branding)
st.subheader(f"Client: **{client_name}**")

if len(f) == 0:
    st.warning("No data for the current filters.")
    st.stop()

st.download_button(
    "Download filtered posts (CSV)",
    data=f.to_csv(index=False).encode("utf-8"),
    file_name="filtered_posts.csv",
    mime="text/csv",
)

st.caption(f"Driver keyword filter active: {st.session_state.get('driver_keywords', [])}")

show_cols = [c for c in ["created_utc", "platform", "source", "sentiment_label", "sentiment_score", "sentiment_confidence", "text"] if c in f.columns]

f_show = f.sort_values("created_utc", ascending=False).reset_index(drop=True)

sort_mode = st.selectbox("Sort by", ["Newest", "Oldest", "Most negative score", "Most confident"], index=0)
if sort_mode == "Oldest":
    f_show = f_show.sort_values("created_utc", ascending=True).reset_index(drop=True)
elif sort_mode == "Most negative score" and "sentiment_score" in f_show.columns:
    f_show = f_show.sort_values("sentiment_score", ascending=True).reset_index(drop=True)
elif sort_mode == "Most confident" and "sentiment_confidence" in f_show.columns:
    f_show = f_show.sort_values("sentiment_confidence", ascending=False).reset_index(drop=True)

st.dataframe(f_show[show_cols], use_container_width=True, height=520)

idx = st.number_input("Open full post (row #)", 0, len(f_show) - 1, 0, 1)
st.markdown("#### Full text")
st.write(f_show.loc[int(idx), "text"])
