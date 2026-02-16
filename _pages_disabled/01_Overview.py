import streamlit as st
import plotly.express as px

from dashboard_utils import (
    build_sidebar_and_get_df,
    apply_filters,
    compute_kpis,
    daily_net_sentiment,
    top_keywords_cached,
    render_brand_header,
)


st.set_page_config(page_title="Overview", layout="wide")


# ---- Reset button helpers ----
def reset_filters():
    # Keys created by dashboard_utils sidebar
    keys_to_clear = [
        "driver_keywords",
        "upload_brand_name",
        "upload_brand_tagline",
        "upload_brand_color",
        "upload_logo_bytes",
        "upload_logo_file",
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()


# NOTE: build_sidebar_and_get_df now returns (df, client_name, filters, branding)
df, client_name, filters, branding = build_sidebar_and_get_df()
f = apply_filters(df, filters)

st.title("Overview")
render_brand_header(branding)

topbar_l, topbar_r = st.columns([4, 1])
with topbar_r:
    if st.button("Reset filters", use_container_width=True):
        reset_filters()

st.subheader(f"Client: **{client_name}**")


# ---- Data quality warning (uploaded datasets often come unlabeled) ----
if (df["sentiment_label"] == "unlabeled").mean() > 0.5:
    st.warning(
        "Most posts are **unlabeled** (no sentiment). "
        "Upload a dataset that includes `sentiment_label` (positive/neutral/negative), "
        "or run your sentiment pipeline to generate it. "
        "The dashboard still works, but sentiment charts won’t be meaningful."
    )


# ---- KPIs ----
k = compute_kpis(f)
a, b, c, d = st.columns(4)
a.metric("Mentions", f"{int(k['mentions']):,}")
b.metric("% Positive", f"{k['pos_pct']:.1f}%")
c.metric("% Negative", f"{k['neg_pct']:.1f}%")
d.metric("Net Sentiment", f"{k['net_sent']:+.2f}")

if len(f) == 0:
    st.warning("No data for the current filters.")
    st.stop()


# ---- Highlights ----
dn = daily_net_sentiment(f)
if len(dn) > 0:
    worst = dn.sort_values("net_sent").iloc[0]
    worst_date = worst["date"]
    st.info(
        f"**Most negative day:** {worst_date} — Net sentiment {float(worst['net_sent']):+.2f} "
        f"on {int(worst['mentions']):,} mentions."
    )

    day_df = f[f["created_utc"].dt.date == worst_date]
    neg_texts = tuple(
        day_df.loc[day_df["sentiment_label"] == "negative", "clean_text"]
        .astype(str)
        .tolist()
    )
    neg_terms = top_keywords_cached(neg_texts, n=10)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Negative keywords (worst day)")
        if len(neg_terms) > 0:
            st.plotly_chart(
                px.bar(neg_terms, x="score", y="term", orientation="h"),
                use_container_width=True,
            )
        else:
            st.caption("No negative keywords available for that day.")

    with c2:
        st.markdown("#### Example posts (worst day)")
        if "sentiment_confidence" in day_df.columns and day_df["sentiment_confidence"].notna().any():
            ex = day_df.sort_values("sentiment_confidence", ascending=False).head(3)
        else:
            ex = day_df.head(3)

        for _, row in ex.iterrows():
            preview = row["text"][:120] + ("..." if len(row["text"]) > 120 else "")
            with st.expander(preview):
                st.write(row["text"])


# ---- Visuals ----
left, right = st.columns(2)

with left:
    st.markdown("### Sentiment mix")
    dist = f["sentiment_label"].value_counts().reset_index()
    dist.columns = ["sentiment_label", "count"]
    st.plotly_chart(px.pie(dist, names="sentiment_label", values="count", hole=0.45), use_container_width=True)

with right:
    st.markdown("### Platform mix")
    by_plat = f.groupby(["platform", "sentiment_label"]).size().reset_index(name="count")
    st.plotly_chart(px.bar(by_plat, x="platform", y="count", color="sentiment_label", barmode="stack"), use_container_width=True)


# ---- Quick report download (client demo friendly) ----
report_md = f"""# Social Listening Report — {branding.get("display_name", client_name)}

## Branding
- Tagline: {branding.get("tagline", "")}
- Primary color: {branding.get("primary_color", "")}

## Filter summary
- Date range: {filters['date_range'][0]} to {filters['date_range'][1]}
- Platforms: {', '.join(filters['platform_filter'])}
- Sources: {', '.join(filters['source_filter'])}
- Sentiments: {', '.join(filters['sentiment_filter'])}
- Driver keywords: {filters.get('driver_keywords', [])}

## KPIs
- Mentions: {int(k['mentions']):,}
- % Positive: {k['pos_pct']:.1f}%
- % Negative: {k['neg_pct']:.1f}%
- Net sentiment: {k['net_sent']:+.2f}
"""

footer = branding.get("report_footer", "")
if footer:
    report_md += f"\n---\n{footer}\n"

st.download_button(
    "Download report (Markdown)",
    report_md.encode("utf-8"),
    file_name="social_listening_report.md",
)
