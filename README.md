# Sentiment-First Social Listening Dashboard (Streamlit)

A client-ready, sentiment-first social listening dashboard that lets users explore public online discussion data (Reddit + YouTube) through filters, trends, drivers, post explorer, exports, and client switching.

## Live Demo
- Streamlit App: https://spotify-social-listening-demo-n7nv2yqjzfxfc92cnqtk4n.streamlit.app
- Repo: https://github.com/prans-code/spotify-social-listening-demo.git

## Key Features
- **Client switcher** (Spotify / Netflix / Nike …)
- **Sentiment-first KPIs**: mentions, %positive, %negative, net sentiment
- **Trends**: volume over time + net sentiment over time
- **Drivers**: TF-IDF keyword drivers + optional topic clustering
- **Explorer**: filterable table + CSV export
- **Compare periods**: Period A vs Period B sentiment mix

## Project Structure
- `app.py` – entrypoint
- `dashboard_utils.py` – shared data loading, normalization, filters, analytics helpers
- `pages/` – Streamlit pages (if enabled)
- `data/clients/<Client>/posts.csv` – client dataset
- `data/clients/<Client>/config.json` – branding per client
- `docs/` – faculty documentation (demo guide, data schema, roadmap)

## How to Run Locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
