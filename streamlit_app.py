import streamlit as st
from src.data_loader import load_csv, preview
from src.recommender import ContentRecommender
from src import external_fetchers
from src import db as dbmod
import json
from pathlib import Path

st.set_page_config(layout="wide", page_title="PLEX - Recommender")

# ------- helper -------
CONFIG_PATH = Path("config/config.json")

def load_config():
    if CONFIG_PATH.exists():
        return json.load(open(CONFIG_PATH, "r"))
    return {}

def save_config(cfg):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

# ------- Sidebar settings -------
st.sidebar.title("PLEX - Menu")
menu = st.sidebar.radio("Seleziona", ["Dashboard", "Movies", "Series", "Anime", "Settings", "About"])

cfg = load_config()

if menu == "Settings":
    st.header("Impostazioni - API keys")
    tmdb = st.text_input("TMDB API Key", value=cfg.get("tmdb_api_key",""))
    omdb = st.text_input("OMDb API Key (opzionale)", value=cfg.get("omdb_api_key",""))
    reddit_id = st.text_input("Reddit client id (opzionale)", value=cfg.get("reddit_client_id",""))
    reddit_secret = st.text_input("Reddit client secret (opzionale)", value=cfg.get("reddit_client_secret",""))
    if st.button("Salva API keys"):
        cfg['tmdb_api_key'] = tmdb.strip()
        cfg['omdb_api_key'] = omdb.strip()
        cfg['reddit_client_id'] = reddit_id.strip()
        cfg['reddit_client_secret'] = reddit_secret.strip()
        save_config(cfg)
        st.success("Salvate!")

if menu == "About":
    st.title("PLEX - Recommender")
    st.write("Dashboard personale per Film / Serie / Anime.")
    st.write("Sviluppato per uso personale. Modifica i CSV in /data per aggiornare il catalogo.")

# Shared: load data previews (fast)
def show_preview(kind):
    st.subheader(f"Preview {kind}")
    try:
        df = load_csv(kind)
        st.dataframe(df.head(10))
    except Exception as e:
        st.error(str(e))

def dashboard():
    st.title("PLEX Dashboard")
    st.write("Seleziona una categoria dal menu a sinistra.")
    st.markdown("---")
    st.subheader("Quick Preview")
    cols = st.columns(3)
    with cols[0]:
        show_preview("movies")
    with cols[1]:
        show_preview("series")
    with cols[2]:
        show_preview("anime")

if menu == "Dashboard":
    dashboard()

# Main pages
def category_page(kind):
    st.title(f"{kind.capitalize()}")
    # three sections: My Discover, Popular, All Time
    st.markdown("### My Discover")
    q = st.text_input(f"Cerca un titolo in {kind} (usa testo per suggerire simili)", key=f"q_{kind}")
    if st.button(f"Cerca e Suggerisci in {kind}", key=f"search_{kind}"):
        rec = ContentRecommender(kind)
        res = rec.recommend_by_title(q, top_k=12)
        st.write("Risultati:")
        st.dataframe(res[['title','year']].head(12))
        # save interactions
        for _, r in res.head(3).iterrows():
            dbmod.add_history(kind, q, r.get('id', ''), r.get('title',''))
    st.markdown("---")
    st.markdown("### Popular (online fallback -> locale)")
    try:
        rec = ContentRecommender(kind)
        pop = rec.get_popular(top_k=12)
        st.dataframe(pop[['title','year']].head(12))
    except Exception as e:
        st.info("Popolari non disponibili: {}".format(e))
    st.markdown("---")
    st.markdown("### All Time (Top by rating within dataset)")
    try:
        rec = ContentRecommender(kind)
        alltime = rec.get_popular(top_k=50)
        st.dataframe(alltime[['title','year']].head(12))
    except Exception as e:
        st.info("All time non disponibile: {}".format(e))

if menu == "Movies":
    category_page("movies")
if menu == "Series":
    category_page("series")
if menu == "Anime":
    category_page("anime")
