import requests
from tmdbv3api import TMDb, Movie
import json
from pathlib import Path
from typing import Optional

CONFIG_PATH = Path("config/config.json")

def load_config():
    if not CONFIG_PATH.exists():
        return {}
    return json.load(open(CONFIG_PATH, "r"))

def tmdb_client():
    cfg = load_config()
    tmdb_key = cfg.get("tmdb_api_key")
    if not tmdb_key:
        return None
    tmdb = TMDb()
    tmdb.api_key = tmdb_key
    movie = Movie()
    return movie

def get_tmdb_popular(kind="movie", page=1):
    """
    kind: movie or tv
    """
    movie = tmdb_client()
    if movie is None:
        return []
    try:
        res = movie.popular(page=page)
        return res
    except Exception as e:
        print("TMDB fetch error:", e)
        return []

def search_tmdb_by_title(title, kind="movie"):
    movie = tmdb_client()
    if movie is None:
        return []
    try:
        return movie.search(title)
    except Exception as e:
        print("TMDB search error:", e)
        return []
