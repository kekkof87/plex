# PLEX - Raccomandatore Film / Serie / Anime (personale)

## Scopo
App personale per raccomandazioni su Film, Serie e Anime basata su:
- tre CSV separati (movies.csv, series.csv, anime.csv)
- motore ibrido content-based (embeddings) + popolarità da fonti esterne (TMDB, Reddit)
- interfaccia grafica in Streamlit in stile dashboard (My Discover / Popular / All Time)
- salvataggio cronologia in SQLite
- Impostazioni → API key per inserire chiavi di servizi esterni

## Requisiti
- Python 3.10+
- GitHub repository chiamata `plex`

## Installazione (locale)
1. Clona la repo
2. Crea virtualenv e attiva
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # linux / mac
   .venv\\Scripts\\activate    # windows
