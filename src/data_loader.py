import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

def load_csv(kind: str):
    """
    kind in ["movies","series","anime"]
    """
    path = DATA_DIR / f"{kind}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} non trovato. Metti il tuo CSV in data/{kind}.csv")
    df = pd.read_csv(path)
    # normalizza colonne: assicurati ci siano almeno title, id, description, genres, year
    # rename se necessario
    return df.fillna("")

def preview(kind: str, n=5):
    df = load_csv(kind)
    return df.head(n)
