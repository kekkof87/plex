import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

def _safe_read_csv(path: Path):
    encs = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    for e in encs:
        try:
            df = pd.read_csv(path, encoding=e)
            return df
        except Exception:
            continue
    # last resort: read as plain text and split lines
    txt = path.read_text(errors="ignore")
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    if not lines:
        return pd.DataFrame(columns=["title"])
    # if first line has commas treat as csv
    if "," in lines[0]:
        import io
        try:
            df = pd.read_csv(io.StringIO("\n".join(lines)))
            return df
        except Exception:
            pass
    # else treat each line as title
    return pd.DataFrame({"title": lines[1:]}) if len(lines) > 1 else pd.DataFrame({"title": lines})

def load_csv(kind: str):
    """
    kind in ["movies","series","anime"]
    """
    path = DATA_DIR / f"{kind}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} non trovato. Metti il tuo CSV in data/{kind}.csv")
    df = _safe_read_csv(path)
    # Normalize to ensure there is a title column and it's clean
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if "title" not in df.columns:
        # try to pick first string-like column
        for col in df.columns:
            if df[col].dtype == object:
                df = df.rename(columns={col: "title"})
                break
        else:
            # fallback: create title from first col
            df["title"] = df.iloc[:,0].astype(str)
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"].str.len() > 0].reset_index(drop=True)
    return df.fillna("")

def preview(kind: str, n=5):
    df = load_csv(kind)
    return df.head(n)
