# scripts/validate_and_clean_csvs.py
import pandas as pd
from pathlib import Path
import chardet
import sys

DATA_DIR = Path("data")
kinds = ["movies", "series", "anime"]

def detect_encoding(path):
    raw = open(path, "rb").read()
    res = chardet.detect(raw)
    return res.get("encoding", "utf-8")

def try_read(path):
    # try common encodings
    encs = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    for e in encs:
        try:
            df = pd.read_csv(path, encoding=e)
            return df, e
        except Exception:
            continue
    # fallback: try detect
    try:
        enc = detect_encoding(path)
        df = pd.read_csv(path, encoding=enc)
        return df, enc
    except Exception as ex:
        return None, None

def normalize_df(df, path):
    # If read failed and df is None -> attempt to read as plain text lines
    if df is None:
        text = Path(path).read_text(errors="ignore")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        # if first line seems header containing comma, attempt to split
        if lines and "," in lines[0]:
            import io
            try:
                df = pd.read_csv(io.StringIO("\n".join(lines)))
            except Exception:
                df = pd.DataFrame({"title": lines[1:]}) if len(lines) > 1 else pd.DataFrame({"title": lines})
        else:
            # assume each line a title, or single column
            df = pd.DataFrame({"title": lines[1:]}) if len(lines) > 1 else pd.DataFrame({"title": lines})
    # If df is a Series (single column without header), convert
    if isinstance(df, pd.Series):
        df = df.to_frame()
    # If df is DataFrame but has no header name matching title, try to detect column with most strings
    if "title" not in df.columns:
        # try to find first col that looks like titles
        for col in df.columns:
            # dropna and sample
            sample = df[col].dropna().astype(str).head(10).tolist()
            if all(isinstance(x, str) for x in sample) and len(sample) > 0:
                # rename this column to title
                df = df.rename(columns={col: "title"})
                break
        else:
            # fallback: if single unnamed column, rename to title
            if len(df.columns) == 1:
                df.columns = ["title"]
            else:
                # create title column by concatenating row values
                df["title"] = df.iloc[:,0].astype(str)
    # Ensure title column exists
    if "title" not in df.columns:
        df["title"] = df.iloc[:,0].astype(str)
    # Strip whitespace, drop empty rows
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"].str.len() > 0].reset_index(drop=True)
    return df

def main():
    DATA_DIR.mkdir(exist_ok=True)
    for k in kinds:
        path = DATA_DIR / f"{k}.csv"
        if not path.exists():
            print(f"[WARN] {path} not found -> skipping")
            continue
        print(f"[INFO] Processing {path}")
        df, enc = try_read(path)
        print(f"  read encoding: {enc}")
        df = normalize_df(df, path)
        # save normalized as UTF-8 no BOM
        out = DATA_DIR / f"{k}.csv"
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"  Saved normalized file: {out} (rows={len(df)})")
    print("All done. Please restart Streamlit.")

if __name__ == "__main__":
    main()
