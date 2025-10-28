from sqlalchemy import create_engine, Column, Integer, String, DateTime, Table, MetaData
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data") / "history.db"
engine = create_engine(f"sqlite:///{DB_PATH}")
Session = sessionmaker(bind=engine)
metadata = MetaData()

def ensure_db():
    if not DB_PATH.exists():
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # create simple table if not exists
    with engine.begin() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT,
            query TEXT,
            item_id TEXT,
            item_title TEXT,
            timestamp DATETIME
        )
        """)

def add_history(kind, query, item_id, item_title):
    ensure_db()
    ts = datetime.utcnow()
    with engine.begin() as conn:
        conn.execute("INSERT INTO history (kind,query,item_id,item_title,timestamp) VALUES (?,?,?,?,?)",
                     (kind, query, str(item_id), item_title, ts))
