from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'

def load_books(path=None):
    p = DATA_DIR / 'books.csv' if path is None else Path(path)
    return pd.read_csv(p)

def load_ratings(path=None):
    p = DATA_DIR / 'ratings.csv' if path is None else Path(path)
    return pd.read_csv(p)

def load_users(path=None):
    p = DATA_DIR / 'users.csv' if path is None else Path(path)
    return pd.read_csv(p)
