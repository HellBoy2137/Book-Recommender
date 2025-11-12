# Book Recommender — Single `books.csv` dataset

This project builds a book recommender using **only** the uploaded `books.csv` file (placed in `data/books.csv`).

The code auto-detects title/author/id/rating columns and uses:
- Content-based recommendations: CountVectorizer + Cosine similarity (primary)

- Popularity fallback: if a rating column exists, the app uses it to rank / break ties.

- Optional demo: generate synthetic users from the aggregate rating (if you want personalized-like results).

## How the code adapts to your file
The script inspects columns and picks sensible defaults for `title`, `authors`, `book_id`, and `rating`. If it can't find a rating column, it will use `ratings_count` or treat popularity uniformly.


## Run locally (Windows PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

## Files included
- `data/books.csv` (your uploaded file)

- `src/` modules: `utils.py`, `preprocessing.py`, `content_model.py`, `popularity.py`

- `app.py` — Streamlit app


If you want me to generate synthetic user ratings for a demo personalized CF model, ask and I'll include precomputed SVD models too.
