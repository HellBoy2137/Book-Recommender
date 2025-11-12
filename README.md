# Book Recommender (Goodbooks-based)

This repository provides a Streamlit-ready book recommender using the Goodbooks dataset.
It includes content-based recommendations (CountVectorizer + Cosine) and a collaborative model (TruncatedSVD).

## How to use
1. Place your Goodbooks CSVs in the `data/` folder: `books.csv`, `ratings.csv`, `users.csv`.
2. Create and activate a virtual environment (Windows PowerShell):
   ```
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   python -m streamlit run app.py
   ```
3. Deploy on Streamlit Cloud by pushing to GitHub and connecting the repo.

