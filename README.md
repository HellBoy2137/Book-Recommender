# Book Recommender — New Dataset Package

This package contains a full Streamlit book recommender built from an uploaded dataset (books.csv).
- Content-based: CountVectorizer + cosine similarity
- Collaborative: TruncatedSVD MF (built on ratings.csv). If ratings.csv was not provided, a synthetic ratings file was generated for demo purposes.

How to run locally (Windows PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

Deploy: push to GitHub and connect to Streamlit Cloud. Avoid packages needing C compilation.

