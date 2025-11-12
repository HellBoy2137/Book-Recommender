import streamlit as st
import pandas as pd
import numpy as np
from src.utils import load_books
from src.preprocessing import prepare_books
from src.content_model import ContentRecommender
from src.popularity import top_popular
from rapidfuzz import process, fuzz
import json, time
import matplotlib.pyplot as plt
import pandas as _pd

# ------------------ Plot helpers ------------------
def plot_rating_histogram(books):
    if '__rating__' not in books.columns or books['__rating__'].dropna().empty:
        st.info("No rating column detected in dataset for histogram.")
        return
    data = books['__rating__'].dropna().astype(float)
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.hist(data, bins=20, edgecolor='black')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    ax.set_title('Rating distribution')
    st.pyplot(fig)

def plot_top_authors(books, topn=10):
    if 'authors' not in books.columns:
        st.info('No authors column found for top authors chart.')
        return
    ser = books['authors'].fillna('Unknown').astype(str)
    counts = ser.value_counts().head(topn)
    fig, ax = plt.subplots(figsize=(7, 0.4*len(counts) + 1.5))
    ax.barh(counts.index[::-1], counts.values[::-1])
    ax.set_xlabel('Number of books')
    ax.set_title('Top authors')
    plt.tight_layout()
    st.pyplot(fig)

def plot_rating_vs_count(books):
    if '__rating__' not in books.columns or 'ratings_count' not in books.columns:
        st.info('ratings_count or rating column not available for scatter plot.')
        return
    df = books.dropna(subset=['__rating__','ratings_count']).copy()
    if df.empty:
        st.info('No data for rating vs count plot.')
        return
    x = df['ratings_count'].astype(float)
    y = df['__rating__'].astype(float)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(x, y, alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Ratings count (log scale)')
    ax.set_ylabel('Rating')
    ax.set_title('Rating vs Ratings count')
    st.pyplot(fig)

def plot_publication_years(books, year_col_candidates=('original_publication_year','year','publication_year')):
    year_col = None
    for c in year_col_candidates:
        if c in books.columns:
            year_col = c
            break
    if year_col is None:
        st.info('No publication year column found.')
        return
    years = _pd.to_numeric(books[year_col], errors='coerce').dropna().astype(int)
    if years.empty:
        st.info('No valid publication years to plot.')
        return
    fig, ax = plt.subplots(figsize=(8,3))
    ax.hist(years, bins=30)
    ax.set_xlabel('Publication year')
    ax.set_ylabel('Count')
    ax.set_title('Publication year distribution')
    st.pyplot(fig)
# ------------------ end plot helpers ------------------


st.set_page_config(page_title='Book Recommender (single books.csv)', layout='wide')

# load meta to know which columns were auto-detected
try:
    meta = json.load(open('data_meta.json','r'))
except Exception:
    meta = None

@st.cache_data
def load_data():
    books = load_books()
    # detect defaults using columns from meta if available
    title_col = meta.get('title_col') if meta else None
    author_col = meta.get('author_col') if meta else None
    id_col = meta.get('id_col') if meta else None
    rating_col = meta.get('rating_col') if meta else None
    books_prepared = prepare_books(books, title_col or 'title', author_col or 'authors', id_col or 'book_id', rating_col)
    return books_prepared

books = load_data()
# models
content_model = ContentRecommender(method='count', max_features=8000).fit(books)

st.title('📚 Book Recommender ')
st.write('This app uses only the uploaded `data/books.csv`. Content-based recommendations + popularity fallback.')

with st.sidebar:
    st.header('Controls')
    mode = st.selectbox('Mode', ['Content (title)', 'Top popular'])
    topn = st.slider('Top N', 5, 30, 10)
    query = st.text_input('Enter book title (partial allowed):')
    fuzzy = st.checkbox('Use fuzzy match', True)

def fuzzy_title_search(q, choices, limit=10):
    res = process.extract(q, choices, scorer=fuzz.WRatio, limit=limit)
    return [r[0] for r in res]


# --- Optional data & charts ---
show_charts = st.sidebar.checkbox('Show data & charts', value=False)
if show_charts:
    st.header('📊 Data exploration')
    st.subheader('Sample of dataset')
    st.dataframe(books.head(200))
    st.subheader('Rating distribution')
    plot_rating_histogram(books)
    st.subheader('Top authors')
    plot_top_authors(books, topn=15)
    # rating vs count (if available)
    plot_rating_vs_count(books)
    st.subheader('Publication year distribution')
    plot_publication_years(books)
if st.button('Recommend'):
    start = time.time()
    if mode == 'Content (title)':
        if not query:
            st.warning('Please enter a title to search.')
        else:
            title = query
            if fuzzy:
                titles = books['title'].astype(str).tolist()
                matches = fuzzy_title_search(query, titles, limit=5)
                if matches:
                    st.write('Top matches:')
                    for m in matches:
                        st.write('-', m)
                    title = matches[0]
            recs = content_model.recommend(title, topn=topn)
            if not recs:
                st.info('No matches found.')
            else:
                st.subheader('Similar books to "{}"'.format(title))
                for r in recs:
                    rating = r.get('__rating__', None)
                    if rating is not None and not (pd.isna(rating)):
                        st.write('**{}** — {} (score: {:.3f}, rating: {})'.format(r['title'], r['authors'], r['score'], rating))
                    else:
                        st.write('**{}** — {} (score: {:.3f})'.format(r['title'], r['authors'], r['score']))
    else:
        recs = top_popular(books, topn=topn)
        st.subheader('Top popular books (from supplied rating/count if available)')
        for r in recs:
            rating = r.get('__rating__', None)
            if rating:
                st.write('**{}** — {} (rating: {})'.format(r['title'], r['authors'], rating))
            else:
                st.write('**{}** — {}'.format(r['title'], r['authors']))
    st.caption('Done in {:.2f}s'.format(time.time()-start))
