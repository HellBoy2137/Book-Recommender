import streamlit as st
import pandas as pd
import numpy as np
from src.utils import load_books
from src.preprocessing import prepare_books
from src.content_model import ContentRecommender
from src.popularity import top_popular
from rapidfuzz import process, fuzz
import json, time
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

st.title('📚 Book Recommender — Single `books.csv`')
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
