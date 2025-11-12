import streamlit as st
import pandas as pd
import numpy as np
from src.utils import load_books, load_ratings, load_users
from src.preprocessing import prepare_books
from src.content_model import ContentRecommender
from src.collaborative_model import CollaborativeRecommender
from rapidfuzz import process, fuzz
import time

st.set_page_config(page_title='Book Recommender (New Dataset)', layout='wide')

@st.cache_data
def load_data():
    books = load_books()
    ratings = load_ratings()
    users = load_users()
    books = prepare_books(books)
    ratings['book_id'] = ratings['book_id'].astype(str)
    ratings['user_id'] = ratings['user_id'].astype(str)
    if 'rating' not in ratings.columns and 'ratings' in ratings.columns:
        ratings.rename(columns={'ratings':'rating'}, inplace=True)
    ratings['rating'] = ratings['rating'].astype(float)
    return books, ratings, users

@st.cache_resource
def build_models(books, ratings, method='count'):
    # content model (count or tfidf)
    content = ContentRecommender(method=method, max_features=15000, ngram_range=(1,2)).fit(books)
    cf = CollaborativeRecommender(n_factors=60).fit(ratings[['user_id','book_id','rating']])
    return content, cf

books, ratings, users = load_data()
content_model, cf_model = build_models(books, ratings, method='count')

st.title('📚 Book Recommender — New Dataset')
st.write('Content (CountVectorizer/TF-IDF) + Collaborative (TruncatedSVD)')

with st.sidebar:
    st.header('Controls')
    mode = st.selectbox('Mode', ['Content (title)', 'Collaborative (user)', 'Hybrid (title + user)'])
    vector_method = st.selectbox('Content vectorizer', ['count', 'tfidf'])
    topn = st.slider('Top N', 5, 30, 10)
    alpha = st.slider('Content weight (alpha)', 0.0, 1.0, 0.6)
    query = st.text_input('Enter title or user id:')
    fuzzy = st.checkbox('Use fuzzy title match', value=True)

def fuzzy_title_search(q, choices, limit=10):
    res = process.extract(q, choices, scorer=fuzz.WRatio, limit=limit)
    return [r[0] for r in res]

if st.button('Recommend'):
    start = time.time()
    if mode == 'Content (title)':
        if not query:
            st.warning('Enter a book title.')
        else:
            title = query
            if fuzzy:
                choices = books['title'].astype(str).tolist()
                matches = fuzzy_title_search(query, choices, limit=5)
                if matches:
                    title = matches[0]
            recs = content_model.recommend(title, topn=topn)
            if not recs:
                st.info('No matches.')
            else:
                st.subheader(f'Content recommendations for "{title}"')
                for r in recs:
                    st.write(f"**{r['title']}** — {r['authors']} (score: {r['score']:.3f})")
    elif mode == 'Collaborative (user)':
        if not query:
            st.warning('Enter user id.')
        else:
            recs = cf_model.recommend_for_user(str(query), topn=topn)
            if not recs:
                st.info('No recs for this user.')
            else:
                st.subheader(f'Collaborative recommendations for user {query}')
                for bid in recs:
                    row = books[books['book_id'] == str(bid)]
                    if not row.empty:
                        st.write(f"**{row.iloc[0]['title']}** — {row.iloc[0]['authors']}")
    else:
        if not query:
            st.warning('Enter a title for hybrid mode.')
        else:
            title = query
            if fuzzy:
                choices = books['title'].astype(str).tolist()
                matches = fuzzy_title_search(query, choices, limit=5)
                if matches:
                    title = matches[0]
            content_recs = content_model.recommend(title, topn=topn*3)
            if not content_recs:
                st.info('No content match.')
            else:
                user_field = st.text_input('Optional: enter user id for personalization')
                merged = []
                for c in content_recs:
                    bid = c['book_id']
                    content_score = c['score']
                    if user_field:
                        cf_score = cf_model.predict(user_field, bid)
                    else:
                        df = ratings[ratings['book_id'] == str(bid)]
                        cf_score = float(df['rating'].mean()) if not df.empty else 3.0
                    hybrid_score = alpha * content_score + (1 - alpha) * (cf_score / 5.0)
                    merged.append({'book_id': bid, 'title': c['title'], 'authors': c['authors'], 'score': hybrid_score})
                merged = sorted(merged, key=lambda x: x['score'], reverse=True)[:topn]
                st.subheader(f'Hybrid recommendations for "{title}" (alpha={alpha:.2f})')
                for r in merged:
                    st.write(f"**{r['title']}** — {r['authors']} (score: {r['score']:.3f})")
    st.write(f"Done in {time.time()-start:.2f}s")
