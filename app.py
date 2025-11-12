# app.py
import streamlit as st
import pandas as pd
import numpy as np
from src.utils import load_books, load_ratings, load_users
from src.preprocessing import prepare_books
from src.content_model import ContentRecommender
from src.collaborative_model import CollaborativeRecommender
from rapidfuzz import process, fuzz
import time

st.set_page_config(page_title='📚 Book Recommender', layout='wide')

@st.cache_data
def load_data():
    books = load_books()
    ratings = load_ratings()
    users = load_users()
    books = prepare_books(books)
    # make sure ids are consistent
    ratings['book_id'] = ratings['book_id'].astype(str)
    ratings['user_id'] = ratings['user_id'].astype(str)
    if 'rating' not in ratings.columns and 'ratings' in ratings.columns:
        ratings.rename(columns={'ratings': 'rating'}, inplace=True)
    ratings['rating'] = ratings['rating'].astype(float)
    return books, ratings, users

@st.cache_resource
def build_models(books, ratings):
    content = ContentRecommender(max_features=15000, ngram_range=(1,2)).fit(books)
    cf = CollaborativeRecommender(n_factors=60).fit(ratings[['user_id','book_id','rating']])
    return content, cf

books, ratings, users = load_data()
content_model, cf_model = build_models(books, ratings)

st.title("📚 Book Recommender — Goodbooks")
st.write("Hybrid system: **Content-based** (CountVectorizer + Cosine) and **Collaborative Filtering** (TruncatedSVD).")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    mode = st.selectbox("Select mode", ["Content (title)", "Collaborative (user)", "Hybrid (title + user)"])
    topn = st.slider("Top N recommendations", 5, 30, 10)
    alpha = st.slider("Hybrid weight (content α)", 0.0, 1.0, 0.6)
    query = st.text_input("Enter title or user id:")
    fuzzy = st.checkbox("Use fuzzy title match", True)

# --- Helper for fuzzy search ---
def fuzzy_title_search(q, choices, limit=10):
    results = process.extract(q, choices, scorer=fuzz.WRatio, limit=limit)
    return [r[0] for r in results]

# --- Main logic ---
if st.button("🔍 Recommend"):
    start = time.time()
    if mode == "Content (title)":
        if not query:
            st.warning("Please enter a book title.")
        else:
            if fuzzy:
                titles = books['title'].astype(str).tolist()
                matches = fuzzy_title_search(query, titles, limit=5)
                if matches:
                    st.caption("Did you mean:")
                    for m in matches:
                        st.write(f"- {m}")
                    title = matches[0]
                else:
                    title = query
            else:
                title = query

            recs = content_model.recommend(title, topn=topn)
            if not recs:
                st.info("No similar books found.")
            else:
                st.subheader(f"📖 Similar to **{title}**")
                for r in recs:
                    st.write(f"**{r['title']}** — {r['authors']} (score: {r['score']:.3f})")

    elif mode == "Collaborative (user)":
        if not query:
            st.warning("Please enter a user ID.")
        else:
            user_id = str(query).strip()
            rec_ids = cf_model.recommend_for_user(user_id, topn=topn)
            if not rec_ids:
                st.info("No recommendations for this user (cold start).")
            else:
                st.subheader(f"👤 Recommendations for User {user_id}")
                for bid in rec_ids:
                    row = books[books['book_id'] == str(bid)]
                    if not row.empty:
                        st.write(f"**{row.iloc[0]['title']}** — {row.iloc[0]['authors']}")

    else:  # Hybrid
        if not query:
            st.warning("Enter a book title to start hybrid recommendations.")
        else:
            title = query
            content_recs = content_model.recommend(title, topn=topn * 3)
            if not content_recs:
                st.info("No content match found.")
            else:
                user_field = st.text_input("Optional: enter user ID for personalization")
                merged = []
                for c in content_recs:
                    bid = c["book_id"]
                    content_score = c["score"]
                    if user_field:
                        cf_score = cf_model.predict(user_field, bid)
                    else:
                        avg = ratings[ratings["book_id"] == str(bid)]["rating"].mean()
                        cf_score = avg if not np.isnan(avg) else 3.0
                    final = alpha * content_score + (1 - alpha) * (cf_score / 5.0)
                    merged.append({
                        "book_id": bid,
                        "title": c["title"],
                        "authors": c["authors"],
                        "score": final
                    })
                merged = sorted(merged, key=lambda x: x["score"], reverse=True)[:topn]
                st.subheader(f"🔗 Hybrid Recommendations for **{title}** (α={alpha:.2f})")
                for r in merged:
                    st.write(f"**{r['title']}** — {r['authors']} (score: {r['score']:.3f})")

    st.caption(f"⏱️ Completed in {time.time() - start:.2f} seconds")
