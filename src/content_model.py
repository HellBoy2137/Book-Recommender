import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentRecommender:
    def __init__(self, max_features=15000, ngram_range=(1,2)):
        self.vectorizer = CountVectorizer(max_features=max_features,
                                          ngram_range=ngram_range,
                                          stop_words='english')
        self.matrix = None
        self.books_df = None
        self.fitted = False

    def fit(self, books_df):
        self.books_df = books_df.reset_index(drop=True)
        texts = self.books_df['combined'].fillna('')
        self.matrix = self.vectorizer.fit_transform(texts)
        self.fitted = True
        return self

    def recommend(self, title_query, topn=10):
        if not self.fitted:
            raise RuntimeError("Call fit() first")
        q = title_query.strip().lower()
        matches = self.books_df[self.books_df['title'].str.lower() == q]
        if matches.empty:
            matches = self.books_df[self.books_df['title'].str.lower().str.contains(q)]
            if matches.empty:
                q_vec = self.vectorizer.transform([q])
                sims = cosine_similarity(q_vec, self.matrix).flatten()
                top_idxs = sims.argsort()[-topn:][::-1]
                results = self.books_df.iloc[top_idxs].copy()
                results['score'] = sims[top_idxs]
                return results[['book_id','title','authors','score']].to_dict(orient='records')
        idx = matches.index[0]
        sims = cosine_similarity(self.matrix[idx], self.matrix).flatten()
        top_idxs = sims.argsort()[-topn-1:-1][::-1]
        results = self.books_df.iloc[top_idxs].copy()
        results['score'] = sims[top_idxs]
        return results[['book_id','title','authors','score']].to_dict(orient='records')

    def save(self, path):
        joblib.dump({'vectorizer': self.vectorizer, 'matrix': self.matrix, 'books': self.books_df}, path)

    def load(self, path):
        d = joblib.load(path)
        self.vectorizer = d['vectorizer']
        self.matrix = d['matrix']
        self.books_df = d['books']
        self.fitted = True
        return self
