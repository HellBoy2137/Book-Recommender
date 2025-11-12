import numpy as np
import joblib
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder

class CollaborativeRecommender:
    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.fitted = False

    def fit(self, ratings_df):
        self.user_encoder = LabelEncoder().fit(ratings_df['user_id'])
        self.item_encoder = LabelEncoder().fit(ratings_df['book_id'])
        users = self.user_encoder.transform(ratings_df['user_id'])
        items = self.item_encoder.transform(ratings_df['book_id'])
        ratings = ratings_df['rating'].astype(float).values
        self.n_users = self.user_encoder.classes_.size
        self.n_items = self.item_encoder.classes_.size
        R = csr_matrix((ratings, (users, items)), shape=(self.n_users, self.n_items))
        n_comp = min(self.n_factors, max(1, min(self.n_users - 1, self.n_items - 1)))
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        U = svd.fit_transform(R)
        Sigma = svd.singular_values_
        VT = svd.components_
        self.user_factors = U * Sigma
        self.item_factors = VT.T
        self.fitted = True
        return self

    def predict(self, user_id, book_id):
        if not self.fitted:
            raise RuntimeError("Call fit() first")
        try:
            uidx = int(self.user_encoder.transform([user_id])[0])
            iidx = int(self.item_encoder.transform([book_id])[0])
        except ValueError:
            return 3.0
        score = float(self.user_factors[uidx].dot(self.item_factors[iidx]))
        return max(1.0, min(5.0, score))

    def recommend_for_user(self, user_id, topn=10):
        if not self.fitted:
            raise RuntimeError("Call fit() first")
        if user_id not in self.user_encoder.classes_:
            return []
        uidx = int(self.user_encoder.transform([user_id])[0])
        scores = self.user_factors[uidx].dot(self.item_factors.T)
        top_idxs = np.argsort(scores)[::-1][:topn]
        return [self.item_encoder.inverse_transform([i])[0] for i in top_idxs]

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)
