import numpy as np
from spacy.compat import pickle


class TFIDFClassifier:
    def __init__(self, model_path='models/linear_svc.pkl'):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)

    def predict(self, text):
        # Feature extraction
        X = self.vectorizer.transform([text])

        # Model prediction
        proba = self.model.predict_proba(X)[0][1]

        # Explainable output
        top_features = self.get_top_features(X)

        return {
            'prediction': int(proba > 0.5),
            'confidence': float(proba),
            'top_features': top_features
        }

    def get_top_features(self, X, top_n=5):
        coef = self.model.coef_.toarray().flatten()
        indices = X.indices
        scores = coef[indices] * X.data
        top_indices = np.argsort(-scores)[:top_n]

        return [
            (self.vectorizer.get_feature_names_out()[i], float(scores[i]))
            for i in top_indices
        ]