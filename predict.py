import os
import joblib
from preprocess import clean_text

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')


class NewsPredictor:
    def __init__(self, model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.is_loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.vectorizer_path):
            return
        
        try:
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.is_loaded = True
        except Exception as e:
            print(f"Error loading prediction artifacts: {e}")
            self.is_loaded = False

    def predict(self, text: str):
        """
        Takes raw news headline or article text.
        Returns dictionary containing:
        - label: 'REAL' or 'FAKE'
        - confidence: float (0.0 to 100.0)
        - raw_prediction: 0 (Fake) or 1 (Real)
        - cleaned_text: preprocessed text
        """
        if not self.is_loaded:
            # Try reloading if not loaded
            self._load_artifacts()
            if not self.is_loaded:
                raise FileNotFoundError("Model or TF-IDF Vectorizer files missing. Please run train_model.py first.")

        cleaned = clean_text(text)
        if not cleaned:
            # Fallback for empty text
            return {
                'label': 'UNKNOWN',
                'confidence': 0.0,
                'raw_prediction': -1,
                'cleaned_text': ''
            }

        tfidf_features = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(tfidf_features)[0]

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(tfidf_features)[0]
            confidence = probabilities[prediction] * 100.0
        else:
            confidence = 100.0

        label = 'REAL' if prediction == 1 else 'FAKE'

        return {
            'label': label,
            'confidence': round(confidence, 2),
            'raw_prediction': int(prediction),
            'cleaned_text': cleaned
        }


def predict_news(text: str):
    predictor = NewsPredictor()
    return predictor.predict(text)


if __name__ == '__main__':
    sample_input = "The government announces free electricity for all citizens."
    print("\n--- SAMPLE PREDICTION TEST ---")
    print(f"Input Text: \"{sample_input}\"")
    try:
        res = predict_news(sample_input)
        print("\nOutput:")
        print(f"Prediction: {res['label']}")
        print(f"Confidence: {res['confidence']}%")
    except Exception as e:
        print(f"Prediction module status: {e}")
