import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import pickle
import re
import os

class SpamCallDetector:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def preprocess_phone_number(self, phone_number):
        phone_str = str(phone_number)
        digits_only = re.sub(r'\D', '', phone_str)
        if len(digits_only) == 0:
            digits_only = "0"
        features = {
            'length': len(digits_only),
            'starts_with_toll_free': 1 if digits_only.startswith(('800','888','877','866','855','844','833','822')) else 0,
            'has_repeating_digits': 1 if len(set(digits_only)) < max(1, int(len(digits_only) * 0.7)) else 0,
            'sequential_digits': self._count_sequential_digits(digits_only)
        }
        return features

    def _count_sequential_digits(self, digits):
        count = 0
        digits = digits if digits is not None else ""
        for i in range(len(digits) - 2):
            try:
                a, b, c = int(digits[i]), int(digits[i+1]), int(digits[i+2])
                if a + 1 == b and b + 1 == c:
                    count += 1
            except ValueError:
                continue
        return count

    def preprocess_message(self, message):
        if pd.isna(message) or str(message).strip() == '':
            return ''
        message = str(message).lower()
        message = re.sub(r'[^a-z\s]', '', message)
        message = re.sub(r'\s+', ' ', message).strip()
        return message

    def create_numeric_features(self, df):
        rows = []
        for _, row in df.iterrows():
            pf = self.preprocess_phone_number(row.get('phone_number', ''))
            duration = float(row.get('duration', 0) if pd.notna(row.get('duration', None)) else 0)
            call_freq = float(row.get('call_frequency', 0) if pd.notna(row.get('call_frequency', None)) else 0)
            rows.append([
                pf['length'], pf['starts_with_toll_free'], pf['has_repeating_digits'],
                pf['sequential_digits'], duration, call_freq
            ])
        return np.array(rows, dtype=float)

    def train(self, data):
        data = data.copy()
        data['processed_message'] = data['message'].apply(self.preprocess_message)
        numeric = self.create_numeric_features(data)
        numeric_scaled = self.scaler.fit_transform(numeric)
        message_feat = self.vectorizer.fit_transform(data['processed_message']).toarray()
        X = np.hstack([numeric_scaled, message_feat])
        y = self.label_encoder.fit_transform(data['label'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("Model Accuracy:", acc)
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        self.is_trained = True
        return acc

    def predict(self, phone_number, duration, call_frequency, message):
        if not self.is_trained:
            raise RuntimeError("Model is not trained or loaded.")
        df = pd.DataFrame([{
            'phone_number': phone_number,
            'duration': duration,
            'call_frequency': call_frequency,
            'message': message
        }])
        df['processed_message'] = df['message'].apply(self.preprocess_message)
        numeric = self.create_numeric_features(df)
        numeric_scaled = self.scaler.transform(numeric)
        message_feat = self.vectorizer.transform(df['processed_message']).toarray()
        X = np.hstack([numeric_scaled, message_feat])
        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[0] if hasattr(self.model, "predict_proba") else None
        label = self.label_encoder.inverse_transform([pred])[0]
        confidence = max(prob) if prob is not None else None
        return label, confidence

    def save_model(self, filename='spam_detector_model.pkl'):
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'is_trained': self.is_trained
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filename='spam_detector_model.pkl'):
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.vectorizer = model_data['vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.is_trained = model_data.get('is_trained', True)
