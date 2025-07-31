import pandas as pd
import numpy as np
import re
import joblib
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

class URLFeatureExtractor:
    def __init__(self, url):
        self.url = url
        self.parsed = urlparse(url)
        self.domain = self.parsed.netloc

    def get_features(self):

        f = {}
        f['url_length'] = len(self.url)
        f['domain_length'] = len(self.domain)
        f['path_length'] = len(self.parsed.path)

        special_chars = ['.', '-', '_', '/', '?', '=', '@', '&', '!', ' ', '~', ',', '+', '*', '#', '$', '%']
        for ch in special_chars:
            f[f'count_{ch if ch != " " else "space"}'] = self.url.count(ch)

        f['has_ip'] = 1 if self.is_ip() else 0
        f['has_https'] = 1 if self.parsed.scheme == 'https' else 0

        # ✅ Safe port detection
        try:
            f['has_port'] = 1 if self.parsed.port else 0
        except ValueError:
            f['has_port'] = 0  # Treat as "no port" if invalid

        f['has_suspicious_keywords'] = 1 if self.has_suspicious_keywords() else 0
        f['suspicious_tld'] = 1 if self.is_suspicious_tld() else 0
        f['entropy'] = self.calc_entropy(self.domain)
        f['is_shortened'] = 1 if self.is_shortened() else 0
        f['is_trusted_domain'] = 1 if self.is_trusted_domain() else 0

        return f


    def is_ip(self):
        return bool(re.match(r'^(\d{1,3}\.){3}\d{1,3}$', self.domain))

    def has_suspicious_keywords(self):
        keywords = ['login', 'signin', 'verify', 'update', 'ebay', 'paypal', 'bank', 'secure', 'account', 'credit']
        return any(k in self.url.lower() for k in keywords)

    def is_suspicious_tld(self):
        tlds = ['.xyz', '.top', '.gq', '.ml', '.tk', '.cf', '.ga', '.men']
        return any(self.url.endswith(tld) for tld in tlds)

    def is_shortened(self):
        return any(x in self.domain for x in ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly'])

    def is_trusted_domain(self):
        trusted = ['google.com', 'youtube.com', 'amazon.com', 'microsoft.com', 'facebook.com', 'linkedin.com']
        return any(domain in self.domain for domain in trusted)

    def calc_entropy(self, text):
        if not text:
            return 0
        probs = [text.count(c) / len(text) for c in set(text)]
        return -sum(p * np.log2(p) for p in probs)

class MaliciousURLDetector:
    def __init__(self, use_tfidf=True):
        self.model = None
        self.vectorizer = None
        self.tfidf_columns = []
        self.use_tfidf = use_tfidf

    def load_data(self, path):
        try:
            df = pd.read_csv(path)
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def preprocess(self, df):
        features = []
        labels = []

        for _, row in df.iterrows():
            url = row['url']
            label = str(row['type']).strip().lower()
            if not isinstance(url, str) or not url:
                continue

            extractor = URLFeatureExtractor(url)
            features.append(extractor.get_features())

            # Label: 0 = legitimate, 1 = phishing
            labels.append(1 if label != 'legitimate' else 0)

        X = pd.DataFrame(features)

        if self.use_tfidf:
            self.vectorizer = TfidfVectorizer(
                max_features=30,
                analyzer='char_wb',
                ngram_range=(3, 5),
                min_df=1
            )
            paths = [urlparse(u).path for u in df['url']]
            tfidf_matrix = self.vectorizer.fit_transform(paths).toarray()
            self.tfidf_columns = self.vectorizer.get_feature_names_out()
            tfidf_df = pd.DataFrame(tfidf_matrix, columns=self.tfidf_columns)
            X = pd.concat([X.reset_index(drop=True), tfidf_df], axis=1)

        return X, np.array(labels)

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=25,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        print("\n📊 Model Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(classification_report(y_test, y_pred))

    def predict(self, url):
        if not self.model:
            raise Exception("Model is not trained.")

        extractor = URLFeatureExtractor(url)
        features = pd.DataFrame([extractor.get_features()])

        if self.use_tfidf:
            path = urlparse(url).path
            tfidf = self.vectorizer.transform([path]).toarray()
            tfidf_df = pd.DataFrame(tfidf, columns=self.tfidf_columns)
            for col in self.tfidf_columns:
                if col not in tfidf_df.columns:
                    tfidf_df[col] = 0
            tfidf_df = tfidf_df[self.tfidf_columns]
            features = pd.concat([features, tfidf_df], axis=1)

        prob = self.model.predict_proba(features)[0]
        pred = self.model.predict(features)[0]

        return {
            'url': url,
            'prediction': 'MALICIOUS' if pred == 1 else 'BENIGN',
            'confidence': round(max(prob) * 100, 2),
            'probability_malicious': round(prob[1] * 100, 2)
        }

    def save(self, filename):
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer,
            'tfidf_columns': self.tfidf_columns,
            'use_tfidf': self.use_tfidf
        }, filename)
        print(f"✅ Model saved to {filename}")

    def load(self, filename):
        data = joblib.load(filename)
        self.model = data['model']
        self.vectorizer = data.get('vectorizer')
        self.tfidf_columns = data.get('tfidf_columns', [])
        self.use_tfidf = data.get('use_tfidf', True)
        print(f"✅ Model loaded from {filename}")

def main():
    detector = MaliciousURLDetector(use_tfidf=True)
    df = detector.load_data("URL_dataset (2).csv")
    if df is None or df.empty:
        print("❌ Dataset not found or empty.")
        return

    X, y = detector.preprocess(df)
    detector.train(X, y)
    detector.save("malicious_url_detector.joblib")

    print("\n🔎 Malicious URL Detection CLI")
    print("Type 'quit' to exit")

    while True:
        url = input("\nEnter a URL to check: ").strip()
        if url.lower() == 'quit':
            break
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        try:
            res = detector.predict(url)
            print(f"\n🔗 URL: {res['url']}")
            print(f"⚠️ Prediction: {res['prediction']}")
            print(f"📊 Malicious Probability: {res['probability_malicious']}%")
            print(f"🛡️ Confidence: {res['confidence']}%")
            if res['prediction'] == 'MALICIOUS':
                print("🚨 WARNING: This URL appears to be malicious!")
            else:
                print("✅ This URL appears safe.")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == '__main__':
    main()
