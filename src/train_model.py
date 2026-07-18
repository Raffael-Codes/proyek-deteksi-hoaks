import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def run_training():
    print("--- Membaca Data Bersih ---")
    # Hapus ../
    df = pd.read_csv('data/processed/dataset_bersih.csv')
    df.dropna(subset=['Teks Bersih'], inplace=True)
    
    X = df['Teks Bersih']
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--- Memulai Pelatihan Model ---")
    print("Mengubah teks menjadi vektor...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    print("Mengajari model Logistic Regression...")
    logreg_model = LogisticRegression(max_iter=1000, random_state=42)
    logreg_model.fit(X_train_tfidf, y_train)

    # Hapus ../
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    joblib.dump(logreg_model, 'models/logreg_model.pkl')
    
    print("✅ File 'tfidf_vectorizer.pkl' dan 'logreg_model.pkl' berhasil diperbarui di folder models!")

if __name__ == "__main__":
    run_training()