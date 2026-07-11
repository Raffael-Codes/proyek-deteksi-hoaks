import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from data_preprocessing import load_and_clean_data

def run_training():
    # 1. Ambil data bersih
    df = load_and_clean_data()
    
    # 2. Tentukan X dan y (Menggunakan Teks Bersih sesuai notebook Anda)
    X = df['Teks Bersih']
    y = df['Label']

    # 3. Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--- Memulai Pelatihan Model ---")
    
    # 4. Ekstraksi Fitur (TF-IDF dengan max_features=5000)
    print("Mengubah teks menjadi vektor...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # 5. Latih Algoritma Logistic Regression
    print("Mengajari model Logistic Regression...")
    logreg_model = LogisticRegression(max_iter=1000, random_state=42)
    logreg_model.fit(X_train_tfidf, y_train)

    # 6. Simpan Model
    joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')
    joblib.dump(logreg_model, '../models/logreg_model.pkl')
    
    print("✅ File 'tfidf_vectorizer.pkl' dan 'logreg_model.pkl' berhasil diperbarui di folder models!")

if __name__ == "__main__":
    run_training()