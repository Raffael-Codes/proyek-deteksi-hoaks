import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_and_clean_data

def evaluate():
    # 1. Ambil data bersih
    df = load_and_clean_data()
    X = df['Teks Bersih']
    y = df['Label']

    # 2. Load model yang sudah disave
    vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')
    model = joblib.load('../models/logreg_model.pkl')

    # 3. Transformasi dan Splitting untuk evaluasi
    X_tfidf = vectorizer.transform(X)
    _, X_test_tfidf, _, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    print("\n==============================================")
    print("       HASIL EVALUASI LOGISTIC REGRESSION     ")
    print("==============================================")
    
    # 4. Prediksi dan Laporan Klasifikasi
    prediksi = model.predict(X_test_tfidf)
    print(f"Akurasi Utama : {accuracy_score(y_test, prediksi) * 100:.2f}%\n")
    print("Metrik Pendukung (Precision, Recall, F1-Score):")
    print(classification_report(y_test, prediksi))

    print("\n--- Melakukan 5-Fold Cross Validation ---")
    
    # 5. Ujian Silang
    scores = cross_val_score(model, X_tfidf, y, cv=5, scoring='accuracy')
    for i, skor in enumerate(scores):
        print(f"Ujian ke-{i+1}: {skor * 100:.2f}%")
        
    print(f"\nRata-rata Akurasi Validasi: {scores.mean() * 100:.2f}%")

if __name__ == "__main__":
    evaluate()