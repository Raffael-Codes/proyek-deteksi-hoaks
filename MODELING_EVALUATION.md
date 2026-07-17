# Dokumentasi Modeling & Evaluation: Deteksi Hoaks Politik

## 1. Implementasi Model (Skenario Pemodelan)
Berdasarkan hasil ekstraksi fitur teks menggunakan *TF-IDF Vectorizer* (5.000 fitur), tahap pemodelan dilakukan menggunakan 2 algoritma *Machine Learning* yang berbeda sebagai perbandingan performa:
1.  **Logistic Regression (Model Utama):** Algoritma parametrik linier yang secara teori sangat efisien dan tangguh dalam menangani data teks berdimensi tinggi berkat pendekatannya yang berbasis probabilitas.
2.  **Random Forest Classifier (Model Pembanding):** Algoritma *ensemble learning* berbasis pohon keputusan (*decision tree*) yang diekspektasikan mampu menangkap pola hubungan non-linier antar kata dalam kalimat.

## 2. Hyperparameter Tuning
Untuk memaksimalkan kapabilitas masing-masing model dan mencegah *overfitting*, dilakukan optimasi parameter menggunakan teknik **GridSearchCV** yang dikombinasikan dengan *5-Fold Cross Validation*.

*   **Logistic Regression:** Ruang pencarian (*search space*) difokuskan pada kekuatan regularisasi (`C`) dan jenis penalti (`penalty`). 
    *   Parameter terbaik yang diperoleh: `{'C': 1.0, 'penalty': 'l2'}`
*   **Random Forest:** Ruang pencarian difokuskan pada jumlah pohon (`n_estimators`) dan kedalaman maksimal pohon (`max_depth`). 
    *   Parameter terbaik yang diperoleh: `{'n_estimators': 100, 'max_depth': None}`

*(Catatan: Kode implementasi *training* dan *tuning* selengkapnya terdapat pada file `notebooks/modeling.ipynb`)*.

## 3. Evaluasi Komprehensif
Model yang telah di-*tuning* dievaluasi secara ketat pada 20% data uji (*Testing Set*) menggunakan metrik klasifikasi standar.

### Tabel Perbandingan Performa Model
| Metrik Evaluasi | Logistic Regression (Tuned) | Random Forest (Tuned) |
| :--- | :--- | :--- |
| **Accuracy** | **99.35%** | 97.80% |
| **Precision (Macro)** | **0.99** | 0.98 |
| **Recall (Macro)** | **0.99** | 0.97 |
| **F1-Score (Macro)** | **0.99** | 0.98 |
| **ROC-AUC Score** | **0.998** | 0.991 |

*(Catatan: Visualisasi grafis hasil evaluasi seperti kurva ROC dan Confusion Matrix terlampir secara menyeluruh di dalam *notebook* pemodelan)*.

## 4. Analisis Feature Importance & Interpretasi
Karena data berupa teks yang direpresentasikan dengan TF-IDF, interpretasi model dilakukan dengan menganalisis bobot koefisien (*coefficients*) dari persamaan Logistic Regression. Hal ini memperlihatkan kosa kata apa yang memicu keputusan model.

*   **Pemicu Kelas Hoaks (Koefisien Negatif Kuat):** Model mempelajari bahwa diksi yang bersifat provokatif, hiperbolis, atau bersifat instruksi emosional (contoh kata: "viralkan", "sebarkan", "rezim", "antek", "awas") berkorelasi sangat kuat dengan probabilitas berita hoaks.
*   **Pemicu Kelas Valid (Koefisien Positif Kuat):** Kosakata jurnalistik baku, rujukan waktu, dan kutipan (contoh kata: "mengatakan", "menurutnya", "rabu", "dilansir") menjadi jangkar utama bagi model untuk memprediksi bahwa teks tersebut adalah berita resmi.

## 5. Pemilihan Model Terbaik dan Justifikasi
Berdasarkan perbandingan kinerja dan kebutuhan arsitektur aplikasi, **Logistic Regression (Tuned)** secara mutlak dipilih sebagai *Best Model* yang di-*deploy* ke sistem *production* (Streamlit).

**Justifikasi Akademis dan Teknis:**
1.  **Dominasi Metrik:** Logistic Regression mengungguli Random Forest di seluruh matriks evaluasi, memvalidasi teori bahwa model linier bekerja sangat baik pada kumpulan fitur teks (TF-IDF) yang jumlahnya masif namun *sparse* (jarang).
2.  **Transparansi (White-box Model):** Keputusan Logistic Regression dapat dijelaskan dengan sangat mudah (*explainable*) melalui analisis koefisien, sangat cocok untuk audit verifikasi fakta, berbeda dengan Random Forest yang cenderung *black-box*.
3.  **Efisiensi *Deployment*:** Ukuran artefak model Logistic Regression (`logreg_model.pkl`) jauh lebih kecil di RAM dan memiliki waktu inferensi (*latency*) di bawah 100 ms, menjadikannya sangat ideal dan hemat biaya untuk diluncurkan di *cloud computing* gratis seperti Streamlit Community Cloud.