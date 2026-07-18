# Sistem Deteksi Hoaks Menggunakan Metode TF-IDF dan Algoritma Machine Learning

Proyek ini merupakan sebuah aplikasi berbasis *Machine Learning* dan *Natural Language Processing* (NLP) yang dirancang secara khusus untuk mengklasifikasikan teks berita politik berbahasa Indonesia ke dalam kategori **Hoaks** atau **Valid**. 

Proyek ini dikembangkan sebagai bentuk pemenuhan *Capstone Project* pada Ujian Akhir Semester (UAS) Genap 2025/2026 untuk Mata Kuliah Pembelajaran Mesin di Fakultas Ilmu Komputer. Proyek ini mendemonstrasikan implementasi *pipeline end-to-end* dalam penyelesaian masalah di dunia nyata, mulai dari akuisisi data, pemrosesan, pemodelan, evaluasi, hingga tahap *deployment*.

---

## 🚀 Fitur Utama Aplikasi
1. **Deteksi Teks Manual (Input Langsung):** Pengguna dapat mengetik atau menempelkan (*paste*) teks berita yang dicurigai secara langsung ke dalam kolom yang disediakan pada antarmuka web.
2. **Deteksi via Ekstraksi Gambar (Optical Character Recognition):** Pengguna dapat mengunggah gambar hasil tangkapan layar (*screenshot*) dari artikel berita atau pesan berantai. Sistem akan secara otomatis mengekstrak teks di dalam gambar tersebut menggunakan integrasi **Tesseract OCR** sebelum dianalisis oleh model *Machine Learning*.
3. **Interpretasi & Evaluasi Model Interaktif:** Aplikasi menyediakan transparansi metrik performa model dan hasil prediksi, memungkinkan pengguna dan *stakeholder* untuk memahami bagaimana keputusan klasifikasi diambil.
4. **Antarmuka Pengguna Responsif:** Dibangun menggunakan *framework* Streamlit untuk menghasilkan purwarupa aplikasi web yang cepat, ringan, dan sangat mudah digunakan.

---

## 🛠️ Teknologi & Arsitektur yang Digunakan
Sistem ini dibangun dengan tumpukan teknologi modern berstandar industri sains data:
* **Bahasa Pemrograman Utama:** Python
* **Pemrosesan & Manipulasi Data:** Pandas, NumPy, Openpyxl (untuk membaca format Excel)
* **Algoritma Machine Learning Utama:** Logistic Regression (Dipilih karena keseimbangan optimal antara akurasi tinggi dan interpretabilitas model)
* **Algoritma Komparasi:** Multinomial Naive Bayes
* **Ekstraksi Fitur Teks (NLP):** TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer
* **Computer Vision & Pemrosesan Gambar:** Pytesseract (Tesseract OCR)
* **Penyimpanan Objek Model:** Joblib
* **Web Framework & Deployment:** Streamlit

---

## 📂 Struktur Direktori Proyek
Repositori ini disusun secara terstruktur, modular, dan mengikuti prinsip rekayasa perangkat lunak yang baik sesuai dengan panduan dan standar proyek data:

```text
proyek-deteksi-hoaks/
├── data/
│   ├── raw/                   # Direktori penyimpanan data mentah (.xlsx)
│   └── processed/             # Direktori penyimpanan data yang telah dibersihkan (.csv)
├── notebooks/                 
│   ├── 01_eda.ipynb           # Eksplorasi Data (EDA) dan analisis awal
│   ├── 02_modeling.ipynb      # Eksperimen pemodelan Machine Learning
│   └── 03_interpretation.ipynb# Uji coba manual dan interpretasi prediksi
├── src/                       
│   ├── data_preprocessing.py  # Skrip otomatisasi pembersihan dan penggabungan data
│   ├── train_model.py         # Skrip otomatisasi ekstraksi fitur dan pelatihan model
│   ├── evaluate_model.py      # Skrip evaluasi performa dan cross-validation
│   └── utils.py               # Kumpulan fungsi modular pembantu (seperti regex pembersih teks)
├── models/                    
│   ├── logreg_model.pkl       # Berkas model Logistic Regression terbaik yang siap pakai
│   └── tfidf_vectorizer.pkl   # Berkas pemroses vektor teks (Pipeline)
├── app/                       
│   └── app.py                 # Kode sumber utama antarmuka web Streamlit
├── README.md                  # Dokumentasi komprehensif proyek
└── requirements.txt           # Daftar dependensi library Python
```

---

## 📊 Informasi & Cara Mengunduh Dataset

Model dilatih menggunakan dataset publik **Indonesian Fact and Hoax Political News** dari Kaggle. Dataset ini dipilih karena sangat relevan dengan domain klasifikasi teks berita.

* **Label `Valid`**: Data diekstraksi secara terstruktur dari portal berita arus utama dan resmi di Indonesia (CNN Indonesia, Kompas, Tempo).
* **Label `Hoaks`**: Data dikumpulkan dari arsip situs web independen pengecekan fakta terkemuka (TurnBackHoax.id).
* *Catatan Domain:* Ruang lingkup model ini secara khusus difokuskan pada narasi bertema politik, kebijakan pemerintahan, dan isu-isu nasional terkini.

**Langkah-langkah Penyiapan Dataset Lokal:**
1. Kunjungi tautan dataset Kaggle berikut: [Indonesian Fact and Hoax Political News](https://www.kaggle.com/datasets/linkgish/indonesian-fact-and-hoax-political-news/data)
2. Autentikasi menggunakan akun Kaggle Anda, lalu klik tombol **Download**.
3. Ekstrak arsip berformat `.zip` yang telah diunduh ke komputer Anda.
4. **SANGAT PENTING:** Pindahkan keempat berkas berekstensi `.xlsx` (`dataset_cnn_10k.xlsx`, `dataset_kompas_4k.xlsx`, `dataset_tempo_6k.xlsx`, `dataset_turnbackhoax_10k.xlsx`) ke dalam direktori **`data/raw/`** di dalam repositori ini agar skrip *pipeline* dapat membaca berkas dengan benar.

---

## ⚙️ Panduan Instalasi & Menjalankan Pipeline Machine Learning

Ikuti langkah-langkah sistematis berikut untuk mengkloning, membangun ulang model, dan menjalankan aplikasi di lingkungan lokal (berbasis Windows).

### 1. Persiapan Lingkungan Tesseract OCR (Kebutuhan Esensial)
Fitur deteksi gambar dalam aplikasi ini bergantung pada mesin Tesseract.
* Unduh paket instalasi Tesseract OCR untuk Windows dari repositori resmi: [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
* Jalankan proses instalasi dan pastikan aplikasi terpasang di direktori bawaan sistem: `C:\Program Files\Tesseract-OCR\tesseract.exe`. (Aplikasi web akan mencari berkas `.exe` ini di jalur tersebut).

### 2. Kloning Repositori & Instalasi Dependensi
Buka terminal (seperti Command Prompt, PowerShell, atau terminal terintegrasi VS Code) dan eksekusi perintah berikut secara berurutan:

```bash
# Mengunduh salinan repositori ini ke dalam direktori lokal Anda
git clone [https://github.com/Raffael-Codes/proyek-deteksi-hoaks.git](https://github.com/Raffael-Codes/proyek-deteksi-hoaks.git)

# Berpindah ke dalam direktori proyek yang baru saja diunduh
cd proyek-deteksi-hoaks

# Menginstal seluruh library Python yang dipersyaratkan oleh sistem
pip install pandas scikit-learn streamlit pytesseract joblib openpyxl
# Alternatif jika Anda menggunakan file requirements: pip install -r requirements.txt
```

### 3. Menjalankan Ulang Pipeline Machine Learning End-to-End
Proyek ini mengadopsi arsitektur *pipeline* terpisah untuk efisiensi komputasi. Anda wajib menjalankan skrip ini dari *root folder* (beranda utama proyek) secara berurutan:

```bash
# Tahap 1: Membersihkan data mentah Excel dan menyimpannya menjadi CSV yang ringan
python src/data_preprocessing.py

# Tahap 2: Membaca CSV bersih, mengekstraksi TF-IDF, melatih algoritma, dan menyimpan file .pkl
python src/train_model.py

# Tahap 3: Melakukan validasi pengujian akhir dan 5-Fold Cross Validation
python src/evaluate_model.py
```

### 4. Menjalankan Aplikasi Web (Streamlit)
Setelah model (`.pkl`) berhasil diperbarui di folder `models/`, jalankan antarmuka *front-end* dengan perintah:

```bash
# Menjalankan server lokal Streamlit
python -m streamlit run app/app.py
```
*Tautan lokal (umumnya `http://localhost:8501`) akan otomatis terbuka di peramban web bawaan Anda.*

---

## 📈 Performa Evaluasi Model
Berdasarkan hasil pengujian pada skrip `evaluate_model.py`, algoritma **Logistic Regression** yang telah melalui proses pelatihan menunjukkan kinerja klasifikasi yang luar biasa solid dalam memisahkan teks berita, dengan metrik keberhasilan sebagai berikut:

* **Akurasi Utama (Testing):** `99.35%`
* **Rata-rata Akurasi Validasi Silang (5-Fold Cross Validation):** `98.91%`
* **Skor Precision (Rata-rata tertimbang):** `0.99`
* **Skor Recall (Rata-rata tertimbang):** `0.99`
* **Skor F1 (Rata-rata tertimbang):** `0.99`

Model ini menunjukkan konsistensi prediksi yang nyaris sempurna tanpa indikasi *overfitting* yang parah berkat mekanisme regularisasi di dalam Logistic Regression, menjadikannya sangat andal untuk diluncurkan ke tahap produksi *deployment*.

---

## 🌐 Akses Aplikasi Publik (Deployment)

Aplikasi deteksi hoaks ini telah di-*deploy* menggunakan infrastruktur **Streamlit Community Cloud**. Anda dapat mengakses, menguji, dan mendemonstrasikan sistem ini secara langsung dari perangkat apa pun tanpa perlu melakukan instalasi lokal melalui tautan berikut:

🔗 **https://hoax-detector-nlp.streamlit.app/**
