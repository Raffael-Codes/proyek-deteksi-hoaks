# Sistem Deteksi Hoaks Menggunakan Metode TF-IDF dan Algoritma Machine Learning

Proyek ini adalah aplikasi berbasis web (*Machine Learning*) yang dirancang untuk mengklasifikasikan teks berita politik berbahasa Indonesia ke dalam kategori **Hoaks** atau **Valid**. 

## 🚀 Fitur Utama
1. **Deteksi Teks Manual:** Pengguna dapat mengetik atau menempelkan (*paste*) teks berita secara langsung.
2. **Deteksi via Gambar (OCR):** Pengguna dapat mengunggah gambar hasil tangkapan layar (*screenshot*) berita. Sistem akan otomatis mengekstrak teks di dalam gambar tersebut menggunakan Tesseract OCR sebelum dianalisis.
3. **Antarmuka Interaktif:** Dibangun menggunakan Streamlit sehingga mudah dan cepat digunakan.

## 🛠️ Teknologi yang Digunakan
* **Bahasa Pemrograman:** Python
* **Algoritma Machine Learning:** Logistic Regression
* **Ekstraksi Fitur (NLP):** TF-IDF Vectorizer
* **Computer Vision:** Tesseract OCR (Pytesseract)
* **Web Framework:** Streamlit
* **Library Data:** Pandas, Scikit-learn, Joblib

## 📊 Informasi Dataset
Model dilatih menggunakan dataset publik **Indonesian Fact and Hoax Political News** dari Kaggle. 
* Label **Valid**: Data diambil dari portal berita resmi (CNN Indonesia, Kompas, Tempo).
* Label **Hoaks**: Data diambil dari arsip situs pengecekan fakta (TurnBackHoax.id).
* *Catatan:* Model ini berfokus pada narasi bertema politik, pemerintahan, dan isu nasional.

