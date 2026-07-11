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
* **Library Data:** Pandas, Scikit-learn, Joblib, Openpyxl

---

## 📊 Informasi & Cara Mengunduh Dataset

Model dilatih menggunakan dataset publik **Indonesian Fact and Hoax Political News** dari Kaggle. 
* Label **Valid**: Data diambil dari portal berita resmi (CNN Indonesia, Kompas, Tempo).
* Label **Hoaks**: Data diambil dari arsip situs pengecekan fakta (TurnBackHoax.id).
* *Catatan:* Model ini berfokus pada narasi bertema politik, pemerintahan, dan isu nasional.

**Langkah-langkah Mengunduh Dataset:**
1. Kunjungi tautan dataset Kaggle berikut: https://www.kaggle.com/datasets/linkgish/indonesian-fact-and-hoax-political-news/data
2. Login menggunakan akun Kaggle Anda, lalu klik tombol **Download**.
3. Ekstrak file berformat `.zip` yang telah diunduh.
4. Pindahkan keempat file dataset berekstensi `.xlsx` tersebut (`dataset_cnn_10k.xlsx`, `dataset_kompas_4k.xlsx`, `dataset_tempo_6k.xlsx`, `dataset_turnbackhoax_10k.xlsx`) ke dalam folder `data/` di dalam repositori ini.

---

## ⚙️ Cara Menjalankan Proyek (Instalasi)

Ikuti panduan berikut untuk menjalankan aplikasi ini di komputer lokal (Windows).

### 1. Persiapan Tesseract OCR (Wajib)
Aplikasi ini memerlukan Tesseract untuk membaca teks dari gambar.
* Unduh *installer* Tesseract OCR untuk Windows dari [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
* Lakukan instalasi dan pastikan aplikasi terinstal di lokasi bawaan: `C:\Program Files\Tesseract-OCR\tesseract.exe`.

### 2. Kloning Repositori & Instalasi Library
Buka terminal (Command Prompt / VS Code) dan jalankan perintah berikut secara berurutan:

```bash
# Kloning repositori ini ke komputer Anda
git clone [https://github.com/Raffael-Codes/proyek-deteksi-hoaks.git](https://github.com/Raffael-Codes/proyek-deteksi-hoaks.git)

# Masuk ke dalam folder proyek
cd proyek-deteksi-hoaks

# Instalasi semua library Python yang dibutuhkan
pip install pandas scikit-learn streamlit pytesseract joblib openpyxl
