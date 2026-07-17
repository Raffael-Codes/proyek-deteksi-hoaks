# Problem Statement: Deteksi Hoaks Politik Berbahasa Indonesia

## 1. Latar Belakang

Penyebaran berita bohong (hoaks), terutama di ranah politik, telah menjadi salah satu ancaman terbesar bagi stabilitas sosial di era digital. Kecepatan penyebaran informasi melalui media sosial seringkali tidak diimbangi dengan proses verifikasi fakta yang memadai. Menurut berbagai pengamat literasi digital, masyarakat seringkali kesulitan membedakan antara berita jurnalistik yang valid dan narasi provokatif yang telah dimanipulasi.

Proses pengecekan fakta (*fact-checking*) yang dilakukan oleh jurnalis atau dewan pers saat ini masih banyak yang berjalan secara manual. Hal ini tentu memakan waktu yang lama dan membuat kewalahan ketika menghadapi volume informasi yang sangat masif, terutama menjelang tahun-tahun politik. 

Untuk mengatasi masalah ini, teknologi pemrosesan bahasa alami (*Natural Language Processing* / NLP) dan *Machine Learning* dapat dimanfaatkan. Dengan mengidentifikasi pola struktur kalimat, frekuensi kata, dan gaya bahasa secara komputasional, kita dapat mendeteksi indikasi hoaks secara *real-time* dan objektif.

## 2. Tujuan Bisnis/Analisis

**Tujuan utama:** Membangun sistem deteksi teks berbasis *Machine Learning* yang mampu mengklasifikasikan kredibilitas berita politik berbahasa Indonesia ke dalam 2 kategori:

| Kelas | Label | Deskripsi |
| :--- | :--- | :--- |
| 0 | Hoaks | Berita terindikasi palsu, hiperbolis, manipulatif, atau provokatif |
| 1 | Valid | Berita dari sumber jurnalistik resmi yang mengikuti kaidah bahasa baku |

**Tujuan analitis:**
* Mengekplorasi dan membersihkan data teks bahasa Indonesia (penghapusan *missing values*, *case folding*, dan *regex*).
* Mengekstraksi fitur teks menjadi vektor angka menggunakan metode pembobotan TF-IDF (berfokus pada 5.000 kata terpenting).
* Membangun model klasifikasi dengan akurasi tinggi menggunakan algoritma *Logistic Regression*.
* Memastikan stabilitas model agar tidak mengalami *overfitting* menggunakan pengujian *5-Fold Cross Validation*.

**Tujuan bisnis:**
* Mengurangi beban kerja dan memangkas waktu kerja jurnalis dalam memverifikasi berita.
* Membantu platform media sosial atau pemerintah memitigasi penyebaran disinformasi lebih awal.
* Menyediakan alat bantu bagi masyarakat awam untuk melakukan validasi mandiri guna meningkatkan literasi digital.

## 3. Metrik Kesuksesan Proyek

| Metrik | Target | Alasan |
| :--- | :--- | :--- |
| Accuracy | ≥ 95% | Akurasi keseluruhan dalam mengklasifikasikan kelas Hoaks dan Valid. |
| F1-Score (Macro) | ≥ 0.95 | Memastikan keseimbangan performa *precision* dan *recall* agar model tidak bias pada satu kelas. |
| Cross-Validation Score | ≥ 95% | Bukti ilmiah bahwa model bersifat *robust* (andal) saat diuji pada berbagai partisi data. |
| Waktu Inferensi | < 2 detik | Mendukung *deployment* aplikasi *web* yang interaktif dan responsif bagi pengguna. |

## 4. Sumber Dataset

**Indonesian Fact and Hoax Political News — Kaggle**
* **Deskripsi:** Kompilasi dataset teks berita politik yang memuat artikel berita asli dari media ternama dan arsip berita palsu hasil penelusuran.
* **Total Data:** Lebih dari 31.000 baris teks berita.
* **Sumber Portal Berita Valid:** CNN Indonesia (10.000 artikel), Tempo (6.000 artikel), Kompas (4.000 artikel).
* **Sumber Portal Berita Hoaks:** TurnBackHoax (10.000 artikel).
* **Kelas Label:** Hoaks (0) dan Valid (1).

## 5. Statistik Deskriptif Awal

| Aspek | Nilai |
| :--- | :--- |
| Total Baris Data | ~31.000 |
| Distribusi Kelas (Valid) | ~20.000 artikel |
| Distribusi Kelas (Hoaks) | ~11.000 artikel |
| Ekstraksi Fitur (*Vocab*) | 5.000 dimensi (*TF-IDF*) |
| Pendekatan Pemodelan | Klasifikasi Teks Linier |