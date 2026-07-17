import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sistem Deteksi Hoaks", page_icon="📰", layout="wide")

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vec_path = os.path.join(base_dir, 'models', 'tfidf_vectorizer.pkl')
    model_path = os.path.join(base_dir, 'models', 'logreg_model.pkl')
    
    vectorizer = joblib.load(vec_path)
    model = joblib.load(model_path)
    return vectorizer, model

try:
    vectorizer, model = load_models()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Gagal memuat model. Pastikan file .pkl ada di folder models/. Error: {e}")

# --- SIDEBAR NAVIGASI ---
st.sidebar.title("🧭 Navigasi Menu")
st.sidebar.markdown("Silakan pilih menu di bawah ini sesuai kriteria **Soal 4**:")
menu = st.sidebar.radio(
    "",
    ["1. Dokumentasi", "2. Dashboard EDA", "3. Model Demo", "4. Evaluasi Model", "5. Interpretasi Hasil"]
)

st.sidebar.markdown("---")
st.sidebar.info("Dibuat untuk Ujian Akhir Semester - Pembelajaran Mesin")

# ==========================================
# MENU 1: DOKUMENTASI
# ==========================================
if menu == "1. Dokumentasi":
    st.title("📚 Dokumentasi Aplikasi")
    st.markdown("""
    Selamat datang di Aplikasi Deteksi Hoaks Politik Berbahasa Indonesia. Aplikasi ini merupakan implementasi nyata dari pemodelan *Machine Learning* untuk memverifikasi kredibilitas berita.

    ### 📌 Tentang Dataset
    *   **Sumber Data:** Dataset *Indonesian Fact and Hoax Political News* (diambil dari Kaggle).
    *   **Kompilasi:** Terdiri dari berita valid (CNN, Kompas, Tempo) dan arsip berita palsu dari TurnBackHoax.
    *   **Volume:** Lebih dari 31.000 baris data sebelum pra-pemrosesan.

    ### ⚙️ Metodologi Pembuatan
    1.  **Pra-pemrosesan (Preprocessing):** Meliputi penghapusan *missing values*, *case folding*, dan pembersihan teks.
    2.  **Ekstraksi Fitur:** Menggunakan **TF-IDF Vectorizer** (5.000 fitur).
    3.  **Pemodelan Utama:** Menggunakan **Logistic Regression**.
    """)

# ==========================================
# MENU 2: DASHBOARD EDA
# ==========================================
elif menu == "2. Dashboard EDA":
    st.title("📊 Dashboard EDA (Exploratory Data Analysis)")
    st.write("Visualisasi interaktif distribusi data yang digunakan untuk melatih model.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Label Dataset Asli")
        labels = 'Berita Valid', 'Berita Hoaks'
        sizes = [20000, 11000] 
        colors = ['#2ca02c', '#d62728']
        explode = (0.1, 0)  

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.axis('equal')  
        st.pyplot(fig1)
        
    with col2:
        st.subheader("Sumber Pengambilan Data")
        sumber = ['TurnBackHoax', 'CNN Indonesia', 'Tempo', 'Kompas']
        jumlah = [10000, 10000, 6000, 4000]
        
        fig2, ax2 = plt.subplots()
        ax2.barh(sumber, jumlah, color=['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_xlabel('Jumlah Artikel')
        ax2.set_title('Volume Data per Portal Berita')
        st.pyplot(fig2)

# ==========================================
# MENU 3: MODEL DEMO (DENGAN UPLOAD FOTO)
# ==========================================
elif menu == "3. Model Demo":
    st.title("🤖 Model Demo: Pendeteksi Hoaks")
    st.markdown("Silakan pilih untuk memasukkan teks berita secara manual atau mengunggah *screenshot* gambar berita politik.")
    
    tab1, tab2 = st.tabs(["📝 Input Teks Manual", "📸 Unggah Gambar Berita"])
    
    user_input = ""
    
    with tab1:
        teks_manual = st.text_area("Masukkan teks berita di sini:", height=200, placeholder="Ketik atau paste teks berita panjang di sini...")
        if teks_manual:
            user_input = teks_manual
            
    with tab2:
        uploaded_file = st.file_uploader("Unggah gambar/screenshot berita (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diunggah", use_column_width=True)
            with st.spinner("Sedang mengekstrak teks dari gambar..."):
                try:
                    # Menggunakan parameter bahasa indonesia jika tersedia
                    teks_ekstrak = pytesseract.image_to_string(image, lang='ind')
                except:
                    # Fallback ke default bahasa inggris jika ind tidak tersedia
                    teks_ekstrak = pytesseract.image_to_string(image)
                
            st.success("Teks berhasil diekstrak dari gambar!")
            user_input = st.text_area("Hasil Ekstraksi Teks (Anda bisa mengedit teks ini jika ada kata yang terpotong):", value=teks_ekstrak, height=150)

    # Tombol Prediksi (berlaku untuk kedua tab)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Deteksi Kebenaran Teks", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Mohon masukkan teks atau unggah gambar berita terlebih dahulu!")
        elif model_loaded:
            with st.spinner('Sedang menganalisis struktur kalimat...'):
                teks_bersih = user_input.lower()
                vektor_input = vectorizer.transform([teks_bersih])
                prediksi = model.predict(vektor_input)[0]
                probabilitas = model.predict_proba(vektor_input)[0]
                
                st.markdown("---")
                st.subheader("Hasil Analisis:")
                
                if prediksi == 1 or prediksi == "1" or str(prediksi).lower() == "valid":
                    st.success("✅ **KATEGORI: BERITA VALID**")
                    st.write(f"Model sangat yakin sebesar **{probabilitas[1]*100:.2f}%** bahwa berita ini berasal dari sumber resmi yang dapat dipercaya.")
                else:
                    st.error("🚨 **KATEGORI: INDIKASI HOAKS**")
                    st.write(f"Model mencurigai sebesar **{probabilitas[0]*100:.2f}%** bahwa narasi ini mengandung unsur manipulatif atau hoaks.")

# ==========================================
# MENU 4: EVALUASI MODEL
# ==========================================
elif menu == "4. Evaluasi Model":
    st.title("📈 Evaluasi Performa Model")
    st.write("Metrik evaluasi pada *Testing Set* menggunakan algoritma **Logistic Regression**:")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Akurasi Utama", "99.35%", "Tertinggi")
    col2.metric("Precision", "0.99", "")
    col3.metric("Recall", "0.99", "")
    col4.metric("F1-Score", "0.99", "")
    
    st.markdown("---")
    st.subheader("Rincian Laporan Klasifikasi (*Classification Report*)")
    
    report_data = {
        "Kelas": ["Hoaks", "Valid", "Akurasi Global", "Rata-rata Makro"],
        "Precision": ["0.99", "0.99", "-", "0.99"],
        "Recall": ["0.99", "1.00", "-", "0.99"],
        "F1-Score": ["0.99", "1.00", "0.99", "0.99"],
        "Support": ["2076", "4190", "6266", "6266"]
    }
    st.table(pd.DataFrame(report_data))

# ==========================================
# MENU 5: INTERPRETASI HASIL
# ==========================================
elif menu == "5. Interpretasi Hasil":
    st.title("💡 Interpretasi Hasil & Insights Bisnis")
    st.markdown("""
    ### 🧠 Interpretasi Keputusan Model
    Algoritma membedakan hoaks dan berita valid berdasarkan **pola kemunculan kata (TF-IDF)**. 
    *   Berita **Valid** menggunakan struktur bahasa formal dan objektif.
    *   Berita **Hoaks** seringkali menggunakan gaya bahasa provokatif dan hiperbolis.

    ### 💼 Business & Social Insights
    1.  **Efisiensi Waktu:** Memangkas waktu *fact-checking* menjadi hitungan detik.
    2.  **Mitigasi Konflik:** Membantu memblokir penyebaran hoaks sebelum memicu kegaduhan publik.
    3.  **Literasi Digital:** Alat validasi mandiri bagi masyarakat awam.
    """)