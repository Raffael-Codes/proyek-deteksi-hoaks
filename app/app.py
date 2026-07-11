import streamlit as st
import joblib
from PIL import Image
import pytesseract

# 1. KONFIGURASI TESSERACT (Wajib di Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 2. LOAD MODEL KECERDASAN BUATAN ANDA (Jalur sudah diperbarui!)
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
model = joblib.load('models/logreg_model.pkl')

st.title("Deteksi Hoaks")
st.write("Silakan ketik teks berita atau upload screenshot berita.")

# 3. MEMBUAT DUA PILIHAN INPUT (Tanpa Kamera)
tab1, tab2 = st.tabs(["Ketik Teks", "Upload Gambar"])

teks_input = "" # Variabel penampung akhir

with tab1:
    input_manual = st.text_area("Masukkan teks berita di sini:")
    if input_manual:
        teks_input = input_manual

with tab2:
    gambar_upload = st.file_uploader("Upload screenshot berita (PNG/JPG)", type=['png', 'jpg', 'jpeg'])
    if gambar_upload is not None:
        gambar = Image.open(gambar_upload)
        st.image(gambar, caption="Gambar Berhasil Dimuat", use_container_width=True)
        st.info("Sedang mengekstrak teks dari gambar...")
        
        # Mengekstrak teks dari gambar
        teks_input = pytesseract.image_to_string(gambar)
        st.write("**Hasil Ekstraksi Teks:**", teks_input)

# 4. LOGIKA PREDIKSI UTAMA
st.divider()
if st.button("⚡ Periksa Keaslian"):
    if teks_input.strip() != "":
        # Proses teks menggunakan TF-IDF
        teks_vektor = vectorizer.transform([teks_input])
        
        # Lakukan Prediksi
        prediksi = model.predict(teks_vektor)[0]
        
        # Tampilkan Hasil
        if prediksi == 'Hoaks':
            st.error("🚨 PERINGATAN: Berita ini terindikasi HOAKS!")
        else:
            st.success("✅ AMAN: Berita ini terindikasi VALID.")
    else:
        st.warning("Sistem tidak mendeteksi adanya teks. Silakan berikan input terlebih dahulu.")