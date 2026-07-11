import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

def bersihkan_teks(teks):
    teks = str(teks)
    teks = teks.lower()
    teks = re.sub(r'[^a-z\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

def load_and_clean_data():
    print("--- Memulai Proses Pembacaan & Pembersihan Data ---")
    
    # 1. Membaca dataset
    df_cnn = pd.read_excel('../data/dataset_cnn_10k.xlsx')
    df_kompas = pd.read_excel('../data/dataset_kompas_4k.xlsx')
    df_tempo = pd.read_excel('../data/dataset_tempo_6k.xlsx')
    df_hoax = pd.read_excel('../data/dataset_turnbackhoax_10k.xlsx')

    # 2. Memberikan label
    df_cnn['Label'] = 'Valid'
    df_kompas['Label'] = 'Valid'
    df_tempo['Label'] = 'Valid'
    df_hoax['Label'] = 'Hoaks'

    # 3. Menggabungkan data
    df_all = pd.concat([df_cnn, df_kompas, df_tempo, df_hoax], ignore_index=True)
    
    # 4. Merapikan kolom
    df = df_all[['FullText', 'Label']].copy()
    df.rename(columns={'FullText': 'Teks Berita'}, inplace=True)
    df.dropna(subset=['Teks Berita'], inplace=True)
    
    # 5. Menerapkan pembersihan teks
    print("Sedang membersihkan teks (harap tunggu)...")
    df['Teks Bersih'] = df['Teks Berita'].apply(bersihkan_teks)
    
    print(f"Selesai! Total data bersih: {len(df)} baris")
    return df