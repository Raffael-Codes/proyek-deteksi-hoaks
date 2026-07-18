import pandas as pd
from utils import bersihkan_teks
import warnings
warnings.filterwarnings('ignore')

def run_preprocessing():
    print("--- Memulai Proses Pembacaan & Pembersihan Data ---")
    
    # Hapus ../ karena kita menjalankannya dari root folder
    df_cnn = pd.read_excel('data/raw/dataset_cnn_10k.xlsx')
    df_kompas = pd.read_excel('data/raw/dataset_kompas_4k.xlsx')
    df_tempo = pd.read_excel('data/raw/dataset_tempo_6k.xlsx')
    df_hoax = pd.read_excel('data/raw/dataset_turnbackhoax_10k.xlsx')

    df_cnn['Label'] = 'Valid'
    df_kompas['Label'] = 'Valid'
    df_tempo['Label'] = 'Valid'
    df_hoax['Label'] = 'Hoaks'

    df_all = pd.concat([df_cnn, df_kompas, df_tempo, df_hoax], ignore_index=True)
    
    df = df_all[['FullText', 'Label']].copy()
    df.rename(columns={'FullText': 'Teks Berita'}, inplace=True)
    df.dropna(subset=['Teks Berita'], inplace=True)
    
    print("Sedang membersihkan teks (harap tunggu)...")
    df['Teks Bersih'] = df['Teks Berita'].apply(bersihkan_teks)
    
    # Hapus ../ di sini juga
    df.to_csv('data/processed/dataset_bersih.csv', index=False)
    print("✅ Selesai! Data bersih berhasil disimpan ke 'data/processed/dataset_bersih.csv'")

if __name__ == "__main__":
    run_preprocessing()