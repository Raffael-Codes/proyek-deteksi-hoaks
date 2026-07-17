import re

def bersihkan_teks(teks):
    """
    Fungsi utilitas untuk membersihkan teks berita.
    Mengubah teks menjadi huruf kecil dan menghapus karakter selain huruf.
    """
    teks = str(teks).lower()
    teks = re.sub(r'[^a-z\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks