import pandas as pd

# ==========================================================
# 0. Baca data hasil filter kategori "Olahraga"
# ==========================================================
df = pd.read_csv('tokopedia_reviews_olahraga.csv')

# ==========================================================
# 1. Cek Sebaran Rating Bintang (1 s.d. 5)
# ==========================================================
sebaran_rating = df['rating'].value_counts().sort_index()
print("=== Sebaran Rating Bintang ===")
print(sebaran_rating)
print()

# ==========================================================
# 2. Buat Kolom Label Sentimen (Biner)
#    Rating 1 & 2 -> negatif
#    Rating 4 & 5 -> positif
#    Rating 3     -> dieliminasi (netral/ambigu)
# ==========================================================
def label_sentimen(rating):
    if rating in [1, 2]:
        return 'negatif'
    elif rating in [4, 5]:
        return 'positif'
    else:
        return None  # rating 3, akan dibuang

df['sentimen'] = df['rating'].apply(label_sentimen)

# Buang baris dengan rating 3 (sentimen None)
df_biner = df.dropna(subset=['sentimen']).copy()

print("=== Jumlah Data Setelah Rating 3 Dieliminasi ===")
print(df_biner['sentimen'].value_counts())
print()

# ==========================================================
# 3. Penyeimbangan Data (Undersampling)
#    Samakan jumlah baris positif dengan jumlah baris negatif
# ==========================================================
jumlah_negatif = df_biner[df_biner['sentimen'] == 'negatif'].shape[0]
jumlah_positif = df_biner[df_biner['sentimen'] == 'positif'].shape[0]

n_sample = min(jumlah_negatif, jumlah_positif)

df_negatif = df_biner[df_biner['sentimen'] == 'negatif'].sample(n=n_sample, random_state=42)
df_positif = df_biner[df_biner['sentimen'] == 'positif'].sample(n=n_sample, random_state=42)

# Gabungkan dan acak urutannya
df_balanced = pd.concat([df_negatif, df_positif]).sample(frac=1, random_state=42).reset_index(drop=True)

print("=== Jumlah Data Setelah Undersampling (Balanced) ===")
print(df_balanced['sentimen'].value_counts())
print()

# ==========================================================
# Simpan hasil akhir
# ==========================================================
df_balanced.to_csv('tokopedia_reviews_olahraga_balanced.csv', index=False)
print("Data balanced berhasil disimpan ke 'tokopedia_reviews_olahraga_balanced.csv'")

