import pandas as pd

# ==========================================================
# Baca data hasil preprocessing
# ==========================================================
df = pd.read_csv('tokopedia_reviews_olahraga_preprocessed.csv')

# ==========================================================
# Pilih kolom yang ingin dipertahankan
# ==========================================================
kolom_dipertahankan = [
    'review_text',
    'review_text_preprocessed',
    'sentimen',
    'rating',
    'product_name',
    'product_category'
]

df_final = df[kolom_dipertahankan]

# ==========================================================
# Tampilkan contoh hasil & simpan
# ==========================================================
print("=== Kolom pada Dataset Final ===")
print(df_final.columns.tolist())
print()
print(df_final.head())

df_final.to_csv('tokopedia_reviews_olahraga_final.csv', index=False)
print("\nData final disimpan ke 'tokopedia_reviews_olahraga_final.csv'")