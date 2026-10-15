import pandas as pd

# Baca file CSV
df = pd.read_csv('tokopedia_product_reviews_2025.csv')

# Filter data dengan kategori "Olahraga"
df_olahraga = df[df['product_category'] == 'Olahraga']

# Tampilkan info singkat
print(f"Jumlah baris kategori Olahraga: {len(df_olahraga)}")
print(df_olahraga.head())

# Simpan hasil filter ke file CSV baru
df_olahraga.to_csv('tokopedia_reviews_olahraga.csv', index=False)