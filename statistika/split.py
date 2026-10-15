import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================================
# [LANGKAH 1] Train-Test Split (80% Latih : 20% Uji)
# ==========================================================

# Baca data bersih (554 baris, sudah seimbang positif/negatif)
df = pd.read_csv('./clean data/tokopedia_reviews_olahraga_final.csv')
df['review_text_preprocessed'] = df['review_text_preprocessed'].fillna('')

X_teks = df['review_text_preprocessed']
y = df['sentimen']

# Split 80% latih : 20% uji
# stratify=y -> menjaga proporsi 50% positif & 50% negatif tetap
#               sama persis di data latih maupun data uji
X_train_teks, X_test_teks, y_train, y_test = train_test_split(
    X_teks, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("=== Hasil Train-Test Split ===")
print(f"Data Latih (X_train) : {len(X_train_teks)} baris")
print(f"Data Uji   (X_test)  : {len(X_test_teks)} baris")
print()
print("Distribusi sentimen - Data Latih:")
print(y_train.value_counts())
print()
print("Distribusi sentimen - Data Uji:")
print(y_test.value_counts())

# ==========================================================
# Simpan hasil split (teks mentah + label) untuk dipakai
# oleh langkah 2 & 3 (TF-IDF)
# ==========================================================
X_train_teks.to_csv('X_train_teks.csv', index=False, header=['review_text_preprocessed'])
X_test_teks.to_csv('X_test_teks.csv', index=False, header=['review_text_preprocessed'])
y_train.to_csv('y_train.csv', index=False, header=['sentimen'])
y_test.to_csv('y_test.csv', index=False, header=['sentimen'])

print("\nFile hasil split disimpan:")
print("- /X_train_teks.csv")
print("- X_test_teks.csv")
print("- y_train.csv")
print("- y_test.csv")