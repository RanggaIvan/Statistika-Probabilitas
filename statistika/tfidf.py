import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

# ==========================================================
# [LANGKAH 2] TF-IDF fit_transform pada Data Latih
# [LANGKAH 3] TF-IDF transform pada Data Uji
# ==========================================================

# Baca hasil split dari langkah 1
X_train_teks = pd.read_csv('./test_split/X_train_teks.csv')['review_text_preprocessed'].fillna('')
X_test_teks = pd.read_csv('./test_split/X_test_teks.csv')['review_text_preprocessed'].fillna('')

# Inisialisasi TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.9,
    ngram_range=(1, 2)   # unigram + bigram
)

# [LANGKAH 2] fit_transform HANYA pada data latih
# -> vocabulary & bobot IDF dipelajari murni dari data latih
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_teks)

# [LANGKAH 3] transform pada data uji
# -> TIDAK fit ulang, hanya memakai vocabulary dari data latih.
#    Kata baru di data uji yang tidak ada di vocabulary latih
#    otomatis diabaikan (bukan error).
X_test_tfidf = tfidf_vectorizer.transform(X_test_teks)

print("=== Info Matriks TF-IDF ===")
print(f"Jumlah fitur (kata/frasa unik dari data latih): {X_train_tfidf.shape[1]}")
print(f"Ukuran matriks data latih : {X_train_tfidf.shape}")
print(f"Ukuran matriks data uji   : {X_test_tfidf.shape}")

# ==========================================================
# Simpan hasil untuk dipakai di langkah 4 (training model)
# ==========================================================
sparse.save_npz('X_train_tfidf.npz', X_train_tfidf)
sparse.save_npz('X_test_tfidf.npz', X_test_tfidf)

fitur_kata = tfidf_vectorizer.get_feature_names_out()
pd.Series(fitur_kata).to_csv('tfidf_feature_names.csv', index=False, header=['kata'])

print("\nFile hasil TF-IDF disimpan:")
print("- X_train_tfidf.npz        (fitur data latih)")
print("- X_test_tfidf.npz         (fitur data uji)")
print("- tfidf_feature_names.csv  (daftar kata/frasa fitur, dari data latih)")