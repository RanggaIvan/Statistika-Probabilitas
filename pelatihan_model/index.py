import pandas as pd
import joblib
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# ==========================================================
# [LANGKAH 4] Pelatihan Model (Model Training)
#   - Fit Multinomial Naive Bayes
#   - Fit Support Vector Machine (LinearSVC)
# ==========================================================

# Baca hasil TF-IDF dari langkah 2 & 3, serta label dari langkah 1
X_train_tfidf = sparse.load_npz('./TF-IDF/X_train_tfidf.npz')
y_train = pd.read_csv('./test_split/y_train.csv')['sentimen']

# ----------------------------------------------------------
# Model 1: Multinomial Naive Bayes
# Cocok untuk data berbasis frekuensi kata seperti TF-IDF,
# ringan dan sering jadi baseline untuk klasifikasi teks.
# ----------------------------------------------------------
model_nb = MultinomialNB()
model_nb.fit(X_train_tfidf, y_train)
print("Model Naive Bayes selesai dilatih.")

# ----------------------------------------------------------
# Model 2: Support Vector Machine (LinearSVC)
# Umumnya memberi performa lebih baik untuk data teks
# berdimensi tinggi (banyak fitur) seperti hasil TF-IDF.
# ----------------------------------------------------------
model_svm = LinearSVC(random_state=42)
model_svm.fit(X_train_tfidf, y_train)
print("Model SVM (LinearSVC) selesai dilatih.")

# ==========================================================
# Simpan kedua model untuk dipakai di langkah 5 (evaluasi)
# ==========================================================
joblib.dump(model_nb, 'model_naive_bayes.joblib')
joblib.dump(model_svm, 'model_svm.joblib')

print("\nModel disimpan:")
print("- model_naive_bayes.joblib")
print("- model_svm.joblib")