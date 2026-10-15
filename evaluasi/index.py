import pandas as pd
import joblib
from scipy import sparse
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ==========================================================
# [LANGKAH 5] Pengujian & Evaluasi (Evaluation)
#   - Predict terhadap X_test_tfidf
#   - Hitung Akurasi, Presisi, Recall, F1-Score
#   - Cetak Confusion Matrix
# ==========================================================

# Baca data uji hasil langkah 2 & 3, dan label asli hasil langkah 1
X_test_tfidf = sparse.load_npz('./TF-IDF/X_test_tfidf.npz')
y_test = pd.read_csv('./test_split/y_test.csv')['sentimen']

# Baca model hasil langkah 4
model_nb = joblib.load('./pelatihan_model/model_naive_bayes.joblib')
model_svm = joblib.load('./pelatihan_model/model_svm.joblib')


def evaluasi_model(nama_model, model, X_test, y_test):
    y_pred = model.predict(X_test)

    akurasi = accuracy_score(y_test, y_pred)
    presisi = precision_score(y_test, y_pred, pos_label='positif')
    recall = recall_score(y_test, y_pred, pos_label='positif')
    f1 = f1_score(y_test, y_pred, pos_label='positif')
    cm = confusion_matrix(y_test, y_pred, labels=['positif', 'negatif'])

    print(f"\n{'='*50}")
    print(f"Evaluasi Model: {nama_model}")
    print(f"{'='*50}")
    print(f"Akurasi  : {akurasi:.4f}")
    print(f"Presisi  : {presisi:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print()
    print("Confusion Matrix (baris = aktual, kolom = prediksi):")
    print("             Prediksi Positif   Prediksi Negatif")
    print(f"Aktual Positif    {cm[0][0]:>10}         {cm[0][1]:>10}")
    print(f"Aktual Negatif    {cm[1][0]:>10}         {cm[1][1]:>10}")
    print()
    print("Classification Report Lengkap:")
    print(classification_report(y_test, y_pred))

    return {
        'model': nama_model,
        'akurasi': akurasi,
        'presisi': presisi,
        'recall': recall,
        'f1_score': f1
    }


# ==========================================================
# Evaluasi kedua model
# ==========================================================
hasil_nb = evaluasi_model("Multinomial Naive Bayes", model_nb, X_test_tfidf, y_test)
hasil_svm = evaluasi_model("Support Vector Machine (LinearSVC)", model_svm, X_test_tfidf, y_test)

# ==========================================================
# Ringkasan perbandingan kedua model (untuk disalin ke naskah)
# ==========================================================
df_ringkasan = pd.DataFrame([hasil_nb, hasil_svm])
print(f"\n{'='*50}")
print("Ringkasan Perbandingan Model")
print(f"{'='*50}")
print(df_ringkasan.to_string(index=False))

df_ringkasan.to_csv('hasil_evaluasi_model.csv', index=False)
print("\nRingkasan disimpan ke 'hasil_evaluasi_model.csv'")