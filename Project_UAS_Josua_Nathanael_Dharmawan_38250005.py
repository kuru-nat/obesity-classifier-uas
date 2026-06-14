"""
Project UAS - AIB02
Judul: Obesity / Health Category Classifier menggunakan ANN (MLP)
Dataset: Estimation of Obesity Levels Based on Eating Habits and Physical Condition
Sumber: UCI Machine Learning Repository / Kaggle
Link: https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels
"""

# ============================================================
# 1. IMPORT LIBRARY
# ============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("=" * 60)
print("OBESITY HEALTH CATEGORY CLASSIFIER - ANN/MLP")
print("=" * 60)

# ============================================================
# 2. LOAD DATASET
# ============================================================
# Pastikan file CSV ada di folder yang sama dengan script ini
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

print(f"\n[DATASET] Total data: {len(df)} baris, {df.shape[1]} kolom")
print(f"[DATASET] Distribusi label:\n{df['NObeyesdad'].value_counts()}\n")

# ============================================================
# 3. PREPROCESSING
# ============================================================
print("[PREPROCESSING] Memulai preprocessing...")

# Pisahkan fitur dan target
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# Encode kolom kategorikal
cat_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

print(f"[PREPROCESSING] Kolom kategorikal diencode: {cat_cols}")
print(f"[PREPROCESSING] Kelas target: {list(le_target.classes_)}")

# ============================================================
# 4. SPLIT DATA TRAIN / TEST
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\n[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")

# ============================================================
# 5. SCALING
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ============================================================
# 6. PEMBUATAN & TRAINING MODEL (ANN/MLP)
# ============================================================
print("\n[MODEL] Training MLPClassifier (ANN)...")

model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),   # 3 hidden layers
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    verbose=False
)

model.fit(X_train_scaled, y_train)
print("[MODEL] Training selesai!")
print(f"[MODEL] Iterasi: {model.n_iter_}")

# ============================================================
# 7. EVALUASI
# ============================================================
y_pred = model.predict(X_test_scaled)
acc    = accuracy_score(y_test, y_pred)

print(f"\n[EVALUASI] Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\n[EVALUASI] Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# Contoh prediksi
print("\n[CONTOH PREDIKSI] 5 sampel pertama dari test set:")
for i in range(5):
    actual    = le_target.classes_[y_test[i]]
    predicted = le_target.classes_[y_pred[i]]
    status    = "✓" if actual == predicted else "✗"
    print(f"  {status} Actual: {actual:25s} | Predicted: {predicted}")

# ============================================================
# 8. SIMPAN MODEL & ARTEFAK
# ============================================================
os.makedirs("model", exist_ok=True)

joblib.dump(model,         "model/mlp_obesity.pkl")
joblib.dump(scaler,        "model/scaler.pkl")
joblib.dump(le_target,     "model/label_encoder_target.pkl")
joblib.dump(label_encoders,"model/label_encoders_features.pkl")
joblib.dump(list(X.columns),"model/feature_cols.pkl")

print("\n[SIMPAN] Model dan artefak berhasil disimpan di folder model/")
print("  - model/mlp_obesity.pkl")
print("  - model/scaler.pkl")
print("  - model/label_encoder_target.pkl")
print("  - model/label_encoders_features.pkl")
print("  - model/feature_cols.pkl")
print("\n[SELESAI] Project UAS berhasil dijalankan!")
