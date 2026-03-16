"""
Model 1 - Geniş Sınıflandırıcı (Broad Classifier)
---------------------------------------------------
Hiyerarşik sınıflandırmanın birinci aşaması.
Tüm yaprakları 5 geniş gruba ayırır:
    0: Healthy
    1: Bacterial
    2: Fungal/Oomycete
    3: Pest
    4: Viral

Ceren Dereli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (matthews_corrcoef, confusion_matrix,
                              classification_report, accuracy_score)

# cascade_classifier.py ile aynı klasörde olduğu varsayılır
from cascade_classifier import CascadeClassifier

# ── Sütun isimleri (focusm.py ile aynı) ──────────────────────────────────────
FEATURE_COLS = [
    'BrennerR', 'BrennerG', 'BrennerB',
    'DlR', 'DlG', 'DlB',
    'EogR', 'EogG', 'EogB',
    'EolR', 'EolG', 'EolB',
    'HmmR', 'HmmG', 'HmmB',
    'HeR', 'HeG', 'HeB',
    'HrR', 'HrG', 'HrB',
    'SfR', 'SfG', 'SfB',
    'TenR', 'TenG', 'TenB',
    'TvR', 'TvG', 'TvB',
    'VcR', 'VcG', 'VcB',
    'GvR', 'GvG', 'GvB',
    'GlvR', 'GlvG', 'GlvB',
    'NgvR', 'NgvG', 'NgvB',
    'TgR', 'TgG', 'TgB',
    'SgR', 'SgG', 'SgB',
    'MlR', 'MlG', 'MlB',
    'VlR', 'VlG', 'VlB',
    'StR', 'StG', 'StB',
]

CLASS_NAMES = {
    0: 'Healthy',
    1: 'Bacterial',
    2: 'Fungal/Oomycete',
    3: 'Pest',
    4: 'Viral'
}

# ── Veriyi yükle ──────────────────────────────────────────────────────────────
print("Veri yükleniyor...")
data = pd.read_csv("/Users/cerendereli/Desktop/tomato-leaf-disease-detection/tomato_project/tomato_all_features.csv")

X = data[FEATURE_COLS]
y = data['broad_target']  # Model 1 hedefi

print(f"Veri kümesi şekli: {data.shape}")
print(f"Sınıf dağılımı:\n{y.value_counts().sort_index()}\n")

# ── 1000 tekrar ───────────────────────────────────────────────────────────────
mccs = []
conf_matrices = []
accuracies = []

for _ in tqdm(range(1000), desc="Model 1 eğitiliyor"):
    # Veriyi karıştır
    data_shuffled = data.sample(frac=1, random_state=None)
    X_shuffled = data_shuffled[FEATURE_COLS]
    y_shuffled = data_shuffled['broad_target']

    # Train/test split (%80/%20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_shuffled, y_shuffled, test_size=0.20, random_state=42, stratify=y_shuffled
    )

    # CascadeClassifier eğit
    cascade = CascadeClassifier(random_state=42)
    cascade.fit(X_train, y_train, verbose=False)

    # Tahmin
    y_pred = cascade.predict(X_test)

    # Metrikler
    mcc = matthews_corrcoef(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)

    mccs.append(mcc)
    accuracies.append(acc)
    conf_matrices.append(cm)

# ── Sonuçlar ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("MODEL 1 — GENIŞ SINIFLANDIRICI SONUÇLARI")
print(f"{'='*50}")
print(f"Ortalama MCC      : {np.mean(mccs):.4f} ± {np.std(mccs):.4f}")
print(f"Ortalama Accuracy : {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")

# Son iterasyon detaylı rapor
data_shuffled = data.sample(frac=1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    data_shuffled[FEATURE_COLS], data_shuffled['broad_target'],
    test_size=0.20, random_state=42, stratify=data_shuffled['broad_target']
)
cascade_final = CascadeClassifier(random_state=42)
cascade_final.fit(X_train, y_train, verbose=False)
y_pred_final = cascade_final.predict(X_test)

print("\nSon İterasyon Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_final,
      target_names=list(CLASS_NAMES.values())))

# ── Görselleştirme ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MCC dağılımı
axes[0].hist(mccs, bins=30, edgecolor='k', color='steelblue')
axes[0].set_title('Model 1 — MCC Dağılımı (1000 tekrar)')
axes[0].set_xlabel('MCC Değeri')
axes[0].set_ylabel('Frekans')

# Ortalama confusion matrix
mean_cm = np.mean(conf_matrices, axis=0).astype(int)
sns.heatmap(mean_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=list(CLASS_NAMES.values()),
            yticklabels=list(CLASS_NAMES.values()))
axes[1].set_title('Model 1 — Ortalama Confusion Matrix')
axes[1].set_xlabel('Tahmin Edilen')
axes[1].set_ylabel('Gerçek')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('model1_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Grafik kaydedildi: model1_results.png")
