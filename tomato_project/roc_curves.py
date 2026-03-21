"""
ROC Curves - Cascade Classification
-------------------------------------
Her cascade aşaması için OvR (One-vs-Rest) ROC eğrisi çizer.
Her aşamada en iyi model (MCC'ye göre) kullanılır.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, confusion_matrix
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ─── Sabitler ────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2

SINIF_ISIMLERI = {
    0: "Healthy",
    1: "Bacterial Spot",
    2: "Early Blight",
    3: "Late Blight",
    4: "Leaf Mold",
    5: "Septoria Leaf Spot",
    6: "Spider Mites",
    7: "Target Spot",
    8: "Mosaic Virus",
    9: "Yellow Leaf Curl Virus"
}

# ─── Model tanımları ──────────────────────────────────────────────────────────
def modelleri_tanimla():
    return {
        "Ridge":         RidgeClassifier(random_state=RANDOM_STATE),
        "KNN":           KNeighborsClassifier(n_neighbors=5),
        "LDA":           LinearDiscriminantAnalysis(solver="svd"),
        "QDA":           QuadraticDiscriminantAnalysis(reg_param=0.1),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "SVM":           SVC(probability=True, random_state=RANDOM_STATE)
    }

# ─── En iyi modeli seç (MCC'ye göre) ─────────────────────────────────────────
def en_iyi_modeli_sec(X_train, X_test, y_train, y_test):
    modeller = modelleri_tanimla()
    en_iyi_mcc   = -999
    en_iyi_model = None
    en_iyi_isim  = ""

    for isim, model in modeller.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mcc = matthews_corrcoef(y_test, y_pred)
        if mcc > en_iyi_mcc:
            en_iyi_mcc   = mcc
            en_iyi_model = model
            en_iyi_isim  = isim

    print(f"  En iyi model: {en_iyi_isim} | MCC: {en_iyi_mcc:.4f}")
    return en_iyi_model, en_iyi_isim

# ─── ROC skoru hesapla (predict_proba veya decision_function) ────────────────
def roc_skoru_hesapla(model, X_test):
    if hasattr(model, "predict_proba"):
        # Pozitif sınıf (1) olasılığı
        return model.predict_proba(X_test)[:, 1]
    else:
        # RidgeClassifier gibi modeller için decision_function
        return model.decision_function(X_test)

# ─── Ana fonksiyon ────────────────────────────────────────────────────────────
def main():
    # 1. Veriyi yükle
    print("Veri yukleniyor...")
    data = pd.read_csv("tomato_all_features.csv")

    X = data.drop("Target", axis=1)
    y = data["Target"]

    # 2. StandardScaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_original = X_scaled.copy()
    y_original = y.copy()

    unique_classes = sorted(y.unique())
    islenmis_siniflar = []

    # ROC verileri
    fpr_list  = []
    tpr_list  = []
    auc_list  = []
    isim_list = []
    model_list = []

    # 3. Her aşama için en iyi modeli bul ve ROC hesapla
    for asama_no, hedef_sinif in enumerate(unique_classes[:-1], start=1):
        sinif_adi = SINIF_ISIMLERI.get(hedef_sinif, str(hedef_sinif))
        print(f"\nAsama {asama_no}: Sinif {hedef_sinif} ({sinif_adi}) vs Others")

        # İşlenmiş sınıfları filtrele
        filtre    = ~y_original.isin(islenmis_siniflar)
        X_filtered = X_original[filtre].reset_index(drop=True)
        y_filtered = y_original[filtre].reset_index(drop=True)

        # İkili etiket: hedef sınıf → 1, diğerleri → 0
        y_binary = (y_filtered == hedef_sinif).astype(int)

        # Train/test ayır
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_binary,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_binary
        )

        # En iyi modeli seç
        en_iyi_model, en_iyi_isim = en_iyi_modeli_sec(X_train, X_test, y_train, y_test)

        # ROC için skor hesapla
        y_score = roc_skoru_hesapla(en_iyi_model, X_test)

        # ROC eğrisi
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc     = auc(fpr, tpr)

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(roc_auc)
        isim_list.append(f"Stage {asama_no}: {sinif_adi} ({en_iyi_isim})")
        model_list.append(en_iyi_isim)

        islenmis_siniflar.append(hedef_sinif)

    # 4. Tüm ROC eğrilerini tek grafikte çiz
    renkler = cycle([
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:cyan", "tab:olive"
    ])

    fig, ax = plt.subplots(figsize=(10, 8))

    for fpr, tpr, roc_auc, isim, renk in zip(fpr_list, tpr_list, auc_list, isim_list, renkler):
        ax.plot(fpr, tpr, label=f"{isim} (AUC = {roc_auc:.2f})", linewidth=1.8, color=renk)

    # Şans çizgisi
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance level (AUC = 0.50)")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Cascade Classification\n(One-vs-Rest, Best Model per Stage)", fontsize=13)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.tight_layout()
    plt.savefig("cascade_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nROC grafigi kaydedildi: cascade_roc_curves.png")

    # 5. AUC özetini yazdır
    print("\n" + "="*55)
    print(f"{'Stage':<35} {'Model':<15} {'AUC':>6}")
    print("="*55)
    for i, (isim, model_adi, roc_auc) in enumerate(zip(isim_list, model_list, auc_list)):
        print(f"Stage {i+1}: {SINIF_ISIMLERI[i]:<25} {model_adi:<15} {roc_auc:.4f}")
    print(f"\nMean AUC: {np.mean(auc_list):.4f}")

if __name__ == "__main__":
    main()