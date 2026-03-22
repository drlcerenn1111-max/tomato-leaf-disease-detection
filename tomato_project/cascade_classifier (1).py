"""
Cascade Classification - Tomato Leaf Disease Detection
-------------------------------------------------------
Makale: Classification of Tomato Leaf Diseases Through Gradual Use of Machine Learning Algorithms
Yazar: Ceren DERELİ

Yöntem:
- 10 sınıf, 9 aşamalı One-vs-Rest (OvR) cascade
- Her aşamada 7 model denenir: Ridge, KNN, LDA, QDA, DT, RF, SVM
- En iyi model MCC'ye göre seçilir
- Metrikler: Accuracy, Sensitivity, Specificity, F1, MCC
- 10-fold stratified cross-validation
- En iyi modeller pickle ile kaydedilir
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, matthews_corrcoef, f1_score,
                             confusion_matrix, classification_report)
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

RANDOM_STATE = 42
TEST_SIZE    = 0.2
N_FOLDS      = 10

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

def ikili_veri_hazirla(X, y, hedef_sinif):
    y_binary = (y == hedef_sinif).astype(int)
    return X.copy(), y_binary

def modelleri_degerlendir(X_train, X_test, y_train, y_test, asama_no, hedef_sinif):
    modeller = modelleri_tanimla()

    print(f"\n  {'Model':<16} {'CV Acc':>8} {'CV MCC':>8} {'Test Acc':>10} {'Sens':>8} {'Spec':>8} {'F1':>8} {'MCC':>8}")
    print(f"  {'-'*78}")

    sonuclar = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for isim, model in modeller.items():
        cv = cross_validate(
            model, X_train, y_train, cv=skf,
            scoring=["accuracy", "matthews_corrcoef"],
            n_jobs=-1
        )
        cv_acc = cv["test_accuracy"].mean()
        cv_mcc = cv["test_matthews_corrcoef"].mean()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        mcc  = matthews_corrcoef(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"  {isim:<16} {cv_acc:>8.4f} {cv_mcc:>8.4f} {acc:>10.4f} {sens:>8.4f} {spec:>8.4f} {f1:>8.4f} {mcc:>8.4f}")

        sonuclar.append({
            "Model":        isim,
            "CV_Accuracy":  round(cv_acc, 4),
            "CV_MCC":       round(cv_mcc, 4),
            "Accuracy":     round(acc, 4),
            "Sensitivity":  round(sens, 4),
            "Specificity":  round(spec, 4),
            "F1":           round(f1, 4),
            "MCC":          round(mcc, 4),
            "Model_Object": model,
            "y_pred":       y_pred
        })

    df = pd.DataFrame(sonuclar).sort_values("MCC", ascending=False).reset_index(drop=True)
    en_iyi = df.iloc[0]

    print(f"\n  En iyi model: {en_iyi['Model']} | MCC: {en_iyi['MCC']:.4f} | F1: {en_iyi['F1']:.4f} | Accuracy: {en_iyi['Accuracy']:.4f}")
    return df, en_iyi["Model_Object"], en_iyi["Model"], en_iyi["y_pred"]

def karisiklik_matrisi_goster(y_test, y_pred, model_adi, hedef_sinif, asama_no):
    sinif_adi = SINIF_ISIMLERI.get(hedef_sinif, str(hedef_sinif))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Others", sinif_adi],
                yticklabels=["Others", sinif_adi])
    plt.title(f"Stage {asama_no}: {model_adi}\n(Class {hedef_sinif}: {sinif_adi} vs Others)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_asama{asama_no}_sinif{hedef_sinif}.png", dpi=150)
    plt.show()

def toplu_karisiklik_matrisi_goster(all_cms, all_model_names, all_classes, all_asama_nos):
    n    = len(all_cms)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for i in range(n):
        ax        = axes[i]
        sinif     = all_classes[i]
        sinif_adi = SINIF_ISIMLERI.get(sinif, str(sinif))
        cm        = all_cms[i]

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Others", f"S{sinif}"],
                    yticklabels=["Others", f"S{sinif}"])
        ax.set_title(f"Stage {all_asama_nos[i]}: {all_model_names[i]}\n(Class {sinif} vs Others)", fontsize=9)
        ax.set_ylabel("Actual", fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Cascade Classification - All Stages Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("cascade_tum_asamalar_confusion.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nAll stages confusion matrix saved: cascade_tum_asamalar_confusion.png")

def ozet_tablo_goster(best_results):
    df = pd.DataFrame(best_results)
    print("\n" + "="*105)
    print("CASCADE SINIFLANDIRICI - TUM ASAMALAR PERFORMANS OZETI")
    print("="*105)
    goster = df[["Asama", "Sinif", "En Iyi Model", "CV_MCC", "Accuracy", "Sensitivity", "Specificity", "F1", "MCC"]]
    print(goster.to_string(index=False))
    df.drop(columns=["En Iyi Model Nesnesi"], errors="ignore").to_csv(
        "cascade_performans_ozeti.csv", index=False)
    print("\nOzet CSV kaydedildi: cascade_performans_ozeti.csv")

def main():
    print("Veri yukleniyor...")
    data = pd.read_csv("tomato_all_features.csv")

    X = data.drop("Target", axis=1)
    y = data["Target"]

    print(f"Veri kumesi boyutu: {data.shape}")
    print(f"Sinif dagilimi:\n{y.value_counts().sort_index()}")

    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_original = X_scaled.copy()
    y_original = y.copy()

    unique_classes = sorted(y.unique())
    print(f"\nSiniflandirilacak siniflar: {unique_classes}")

    islenmis_siniflar = []
    best_models       = {}
    best_results      = []
    all_cms           = []
    all_model_names   = []
    all_classes       = []
    all_asama_nos     = []

    for asama_no, hedef_sinif in enumerate(unique_classes[:-1], start=1):
        sinif_adi = SINIF_ISIMLERI.get(hedef_sinif, str(hedef_sinif))

        print(f"\n{'='*70}")
        print(f"ASAMA {asama_no}: Sinif {hedef_sinif} ({sinif_adi}) vs Digerleri")
        print(f"{'='*70}")

        filtre     = ~y_original.isin(islenmis_siniflar)
        X_filtered = X_original[filtre].reset_index(drop=True)
        y_filtered = y_original[filtre].reset_index(drop=True)

        print(f"  Kalan veri boyutu: {X_filtered.shape}")

        X_binary, y_binary = ikili_veri_hazirla(X_filtered, y_filtered, hedef_sinif)

        X_train, X_test, y_train, y_test = train_test_split(
            X_binary, y_binary,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_binary
        )

        df_sonuc, en_iyi_model, en_iyi_isim, y_pred = modelleri_degerlendir(
            X_train, X_test, y_train, y_test, asama_no, hedef_sinif
        )

        karisiklik_matrisi_goster(y_test, y_pred, en_iyi_isim, hedef_sinif, asama_no)

        all_cms.append(confusion_matrix(y_test, y_pred))
        all_model_names.append(en_iyi_isim)
        all_classes.append(hedef_sinif)
        all_asama_nos.append(asama_no)

        best_models[hedef_sinif] = en_iyi_model
        ozet = df_sonuc.iloc[0]
        best_results.append({
            "Asama":                asama_no,
            "Sinif":                f"{hedef_sinif} ({sinif_adi})",
            "En Iyi Model":         en_iyi_isim,
            "CV_MCC":               ozet["CV_MCC"],
            "Accuracy":             ozet["Accuracy"],
            "Sensitivity":          ozet["Sensitivity"],
            "Specificity":          ozet["Specificity"],
            "F1":                   ozet["F1"],
            "MCC":                  ozet["MCC"],
            "En Iyi Model Nesnesi": en_iyi_model
        })

        islenmis_siniflar.append(hedef_sinif)

    son_sinif     = unique_classes[-1]
    son_sinif_adi = SINIF_ISIMLERI.get(son_sinif, str(son_sinif))
    print(f"\n{'='*70}")
    print(f"Son sinif (direkt atanan): Sinif {son_sinif} ({son_sinif_adi})")
    print(f"{'='*70}")

    toplu_karisiklik_matrisi_goster(all_cms, all_model_names, all_classes, all_asama_nos)
    ozet_tablo_goster(best_results)

    kayit = {
        "models":         best_models,
        "scaler":         scaler,
        "feature_names":  list(X.columns),
        "class_order":    unique_classes,
        "sinif_isimleri": SINIF_ISIMLERI
    }
    with open("cascade_models.pkl", "wb") as f:
        pickle.dump(kayit, f)
    print(f"\nModeller kaydedildi: cascade_models.pkl")
    print(f"Kaydedilen asama sayisi: {len(best_models)}")

if __name__ == "__main__":
    main()
