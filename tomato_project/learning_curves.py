"""
Learning Curves - Cascade Classification
-----------------------------------------
Her cascade aşamasının en iyi modeli için öğrenme eğrisi çizer.
Ezberleme (overfitting) kontrolü için kullanılır.

Yazar: Ceren Dereli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
from sklearn.metrics import matthews_corrcoef
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

# ─── Learning curve çiz ───────────────────────────────────────────────────────
def learning_curve_ciz(ax, model, X, y, baslik):
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=RANDOM_STATE)
    train_sizes = np.linspace(0.1, 1.0, 5)

    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        shuffle=True
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)
    test_mean  = np.mean(test_scores, axis=1)
    test_std   = np.std(test_scores, axis=1)

    ax.grid()
    ax.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_mean - test_std,
                    test_mean + test_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_mean, "o-", color="r", label="Training score")
    ax.plot(train_sizes, test_mean,  "o-", color="g", label="Test score")
    ax.set_title(baslik, fontsize=10)
    ax.set_xlabel("Training set size (ratio)")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="best", fontsize=8)
    ax.annotate(f"Max test: {test_mean.max():.4f}",
                xy=(0.5, 0.03), xycoords="axes fraction",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
                fontsize=8, ha="center")

    return test_mean.max(), train_mean[-1] - test_mean[-1]

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

    unique_classes  = sorted(y.unique())
    islenmis_siniflar = []

    # Her aşama için en iyi modeli ve verisini sakla
    asama_modeller = []

    # 3. Her aşama için en iyi modeli bul
    for asama_no, hedef_sinif in enumerate(unique_classes[:-1], start=1):
        sinif_adi = SINIF_ISIMLERI.get(hedef_sinif, str(hedef_sinif))
        print(f"\nAsama {asama_no}: {sinif_adi} vs Others")

        filtre     = ~y_original.isin(islenmis_siniflar)
        X_filtered = X_original[filtre].reset_index(drop=True)
        y_filtered = y_original[filtre].reset_index(drop=True)

        y_binary = (y_filtered == hedef_sinif).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_binary,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_binary
        )

        en_iyi_model, en_iyi_isim = en_iyi_modeli_sec(X_train, X_test, y_train, y_test)

        asama_modeller.append({
            "asama_no":  asama_no,
            "sinif":     hedef_sinif,
            "sinif_adi": sinif_adi,
            "model":     en_iyi_model,
            "model_adi": en_iyi_isim,
            "X":         X_filtered,
            "y":         y_binary
        })

        islenmis_siniflar.append(hedef_sinif)

    # 4. Learning curve grafikleri — 3x3 subplot
    n       = len(asama_modeller)
    cols    = 3
    rows    = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes    = axes.flatten()

    plt.suptitle("Learning Curves - Cascade Classification\n(Best Model per Stage)", fontsize=14)

    ozet = []
    for i, asama in enumerate(asama_modeller):
        baslik = f"Stage {asama['asama_no']}: {asama['sinif_adi']}\n({asama['model_adi']})"
        max_test, gap = learning_curve_ciz(
            axes[i], asama["model"], asama["X"], asama["y"], baslik
        )
        ozet.append({
            "Stage":      asama["asama_no"],
            "Class":      asama["sinif_adi"],
            "Best Model": asama["model_adi"],
            "Max Test":   round(max_test, 4),
            "Train-Test Gap": round(gap, 4)
        })

    # Kullanılmayan subplot'ları gizle
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("cascade_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nLearning curves kaydedildi: cascade_learning_curves.png")

    # 5. Özet tabloyu yazdır
    print("\n" + "="*70)
    print(f"{'Stage':<8} {'Class':<22} {'Model':<15} {'Max Test':>10} {'Gap':>10}")
    print("="*70)
    for s in ozet:
        print(f"{s['Stage']:<8} {s['Class']:<22} {s['Best Model']:<15} {s['Max Test']:>10.4f} {s['Train-Test Gap']:>10.4f}")
    print("="*70)
    print("\nNasil yorumlanir:")
    print("  Dusuk gap (Train-Test Farki) → Ezber yok, iyi genelleme")
    print("  Yuksek gap                   → Overfitting var")

if __name__ == "__main__":
    main()
