"""
PCA - Tomato Leaf Disease Detection
-------------------------------------
57 özelliği 2D ve 3D PCA ile görselleştirir.
Sınıfların özellik uzayındaki ayrışabilirliğini gösterir.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─── Sabitler ────────────────────────────────────────────────────────────────
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

RENKLER = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#795548", "#E91E63", "#607D8B", "#FFEB3B"
]

# ─── Ana fonksiyon ────────────────────────────────────────────────────────────
def main():
    # 1. Veriyi yükle
    print("Veri yukleniyor...")
    data = pd.read_csv("tomato_all_features.csv")

    X = data.drop("Target", axis=1)
    y = data["Target"]

    # 2. StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ─── 2D PCA ──────────────────────────────────────────────────────────────
    pca_2d = PCA(n_components=2)
    X_2d   = pca_2d.fit_transform(X_scaled)

    var1 = pca_2d.explained_variance_ratio_[0] * 100
    var2 = pca_2d.explained_variance_ratio_[1] * 100

    fig, ax = plt.subplots(figsize=(10, 7))

    for sinif, renk in zip(sorted(y.unique()), RENKLER):
        maske = y == sinif
        ax.scatter(
            X_2d[maske, 0], X_2d[maske, 1],
            label=f"{sinif}: {SINIF_ISIMLERI[sinif]}",
            color=renk, alpha=0.5, s=15, edgecolors="none"
        )

    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=12)
    ax.set_title("PCA - 2D Scatter Plot\nTomato Leaf Disease Feature Space", fontsize=13)
    ax.legend(loc="upper right", fontsize=8, markerscale=2)
    plt.tight_layout()
    plt.savefig("pca_2d.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"2D PCA kaydedildi: pca_2d.png")
    print(f"PC1 + PC2 toplam aciklanan varyans: {var1 + var2:.1f}%")

    # ─── 3D PCA ──────────────────────────────────────────────────────────────
    pca_3d = PCA(n_components=3)
    X_3d   = pca_3d.fit_transform(X_scaled)

    var1 = pca_3d.explained_variance_ratio_[0] * 100
    var2 = pca_3d.explained_variance_ratio_[1] * 100
    var3 = pca_3d.explained_variance_ratio_[2] * 100

    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection="3d")

    for sinif, renk in zip(sorted(y.unique()), RENKLER):
        maske = y == sinif
        ax.scatter(
            X_3d[maske, 0], X_3d[maske, 1], X_3d[maske, 2],
            label=f"{sinif}: {SINIF_ISIMLERI[sinif]}",
            color=renk, alpha=0.5, s=12, edgecolors="none"
        )

    ax.set_xlabel(f"PC1 ({var1:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var2:.1f}%)", fontsize=10)
    ax.set_zlabel(f"PC3 ({var3:.1f}%)", fontsize=10)
    ax.set_title("PCA - 3D Scatter Plot\nTomato Leaf Disease Feature Space", fontsize=13)
    ax.legend(loc="upper left", fontsize=7, markerscale=2)
    plt.tight_layout()
    plt.savefig("pca_3d.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"3D PCA kaydedildi: pca_3d.png")
    print(f"PC1 + PC2 + PC3 toplam aciklanan varyans: {var1 + var2 + var3:.1f}%")

    # ─── Explained Variance (Scree Plot) ─────────────────────────────────────
    pca_full = PCA()
    pca_full.fit(X_scaled)

    kumulatif = np.cumsum(pca_full.explained_variance_ratio_) * 100
    bireysel  = pca_full.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, len(bireysel) + 1), bireysel, alpha=0.6, color="steelblue", label="Individual")
    ax.plot(range(1, len(kumulatif) + 1), kumulatif, "r-o", markersize=4, label="Cumulative")
    ax.axhline(y=90, color="gray", linestyle="--", linewidth=1, label="90% threshold")
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Explained Variance (%)", fontsize=12)
    ax.set_title("PCA - Explained Variance (Scree Plot)", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("pca_scree.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Scree plot kaydedildi: pca_scree.png")

    # Kaç bileşen %90 varyansı açıklıyor?
    n_90 = np.argmax(kumulatif >= 90) + 1
    print(f"\n%90 varyans icin gereken bilesen sayisi: {n_90} / {X.shape[1]}")

if __name__ == "__main__":
    main()
