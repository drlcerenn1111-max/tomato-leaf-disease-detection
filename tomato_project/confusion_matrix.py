import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay

# Veriyi oku
data = pd.read_csv("tomato_all_features.csv")

# Özellikler (X) ve hedef değişken (y)
X = data[['BrennerR', 'BrennerG', 'BrennerB', 
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
          'StR', 'StG', 'StB']]
y = data['Target']

# MCC değerlerini ve confusion matrix'leri saklamak için listeler
mccs = []
conf_matrices = []

# 1000 tekrar için döngü
for _ in tqdm(range(1000)):
    # Veriyi karıştır
    data_shuffled = data.sample(frac=1, random_state=None)
    
    # X ve y'yi ayır
    X_shuffled = data_shuffled[X.columns]
    y_shuffled = data_shuffled['Target']
    
    # Eğitim ve test seti
    X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.20, random_state=42)
    
    # Modeli oluştur ve eğit
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Tahmin yap
    predict = model.predict(X_test)
    
    # MCC hesapla
    mcc = matthews_corrcoef(y_test, predict)
    mccs.append(mcc)
    
    # Confusion matrix hesapla ve kaydet
    cm = confusion_matrix(y_test, predict)
    conf_matrices.append(cm)

# MCC sonuçlarını görselleştir
plt.hist(mccs, bins=30, edgecolor='k')
plt.title('MCC Dağılımı')
plt.xlabel('MCC Değeri')
plt.ylabel('Frekans')
plt.show()

# Ortalama ve standart sapma
print(f"Ortalama MCC: {np.mean(mccs):.4f}")
print(f"Standart Sapma: {np.std(mccs):.4f}")

# Son iterasyon confusion matrix görselleştirme
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrices[-1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Son Iterasyon Confusion Matrix")
plt.show()

# Terminalde matris halinde yazdırma
print("Son iterasyon Confusion Matrix (matris olarak):")
print(conf_matrices[-1])
