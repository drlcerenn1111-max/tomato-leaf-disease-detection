"""
Cascade Classification Implementation
-----------------------------------
Bu modül, sırasıyla her bir sınıfın diğerlerinden ayrılmasına odaklanan 
bir dizi ikili sınıflandırıcı ile kaskad sınıflandırma yaklaşımını uygular. 
Her sınıflandırma adımından sonra, tanımlanan sınıf veri setinden çıkarılır 
ve süreç kalan sınıflarla devam eder.

ceren dereli
Tarih: May 7, 2025
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class CascadeClassifier:
    """
    Bir dizi ikili (binary) sınıflandırıcıyı eğiten ve her bir sınıflandırıcının 
    bir sınıfı diğer tüm sınıflardan ayırdığı kademeli (cascade) sınıflandırıcı.
    """
    
    def __init__(self, models_to_try=None, random_state=42):
        """
        CascadeClassifier sınıfının yapıcı (constructor) fonksiyonu.
        Sınıflandırıcıyı başlatır.
        
        Parametreler:
        -------------
        models_to_try : dict
            Her bir ikili sınıflandırma görevi için denenecek sklearn model nesnelerinin sözlüğü (dictionary).
            Eğer None verilirse, varsayılan (default) model kümesi kullanılacaktır.
            
        random_state : int
            Sonuçların tekrarlanabilir (reproducible) olması için rastgelelik (randomness) tohum değeri.
        """
        self.random_state = random_state  # Rastgelelik tohum değeri (tekrarlanabilirlik için)
        self.binary_classifiers = {}      # Her aşama (stage) için en iyi sınıflandırıcıyı saklar
        self.best_model_names = {}        # Her aşama için en iyi modelin adını saklar
        self.class_order = []             # Sınıfların sınıflandırıldığı sırayı saklar

        # Eğer kullanıcı models_to_try parametresini belirtmediyse (None ise), varsayılan modeller tanımlanır
        if models_to_try is None:
            self.models_to_try = {
                "Ridge": RidgeClassifier(random_state=random_state),   # Ridge sınıflandırıcısı (L2 düzenlemeli doğrusal model)
                "KNN": KNeighborsClassifier(n_neighbors=5),            # K-En Yakın Komşu (K-Nearest Neighbors), k=5
                "LDA": LinearDiscriminantAnalysis(),                   # Lineer Ayırıcı Analiz (Linear Discriminant Analysis)
                "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),                # Kuadratik Ayırıcı Analiz (Quadratic Discriminant Analysis)
                "Decision Tree": DecisionTreeClassifier(random_state=random_state),  # Karar Ağacı Sınıflandırıcısı
                "Random Forest": RandomForestClassifier(random_state=random_state),  # Rastgele Orman (Random Forest)
                "SVM": SVC(probability=True, random_state=random_state)             # Destek Vektör Makineleri (Support Vector Machine), olasılık tahmini açık
            }
        else:
            # Eğer kullanıcı kendi modellerini verdiyse, onları kullan
            self.models_to_try = models_to_try

    def _prepare_binary_task(self, X, y, current_class):
        """
        Çok sınıflı veriyi ikili sınıflandırma formatına dönüştürür. 
        Seçilen sınıf (current_class) **0** olarak etiketlenir (bu sınıf bu adımda sınıflandırılacak), 
        diğer tüm sınıflar **1** olarak etiketlenir (kalan sınıflar).

        Bu işlem, her adımda bir sınıfın ayrılıp elimine edilmesini sağlar.

        Parametreler:
        -------------
        X : DataFrame
            Özellik matrisi (bağımsız değişkenler)
        y : Series
            Hedef vektörü (sınıf etiketleri)
        current_class : int veya str
            Bu adımda ayrılacak (0 olarak etiketlenecek) sınıf
        
        Dönen (Return):
        ----------------
        X_binary : DataFrame
            Özellik matrisi (X aynen kalır)
        y_binary : Series
            İkili hedef vektörü:
            - current_class → 0
            - diğer sınıflar → 1
        """
        # Orijinal veriyi değiştirmemek için y'nin bir kopyasını oluştur
        y_binary = y.copy()
        
        # Yeniden etiketleme işlemi:
        # current_class → 0
        # diğer tüm sınıflar → 1
        y_binary = (y_binary != current_class).astype(int)
        
        # X aynen döndürülür, y ikili forma çevrilmiş şekilde döndürülür
        return X, y_binary
    
    def _evaluate_models(self, X_train, X_test, y_train, y_test, verbose=True):
        """
        Hazırlanan ikili sınıflandırma görevi üzerinde tüm modelleri değerlendirir.
        En iyi performansı gösteren modeli seçer ve detaylı performans raporlarını gösterir.

        Parametreler:
        -------------
        X_train, X_test : DataFrame
            Eğitim ve test veri setleri (özellik matrisi)
        y_train, y_test : Series
            Eğitim ve test hedef vektörleri (ikili etiketli)
        verbose : bool
            Sonuçların ekrana yazdırılıp yazdırılmayacağını belirler (varsayılan: True)
        
        Dönen (Return):
        ----------------
        best_model : sklearn modeli (object)
            En iyi performans gösteren model nesnesi (eğitimli haliyle)
        best_model_name : str
            En iyi modelin adı (sözlük anahtarı olarak kullanılan isim)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        # Tüm modellerin test başarımlarını saklayacağımız sözlük
        results = {}

        # verbose=True ise, değerlendirme başladığını belirt
        if verbose:
            print("\nİkili sınıflandırma görevi için modeller değerlendiriliyor:")

        # self.models_to_try içindeki her bir modeli sırayla deniyoruz
        for name, model in self.models_to_try.items():
            # Modeli eğitim verisi ile eğitiyoruz
            model.fit(X_train, y_train)

            # Test verisi üzerinde doğruluk (accuracy) hesaplıyoruz
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Sonuçları sözlüğe kaydediyoruz
            results[name] = accuracy

            # verbose=True ise modelin doğruluğunu yazdır
            if verbose:
                print(f"{name} doğruluk (accuracy): {accuracy:.4f}")

        # En iyi doğruluk skoruna sahip modeli buluyoruz
        best_model_name = max(results, key=results.get)
        best_model = self.models_to_try[best_model_name]
        best_accuracy = results[best_model_name]

        # verbose=True ise en iyi modeli yazdır
        if verbose:
            print(f"\nEn iyi model: {best_model_name} | Doğruluk (accuracy): {best_accuracy:.4f}")

            # En iyi modelin detaylı sınıflandırma raporunu göster
            y_pred = best_model.predict(X_test)
            print("\nSınıflandırma Raporu (Classification Report):")
            print(classification_report(y_test, y_pred, target_names=["Diğer Sınıflar (0)", "Mevcut Sınıf (1)"]))

            # Karışıklık matrisi (confusion matrix) oluştur ve görselleştir
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"{best_model_name} - Karışıklık Matrisi (Confusion Matrix)")
            plt.xlabel("Tahmin Edilen (0: Diğer, 1: Mevcut Sınıf)")
            plt.ylabel("Gerçek (0: Diğer, 1: Mevcut Sınıf)")
            plt.show()

        # En iyi modeli ve model adını döndür
        return best_model, best_model_name

    def fit(self, X, y, test_size=0.2, verbose=True):
        """
        Cascade sınıflandırıcısını verilen veri seti üzerinde eğitir.

        Parametreler:
        -------------
        X : DataFrame
            Özellik matrisi (bağımsız değişkenler)
        y : Series
            Hedef vektörü (çoklu sınıflar içeren bağımlı değişken)
        test_size : float
            Veri setinin ne kadarının test için ayrılacağı oran (varsayılan: %20 → 0.2)
        verbose : bool
            Eğitim ve değerlendirme süreçlerinde çıktıların yazdırılıp yazdırılmayacağını belirler (varsayılan: True)

        Dönen (Return):
        ----------------
        self : object
            Kendi nesnesini döndürür (method chaining için)
        """

        # Hedef vektöründeki benzersiz sınıfları alıyoruz (başlangıçta tüm sınıflar işlenecek)
        remaining_classes = list(y.unique())

        # verbose=True ise, kaç sınıf olduğunu ve sınıf dağılımını yazdır
        if verbose:
            print(f"Cascade sınıflandırmaya {len(remaining_classes)} sınıfla başlanıyor")
            print(f"Sınıf dağılımı: {y.value_counts().to_dict()}")

        # Orijinal veriyi değiştirmemek için kopyalarını alıyoruz
        X_remaining = X.copy()
        y_remaining = y.copy()

        # Her adımda (son 1 sınıf kalana kadar) sınıfları işlemeye başlayalım
        stage = 0  # Hangi adımda olduğumuzu takip etmek için aşama sayacı

        while len(remaining_classes) > 1:
            # Sıradaki işlenecek sınıf → remaining_classes listesindeki ilk sınıf
            current_class = remaining_classes[0]

            # verbose=True ise aşama bilgisini yazdır
            if verbose:
                print(f"\n--- Aşama {stage+1}: Sınıf '{current_class}' işleniyor ---")
                print(f"Kalan sınıflar: {remaining_classes}")


            if verbose:
                # Bu aşamaya dair detaylı bilgi yazdırılır
                print(f"\n{'='*50}")
                print(f"Aşama {stage+1}: '{current_class}' sınıfı diğerlerine karşı sınıflandırılıyor")
                print(f"Kalan sınıflar: {remaining_classes}")
                print(f"Mevcut veri boyutu: {X_remaining.shape}")
                print(f"Sınıf dağılımı: {y_remaining.value_counts().to_dict()}")
                print('='*50)

            # Binary sınıflandırma görevi için veriyi hazırla
            # current_class → 1 (pozitif), diğer sınıflar → 0 (negatif)
            X_binary, y_binary = self._prepare_binary_task(X_remaining, y_remaining, current_class)

            # Veriyi eğitim ve test setine ayır
            X_train, X_test, y_train, y_test = train_test_split(
                X_binary, y_binary, test_size=test_size,
                random_state=self.random_state, stratify=y_binary
            )

            # Bu binary görev için en iyi modeli bul ve eğit
            best_model, best_model_name = self._evaluate_models(
                X_train, X_test, y_train, y_test, verbose=verbose
            )

            # Eğitilen en iyi modeli sakla (cascade model yapısına ekleniyor)
            self.binary_classifiers[stage] = best_model
            self.best_model_names[stage] = best_model_name
            self.class_order.append(current_class)

            # İşlenen (sınıflandırılan) sınıfı veri setinden çıkar
            mask = (y_remaining != current_class)
            X_remaining = X_remaining[mask]
            y_remaining = y_remaining[mask]

            # Kalan sınıflar listesini güncelle
            remaining_classes = list(y_remaining.unique())

            # Bir sonraki aşama için stage sayacını artır
            stage += 1
            
        return self

    def predict(self, X):
        """
        Eğitilmiş cascade sınıflandırıcısını kullanarak sınıf etiketlerini tahmin eder.

        Parametreler:
        -----------
        X : DataFrame veya array-like
            Tahmin edilecek örnekler (veriler)

        Dönen:
        --------
        y_pred : array
            Tahmin edilen sınıf etiketleri
        """
        # Giriş verisinin kopyasını alıyoruz, böylece her aşamada filtreleme yapılabilir
        X_remaining = X.copy()
        indices_remaining = np.arange(len(X))  # Her örneğin orijinal indekslerini tutar

        # Başlangıçta tahminleri son sınıfla dolduruyoruz
        # (İlk aşamalarda sınıflandırılmayan örnekler son sınıfa atanacaktır)
        predictions = np.full(len(X), self.class_order[-1])  # Son sınıf, tüm örneklerin varsayılan sınıfıdır

        # Sırasıyla her binary (ikili) sınıflandırıcıyı uyguluyoruz
        for stage in range(len(self.binary_classifiers)):
            if len(indices_remaining) == 0:  # Eğer hiçbir örnek kalmadıysa, döngüye devam etme
                break

            # Bu aşama için kullanılacak sınıflandırıcıyı alıyoruz
            classifier = self.binary_classifiers[stage]
            current_class = self.class_order[stage]  # Mevcut aşamada işlenecek sınıf

            # Her ikili sınıflandırıcı, bu aşamada tahmin yapar
            binary_preds = classifier.predict(X_remaining)  # Sınıflandırıcıyı kullanarak tahminler yapılır

            # Pozitif sınıfı (current_class) tahmin eden örneklerin indekslerini alıyoruz
            current_class_indices = indices_remaining[binary_preds == 1]  # Eğer tahmin 1 ise, bu örnek doğru sınıfa ait demektir

            # Bu örneklere mevcut sınıfı atıyoruz
            predictions[current_class_indices] = current_class

            # Tahmin edilen sınıflar çıkarıldıktan sonra, kalan örnekleri belirliyoruz
            remain_mask = (binary_preds == 0)  # 0 (negatif) olarak tahmin edilen örnekler devam edecektir
            X_remaining = X_remaining[remain_mask]
            indices_remaining = indices_remaining[remain_mask]

        return predictions  # Nihai tahminler geri döndürülür

    def predict_proba(self, X):
        """
        Her örnek için sınıf olasılıklarını tahmin eder.
        Not: Bu, cascade yapısına dayalı bir yaklaşımdır.
        
        Parametreler:
        -----------
        X : DataFrame veya array-like
            Tahmin yapılacak örnekler
            
        Dönüş:
        ------
        probas : (n_samples, n_classes) şekline sahip bir dizi
            Her örnek için sınıf olasılıkları
        """
        n_samples = len(X)  # Örnek sayısını alıyoruz
        n_classes = len(self.class_order)  # Sınıf sayısını alıyoruz
        
        # Olasılık matrisini sıfırlayarak başlatıyoruz
        probas = np.zeros((n_samples, n_classes))
        
        # Sonraki filtreleme işlemleri için giriş verisini kopyalıyoruz
        X_remaining = X.copy()
        indices_remaining = np.arange(n_samples)  # Örneklerin indekslerini saklıyoruz
        
        # Her binary sınıflandırıcıyı sırasıyla uyguluyoruz
        for stage in range(len(self.binary_classifiers)):
            if len(indices_remaining) == 0:  # Eğer işlenecek örnek kalmazsa, döngüyü bitir
                break
            
            # Bu aşama için sınıflandırıcıyı alıyoruz
            classifier = self.binary_classifiers[stage]
            class_idx = self.class_order.index(self.class_order[stage])  # Mevcut sınıfın indeksini alıyoruz
            
            try:
                # Eğer predict_proba fonksiyonu varsa, bunu kullanmayı deniyoruz
                stage_probas = classifier.predict_proba(X_remaining)
                positive_probas = stage_probas[:, 1]  # Pozitif sınıfın (mevcut sınıf) olasılıkları
            except:
                # Eğer predict_proba yoksa, decision_function kullanmayı deniyoruz
                try:
                    decision_scores = classifier.decision_function(X_remaining)
                    # Decision scores'u sigmoid fonksiyonu ile olasılıklara çeviriyoruz
                    positive_probas = 1 / (1 + np.exp(-decision_scores))
                except:
                    # Eğer decision_function da yoksa, binary tahminler kullanıyoruz
                    binary_preds = classifier.predict(X_remaining)
                    positive_probas = binary_preds.astype(float)  # Binary tahminleri float tipine dönüştürüyoruz
            
            # Mevcut sınıf için olasılıkları probas matrisine atıyoruz
            probas[indices_remaining, class_idx] = positive_probas
            
            # Her örnek için "diğer" olasılığını hesaplıyoruz
            other_probas = 1 - positive_probas  # Pozitif olasılıkların zıttı olan olasılık
            
            # Bir sonraki aşama için örnekleri filtreliyoruz
            # Karar vermek için 0.5'lik bir eşik değeri kullanıyoruz
            remain_mask = (positive_probas < 0.5)  # 0.5'ten küçük olan örnekler bir sonraki aşamaya geçer
            X_remaining = X_remaining[remain_mask]  # Bu örnekler X_remaining'de kalır
            
            # Sınıflandırılmamış örnekler için olasılıkları yeniden ölçeklendiriyoruz
            scaling_factor = other_probas[remain_mask].reshape(-1, 1)  # Diğer olasılıkları ölçekliyoruz
            
            # Kalan örneklerin indekslerini güncelliyoruz
            indices_remaining = indices_remaining[remain_mask]
            
        # Son sınıf için kalan olasılıkları atıyoruz
        if indices_remaining.size > 0:
            last_class_idx = self.class_order.index(self.class_order[-1])
            probas[indices_remaining, last_class_idx] = 1.0
        
        # Olasılıkları 1'e normalize ediyoruz
        row_sums = probas.sum(axis=1).reshape(-1, 1)  # Satırlardaki toplamları alıyoruz
        row_sums[row_sums == 0] = 1  # Bölme işlemi sırasında sıfır hatasından kaçınmak için sıfırları 1'e eşitliyoruz
        probas = probas / row_sums  # Olasılıkları normalize ediyoruz
        
        return probas  # Sonuç olarak normalleşmiş olasılıkları döndürüyoruz
        
    def score(self, X, y):
        """
        Verilen test verileri ve etiketler üzerinde doğruluk skorunu hesaplar.
        
        Parametreler:
        -----------
        X : DataFrame veya array-like
            Test örnekleri
        y : Series veya array-like
            X için doğru etiketler
            
        Dönüş:
        ------
        score : float
            Doğruluk skoru
        """
        return accuracy_score(y, self.predict(X))  # Test verisi üzerinde tahminler yaparak doğruluk skorunu hesaplıyoruz

    def get_summary(self):
        """
        Cascade sınıflandırıcısının bir özetini alır.
        
        Dönüş:
        ------
        summary : dict
            Özet bilgilerini içeren sözlük
        """
        summary = {
            "class_order": self.class_order,  # Sınıf sıralaması
            "stages": len(self.binary_classifiers),  # Aşama sayısı (sınıflandırıcı sayısı)
            "best_models": self.best_model_names  # Her aşama için en iyi modellerin isimleri
        }
        return summary  # Özet bilgilerini döndürüyoruz

def main():
    """
    CascadeClassifier sınıfının örnek kullanımını gösterir.
    """
    # 1. Veriyi yükle
    print("Veri yükleniyor...")
    data = pd.read_csv("birlesik_dataset.csv")  # Veriyi CSV dosyasından okuyoruz
    
    # 2. Özellikler ve hedef değişkeni hazırla
    X = data.drop("Target", axis=1)  # Hedef dışındaki tüm sütunları özellik olarak alıyoruz
    y = data["Target"]  # Hedef değişkeni alıyoruz
    
    print(f"Veri kümesi şekli: {data.shape}")
    print(f"Sınıf dağılımı:\n{y.value_counts()}")  # Hedef değişkeninin sınıf dağılımını yazdırıyoruz
    
    # 3. Veriyi eğitim ve test olarak ayır
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # Eğitim ve test verisini ayırıyoruz
    )
    
    # 4. Cascade sınıflandırıcısını oluştur ve eğit
    print("\nCascade sınıflandırıcısı eğitiliyor...")
    cascade = CascadeClassifier(random_state=42)  # Cascade sınıflandırıcısını oluşturuyoruz
    cascade.fit(X_train, y_train, verbose=True)  # Modeli eğitiyoruz
    
    # 5. Test kümesinde değerlendir
    y_pred = cascade.predict(X_test)  # Test verisi üzerinde tahmin yapıyoruz
    accuracy = accuracy_score(y_test, y_pred)  # Doğruluk skorunu hesaplıyoruz
    
    print("\nTest kümesindeki son değerlendirme:")
    print(f"Doğruluk: {accuracy:.4f}")  # Doğruluk skorunu yazdırıyoruz
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))  # Sınıflandırma raporunu yazdırıyoruz
    
    # 6. Karışıklık Matrisi
    cm = confusion_matrix(y_test, y_pred)  # Karışıklık matrisini hesaplıyoruz
    plt.figure(figsize=(10, 8))  # Matrisin boyutunu ayarlıyoruz
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=cascade.class_order, 
                yticklabels=cascade.class_order)  # Karışıklık matrisini görselleştiriyoruz
    plt.title("Cascade Sınıflandırıcı - Karışıklık Matrisi")
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.show()  # Matrisin görselini gösteriyoruz
    
    # 7. Cascade sınıflandırıcısının özetini al
    summary = cascade.get_summary()  # Özet bilgisini alıyoruz
    print("\nCascade Sınıflandırıcı Özeti:")
    print(f"Sınıf sıralaması: {summary['class_order']}")  # Sınıf sıralamasını yazdırıyoruz
    print(f"Aşama sayısı: {summary['stages']}")  # Aşama sayısını yazdırıyoruz
    print("Her aşamadaki en iyi modeller:")
    for stage, model_name in summary['best_models'].items():
        print(f"  Aşama {stage+1} (Sınıf {cascade.class_order[stage]} vs diğerleri): {model_name}")  # Her aşama için en iyi modeli yazdırıyoruz

if __name__ == "__main__":
    main()  # Ana fonksiyonu çalıştırıyoruz