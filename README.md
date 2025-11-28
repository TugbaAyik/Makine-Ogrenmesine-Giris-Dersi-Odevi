#  Chess Match Outcome Prediction — Logistic Regression

Bu proje, bir satranç veri seti kullanılarak oyuncu rating’leri, oyun süresi (time control) ve açılış hamlesi gibi özelliklerden **maç kazananını tahmin etmek** amacıyla Logistic Regression modeli kurar. Projede veri temizleme, görselleştirme, feature engineering, model eğitimi ve değerlendirme adımları bulunmaktadır.

##  Proje Amacı

Bu çalışmanın asıl amacı, satranç oyunlarında **kazanan tarafı tahmin etmek** için Logistic Regression modelinin uygulanmasıdır.
Model;

* **white_rating**
* **black_rating**
* **time_control**
* **opening (ilk hamle)**
  gibi özelliklerden yola çıkarak “winner” değerini tahmin eder.

## Kullanılan Kütüphaneler

Projede kullanılan temel Python kütüphaneleri:

```python
pandas, numpy, matplotlib, seaborn
scikit-learn (LabelEncoder, LogisticRegression, train_test_split)
```

## Veri Seti Açıklaması

Veri Kaggle üzerinde yayınlanan bir satranç oyunları veri setidir.

Projede kullanılan önemli sütunlar:

* **white_rating**: Beyaz oyuncunun ratingi
* **black_rating**: Siyah oyuncunun ratingi
* **winner**: Kazanan taraf (“white”, “black”, “draw”)
* **pgn**: Oyunun hamlelerini içeren PGN formatı
* **time_control**: Oyun süresi

##  Veri Ön İşleme

1. **Başlamamış oyunların kaldırılması**

```python
df = df[df['status'] != 'noStart']
```

2. **Eksik değerlerin silinmesi**

```python
df = df.dropna(subset=['white_rating', 'black_rating', 'winner', 'pgn', 'time_control'])
```

3. **25000 satırlık örnek seçimi**

```python
df = df.sample(n=25000, random_state=42)
```

## Açılış Hamlesi (Opening) Çıkarımı

PGN formatındaki oyunun *ilk hamlesi* modellenmiştir.

```python
def simplify_opening(pgn):
    moves = pgn.strip().split()
    return moves[0] if len(moves) > 0 else "Unknown"

df["Opening"] = df["pgn"].apply(simplify_opening)
```

Bu işlem daha sonra Label Encoding ile sayısal hale getirilir.

## Veri Görselleştirme

Projede yapılan başlıca grafikler:

### Kazanan Dağılımı

* Hangi tarafın daha çok kazandığını gösterir.

### Rating Dağılımı

* Beyaz ve siyah oyuncu rating histogramları.

### En Popüler 10 Açılış

* İlk hamleye göre sıralanmış açılış popülerlik grafiği.

## Model Eğitimi

Model için kullanılan özellikler:

```python
features = ["white_rating", "black_rating", "time_control", "Opening"]
target = "winner"
```

Kategorik değişkenler LabelEncoder ile sayısallaştırılır:

```python
le_opening.fit_transform(...)
le_time.fit_transform(...)
```

Model eğitimi:
Modeli eğitmek için veri setimizi %80 eğitim ve %20 test için olmak üzere parçalıyoruz.

```python
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
```

## Model Değerlendirme

Modelin başarı ölçütleri:

* **Accuracy Score**
* **Classification Report**
* **Confusion Matrix**

Örnek:

```python
print(classification_report(y_test, y_pred, target_names=le_y.classes_))
```

## ROC ve Çok Sınıflı F1 Analizi

Veri seti **3 sınıf** içerdiğinden klasik ROC eğrisi yalnızca binary veri için çalışır.

Kod:

```python
if len(le_y.classes_) == 2:
    ...
else:
    çok sınıflı F1-score karşılaştırma grağiği
```

Bu nedenle projede 3 sınıf için:

F1-score
Sınıf bazlı başarı dağılımı

gösterilmiştir.

## Özellik Önem Dereceleri

Logistic Regression katsayılarının mutlak değerleri alınarak özellik önem sıralaması çıkarılır:

```python
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': np.abs(model.coef_[0])
})
```

Bu grafik modelin hangi değişkenlere daha çok ağırlık verdiğini gösterir.


## Sonuç

Bu proje, satranç oyuncularının ratingleri ve oyun açılış hareketi gibi bilgiler kullanılarak maç sonucunu tahmin eden geniş kapsamlı bir Logistic Regression uygulamasıdır. Veri temizleme, etiketleme, model eğitimi ve performans görselleştirmelerinin tamamı ayrıntılı bir şekilde yapılmıştır.



