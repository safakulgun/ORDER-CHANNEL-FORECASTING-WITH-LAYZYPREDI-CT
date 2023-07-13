import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from lazypredict.Supervised import LazyClassifier



# Veri setini yükleyin
df = pd.read_csv("/Users/munzurulgun/Downloads/FLOMusteriSegmentasyonu/flo_data_20k.csv")
df.info()
# İstenmeyen sütunları çıkar
df = df.drop(['master_id', 'first_order_date', 'last_order_date', 'last_order_date_online', 'last_order_date_offline'], axis=1)
df.info()
# Sınıflandırma için verileri hazırlayın
X = df.drop('order_channel', axis=1)
y = df['order_channel']

# Kategorik sütunları sayısal değerlere dönüştürün
le = LabelEncoder()
X_encoded = X.apply(le.fit_transform)

# Eğitim ve test kümelerini ayırın
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Sınıflandırma modelini oluşturun ve eğitin
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test veri kümesi üzerinde tahmin yapın
y_pred = model.predict(X_test)

# Modelin performansını değerlendirin
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Kategorik sütunları sayısal değerlere dönüştürün
le = LabelEncoder()
X_encoded = X.apply(le.fit_transform)

# Eğitim ve test kümelerini ayırın
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# LazyClassifier modelini oluşturun ve performans değerlendirmesini yapın
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# Model performansını yazdırın
print(models)

# Model seçimi ve eğitimi
model = AdaBoostClassifier()
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)

# Performans değerlendirmesi
accuracy = accuracy_score(y_test, y_pred)
precision_micro = precision_score(y_test, y_pred, average='micro')
recall_micro = recall_score(y_test, y_pred, average='micro')

print("Accuracy:", accuracy)
print("Precision:", precision_micro)
print("Recall:", recall_micro)

# F1-score, confusion matrix,cross-validation
# F1-score'u hesaplayın
f1_micro = f1_score(y_test, y_pred, average='micro')
print("F1-Score (Micro):", f1_micro)

# Confusion Matrix'i hesaplayın
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
# Çapraz doğrulama yapın
cv_scores = cross_val_score(model, X_encoded, y, cv=5)  # 5 katlı çapraz doğrulama
print("Cross-Validation Scores:", cv_scores)
 #Grafik
# Gerçek ve tahmin edilen sipariş kanallarını içeren bir DataFrame oluşturun
comparison_df = pd.DataFrame({'Gerçek': y_test, 'Tahmin': y_pred})

# Sipariş kanalı değerlerini sıralayın
order_channels = sorted(df['order_channel'].unique())

# Her bir sipariş kanalı için gerçek ve tahmin edilen sayıları hesaplayın
real_counts = [len(comparison_df[comparison_df['Gerçek'] == channel]) for channel in order_channels]
predicted_counts = [len(comparison_df[comparison_df['Tahmin'] == channel]) for channel in order_channels]

# Sipariş kanalı tahmini grafiğini oluşturun
plt.figure(figsize=(10, 6))
plt.bar(order_channels, real_counts, label='Gerçek')
plt.bar(order_channels, predicted_counts, alpha=0.7, label='Tahmin')
plt.xlabel('Sipariş Kanalı')
plt.ylabel('Sayı')
plt.title('Gerçek ve Tahmin Edilen Sipariş Kanalları')
plt.legend()
plt.show()