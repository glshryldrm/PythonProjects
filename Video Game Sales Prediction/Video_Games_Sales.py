import pandas as pd

# CSV dosyasını yükle
df = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv') #https://www.kaggle.com/datasets/xtyscut/video-games-sales-as-at-22-dec-2016csv

# "Global_Sales" sütununa kadar olan sütunları seç
columns_to_keep = df.loc[:, :'Global_Sales'].columns

# Yeni veri çerçevesi oluştur
df_subset = df[columns_to_keep]

# Eksik verileri kontrol et (kontrol amaçlı)
print("Eksik değerler:\n", df_subset.isnull().sum())

# Sayısal sütunları ortalama ile doldur
numeric_columns = df_subset.select_dtypes(include=['number']).columns
df_subset[numeric_columns] = df_subset[numeric_columns].fillna(df_subset[numeric_columns].mean())

# Kategorik sütunları en sık görülen değer ile doldur
categorical_columns = df_subset.select_dtypes(include=['object']).columns
for column in categorical_columns:
    df_subset[column] = df_subset[column].fillna(df_subset[column].mode()[0])

# Eksik verileri kontrol et (doldurma sonrası)
print("Eksik değerler doldurulduktan sonra:\n", df_subset.isnull().sum())

# Düzenlenmiş veri çerçevesini yeni bir CSV dosyasına kaydet
df_subset.to_csv('doldurulmus_bolunmus_veri.csv', index=False)
#%%
# Temizlenmiş veri setini yükle
df = pd.read_csv('doldurulmus_bolunmus_veri.csv')

# İlk birkaç satırı görüntüleyerek veriyi inceleyin
print(df.head())
#%%
# Kategorik sütunları tespit edin
categorical_columns = df.select_dtypes(include=['object']).columns

# Kategorik sütunları one-hot encoding ile sayısal verilere dönüştürün
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# İlk birkaç satırı tekrar görüntüleyerek veriyi inceleyin
print(df.head())


#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Hedef değişken ve özelliklerin belirlenmesi
X = df.drop(columns=['Global_Sales'])
y = df['Global_Sales']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özelliklerin standartlaştırılması
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#%%
from sklearn.linear_model import LinearRegression

# Lineer regresyon modeli oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)
#%%
from sklearn.svm import SVR

# SVR modeli oluşturma ve eğitme
svr_model = SVR()
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)
#%%
from sklearn.tree import DecisionTreeRegressor

# Karar Ağacı Regresyonu modeli oluşturma ve eğitme
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
#%%
from sklearn.ensemble import RandomForestRegressor

# Rastgele Orman Regresyonu modeli oluşturma ve eğitme
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

#%%
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Performans metrikleri ve görselleştirme için fonksiyon
def plot_performance(y_test, y_pred, model_name):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} Mean Squared Error:", mse)
    print(f"{model_name} R^2 Score:", r2)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Gerçek Değerler')
    plt.ylabel('Tahmin Edilen Değerler')
    plt.title(f'Gerçek vs Tahmin Edilen Değerler ({model_name})')
    plt.text(max(y_test)*0.6, max(y_pred)*0.9, f"MSE: {mse:.2f}\nR^2: {r2:.2f}", 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

# Decision Tree model performansı
plot_performance(y_test, y_pred_dt, 'Decision Tree')

# Diğer modeller için performans metrikleri ve görselleştirme
models = {
    'Linear Regression': (y_pred, model),
    'SVR': (y_pred_svr, svr_model),
    'Random Forest': (y_pred_rf, rf_model)
}

for model_name, (y_pred, model_instance) in models.items():
    plot_performance(y_test, y_pred, model_name)


#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Temizlenmiş veri setini yükle
df = pd.read_csv('doldurulmus_bolunmus_veri.csv')

# İlk birkaç satırı görüntüleyerek veriyi inceleyin
print(df.head())

# Kategorik sütunları tespit edin
categorical_columns = df.select_dtypes(include=['object']).columns

# Kategorik sütunları one-hot encoding ile sayısal verilere dönüştürün
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# İlk birkaç satırı tekrar görüntüleyerek veriyi inceleyin
print(df.head())

# Veri çerçevesinin özet istatistiklerini görüntüleyin
print(df.describe())
#%%
# Veri setinin boyutunu kontrol et
print(f"Veri setinin boyutu: {df.shape}")

# Küçük bir örnekleme alın (örneğin, 1000 satır)
df_sample = df.sample(n=1000, random_state=42)

# Veri çerçevesinin özet istatistiklerini görüntüleyin
print(df_sample.describe())

# Sütunlar arasındaki korelasyonları ısı haritası ile gösterin
plt.figure(figsize=(10, 8))
sns.heatmap(df_sample.corr(), annot=True, cmap='coolwarm')
plt.show()

# Hedef değişkenin dağılımını inceleyin
sns.histplot(df_sample['Global_Sales'], kde=True)
plt.title('Global Sales Distribution')
plt.show()

#%%
# Hedef değişken ve özelliklerin belirlenmesi
X = df.drop(columns=['Global_Sales'])
y = df['Global_Sales']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özelliklerin standartlaştırılması
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#%%
# Lineer regresyon modeli oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)
#%%
# Performans metrikleri
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Gerçek ve tahmin edilen değerlerin karşılaştırılması
plt.scatter(y_test, y_pred_rf)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek vs Tahmin Edilen Değerler')
plt.show()
#%%
# SVR modeli oluşturma ve eğitme
svr_model = SVR()
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)
#%%
# Performans metrikleri
mse = mean_squared_error(y_test, y_pred_svr)
r2 = r2_score(y_test, y_pred_svr)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Gerçek ve tahmin edilen değerlerin karşılaştırılması
plt.scatter(y_test, y_pred_svr)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek vs Tahmin Edilen Değerler')
plt.show()
#%%
# Karar Ağacı Regresyonu modeli oluşturma ve eğitme
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
#%%
# Performans metrikleri
mse = mean_squared_error(y_test, y_pred_dt)
r2 = r2_score(y_test, y_pred_dt)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Gerçek ve tahmin edilen değerlerin karşılaştırılması
plt.scatter(y_test, y_pred_dt)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek vs Tahmin Edilen Değerler')
plt.show()
#%%
# Rastgele Orman Regresyonu modeli oluşturma ve eğitme
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
#%%
# Performans metrikleri
mse = mean_squared_error(y_test, y_pred_rf)
r2 = r2_score(y_test, y_pred_rf)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Gerçek ve tahmin edilen değerlerin karşılaştırılması
plt.scatter(y_test, y_pred_rf)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek vs Tahmin Edilen Değerler')
plt.show()
#%%
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Hedef değişkenin sınıflarını belirleme
y_class = pd.cut(y, bins=3, labels=['Low', 'Medium', 'High'])

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Özelliklerin standartlaştırılması
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# K-En Yakın Komşu Sınıflandırıcısı modeli oluşturma ve eğitme
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Destek Vektör Makineleri Sınıflandırıcısı modeli oluşturma ve eğitme
svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Karar Ağaçları Sınıflandırıcısı modeli oluşturma ve eğitme
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Rastgele Orman Sınıflandırıcısı modeli oluşturma ve eğitme
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Performans metrikleri
accuracy_knn = accuracy_score(y_test, y_pred_knn)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("K-Nearest Neighbors Accuracy:", accuracy_knn)
print("Support Vector Machine Accuracy:", accuracy_svm)
print("Decision Tree Accuracy:", accuracy_dt)
print("Random Forest Accuracy:", accuracy_rf)

# Sınıflandırma raporları
print("\nK-Nearest Neighbors Classification Report:")
print(classification_report(y_test, y_pred_knn))

print("\nSupport Vector Machine Classification Report:")
print(classification_report(y_test, y_pred_svm))

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Karışıklık matrisleri
plt.figure(figsize=(10, 8))
plt.subplot(221)
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, cmap='Blues', fmt='g')
plt.title('K-Nearest Neighbors Confusion Matrix')

plt.subplot(222)
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, cmap='Greens', fmt='g')
plt.title('Support Vector Machine Confusion Matrix')

plt.subplot(223)
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, cmap='Reds', fmt='g')
plt.title('Decision Tree Confusion Matrix')

plt.subplot(224)
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, cmap='Oranges', fmt='g')
plt.title('Random Forest Confusion Matrix')

plt.tight_layout()
plt.show()
