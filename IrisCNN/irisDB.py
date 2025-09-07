import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
from tensorflow.keras import layers
#%%
df = pd.read_csv('Iris.csv')

# Eksik verileri kontrol et (kontrol amaçlı)
print("Eksik değerler:\n", df.isnull().sum())

#%%
from sklearn.preprocessing import OneHotEncoder

df = pd.get_dummies(df, columns=['Species'])


#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Özellikler (X) ve hedef değişken (Y) ayrımı
X = df.iloc[:, :-3]  # Son üç sütun hariç tüm sütunlar
Y = df.iloc[:, -3:]   # Son üç sütun

# Eğitim ve test setlerine ayırma (80% eğitim, 20% test)
seed=8
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

# Özelliklerin standartlaştırılması
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%
# Modeli oluşturma
model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),  # İlk katman
    layers.Dense(10, activation='relu'),  # İkinci katman
    layers.Dense(3, activation='softmax')  # Çıkış katmanı (3 sınıf)
])

# Modeli derleme
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modelin özetini görüntüleme
model.summary()
#%%
history = model.fit(X_train, Y_train, epochs=100, batch_size=5, validation_split=0)

#%%
# Test seti üzerinde modelin doğruluğunu değerlendirme
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f'Test seti doğruluğu: {test_accuracy:.2f}')
#%%
# Tahmin yapma
predictions = model.predict(X_test)

# Olasılıkları sınıf etiketlerine dönüştürme
predicted_classes = predictions.argmax(axis=1)

# Tahmin sonuçlarını orijinal sınıf etiketlerine döndürme
class_labels = ['Setosa', 'Versicolor', 'Virginica']

original_classes = [class_labels[i] for i in predicted_classes]

# Tahmin edilen sınıfları yazdırma
print("Tahmin Edilen Sınıflar:", original_classes)

