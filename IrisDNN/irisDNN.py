"""
Created on Mon Dec 16 14:24:19 2024

@author: glshr
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

#%% Veri setini yükleme
df = pd.read_csv('Iris.csv')

# Eksik değer kontrolü
print("Eksik değerler:\n", df.isnull().sum())

# Sınıf etiketlerini encode etme (Label Encoding)
label_encoder = LabelEncoder()
df['Species'] = label_encoder.fit_transform(df['Species'])

# Özellikler (X) ve hedef değişken (Y) ayrımı
X = df.iloc[:, :-1].values  # 'Species' hariç tüm sütunlar
Y = tf.keras.utils.to_categorical(df['Species'])  # Sınıfları one-hot encode yap

# Eğitim ve test setlerine ayırma
seed = 8
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

# Özelliklerin ölçeklendirilmesi (Standardizasyon)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% Modeli oluşturma
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # İlk katman
    Dropout(0.3),  # Overfitting'i önlemek içiin dropout
    Dense(32, activation='relu'),  # İkinci gizli katman
    Dropout(0.3),  # Dropout
    Dense(Y.shape[1], activation='softmax')  # Çıkış katmanı
])
from tensorflow.keras.optimizers import Adam

# Adam optimizörünü öğrenme oranı ile tanımlama
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)

# Modeli derleme
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modelin özetini görüntüleme
model.summary()

#%% Modeli eğitme
history = model.fit(X_train, Y_train, 
                    epochs=7, 
                    batch_size=8, 
                    validation_split=0.2, 
                    verbose=1)

#%% Test seti üzerinde doğruluk değerlendirme
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f'Test seti doğruluğu: {test_accuracy:.2f}')

#%% Tahmin yapma
predictions = model.predict(X_test)

# Olasılıkları sınıf etiketlerine dönüştürme
predicted_classes = predictions.argmax(axis=1)
true_classes = Y_test.argmax(axis=1)

# Tahmin edilen sınıflar ve orijinal sınıfları karşılaştırma
print(classification_report(true_classes, predicted_classes, target_names=label_encoder.classes_))
 