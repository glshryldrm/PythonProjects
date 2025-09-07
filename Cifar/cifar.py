
import pandas as pd

# Train ve test dosyalarını yükle (örneğin CSV dosyası)
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#%%
# Train verisi
X_train = train_df.iloc[:, :-1].values  # Son sütun haricindeki tüm sütunlar piksel değerleri
y_train = train_df.iloc[:, -1].values   # Son sütun etiketler

# Test verisi
X_test = test_df.iloc[:, :-1].values    # Piksel değerleri
y_test = test_df.iloc[:, -1].values     # Etiketler
#%%
# Train verisini yeniden şekillendirme (32x32x3)
X_train = X_train.reshape(-1, 32, 32, 3)

# Test verisini yeniden şekillendirme (32x32x3)
X_test = X_test.reshape(-1, 32, 32, 3)

#%%
# Piksel değerlerini 0-1 aralığına normalize et
X_train = X_train / 255.0
X_test = X_test / 255.0

#%%
from tensorflow.keras import layers, models

# Modeli oluşturma
model = models.Sequential()

# İlk Convolution katmanı
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# İkinci Convolution katmanı
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Üçüncü Convolution katmanı
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Tam bağlantı katmanları
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Modeli derleme
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modeli özetleme
model.summary()
#%%
# Modeli eğitme
history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))
#%%
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")
