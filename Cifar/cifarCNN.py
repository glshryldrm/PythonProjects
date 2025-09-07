
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# CIFAR-10 veri setini yükleme
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

#%%
# Verileri normalize etme
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
#%%
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

#%%
from tensorflow.keras import layers, models

# Modeli oluşturma
model = models.Sequential()

# İlk Convolution katmanı
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# İkinci Convolution katmanı
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Üçüncü Convolution katmanı
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

# Tam bağlantı katmanları
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Modeli derleme
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modeli özetleme
model.summary()
#%%
# Modeli eğitme
history = model.fit(x_train, y_train, epochs=20, 
                    validation_data=(x_test, y_test))
#%%
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")
#%%
import matplotlib.pyplot as plt

# Eğitim ve doğrulama kayıpları
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.legend()
plt.show()

# Eğitim ve doğrulama doğruluğu
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.legend()
plt.show()
#%%
import numpy as np

# İlk 10 test verisi ve tahminlerini görselleştirelim
predictions = model.predict(x_test)

# Sınıfları 0-9 arasında olacak şekilde argmax ile alıyoruz
predicted_classes = np.argmax(predictions, axis=1)
y_test_classes = y_test.flatten()  # Etiketleri düzleştiriyoruz

# İlk 10 görseli ve tahminleri görselleştir
plt.figure(figsize=(15,15))
for i in range(10):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i])
    plt.title(f"Gerçek: {y_test_classes[i]}, Tahmin: {predicted_classes[i]}")
plt.show()
