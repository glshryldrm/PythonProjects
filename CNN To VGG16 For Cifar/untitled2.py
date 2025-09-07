import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Modeli CIFAR-10 için giriş boyutuna göre özelleştir
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

from tensorflow.keras import models, layers

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),  # Overfitting'i önlemek için dropout ekleyin
    layers.Dense(10, activation='softmax')  # CIFAR-10 için 10 sınıf
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# CIFAR-10 veri setini yükleme
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# Normalize etme
x_train, x_test = x_train / 255.0, x_test / 255.0

history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

# Test doğruluğunu hesaplama
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Doğruluğu: {test_accuracy * 100:.2f}%")

# Model tahmini
results = model.predict(x_test)
predicted_classes = np.argmax(results, axis=1)

# İlk 10 görüntüyü ve tahminlerini görselleştir
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(15, 8))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title(f"Gerçek: {class_names[y_test[i][0]]}\nTahmin: {class_names[predicted_classes[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
