"""
Kütüphane versiyonları;
tensorflow == 2.6.0 (pip install tensorflow-gpu==2.6.0)
keras == 2.9.0 (pip install keras==2.9.0)
"""

import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import cv2
from datetime import datetime
from random import shuffle
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, confusion_matrix

tf.compat.v1.disable_eager_execution()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#%% değişkenler
image_width = 227
image_height = 227
learning_rate = 1e-5
batch_size = 32
epoch = 3
test_accuracy = 1

class_names = ["Cat", "Dog"]   

train_path = r"C:/Users/glshr/OneDrive/Belgeler/ders/ders 4. sınıf/optimizasyon algoritmaları/veriseti/train"
test_path = r"C:/Users/glshr/OneDrive/Belgeler/ders/ders 4. sınıf/optimizasyon algoritmaları/veriseti/test"
validate_path = r"C:/Users/glshr/OneDrive/Belgeler/ders/ders 4. sınıf/optimizasyon algoritmaları/veriseti/validation"

#%% verileri klasörden okuma

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator() # 1/255 yapıldığında tüm pixeller küçük değerler haline geliyor (0,1)arası
validate_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

train_ds = train_datagen.flow_from_directory(train_path, target_size=(image_width, image_height), batch_size=batch_size, class_mode="categorical")
test_ds = test_datagen.flow_from_directory(test_path, target_size=(image_width, image_height), batch_size=batch_size, class_mode="categorical", shuffle=False)
validate_ds = validate_datagen.flow_from_directory(validate_path, target_size=(image_width, image_height), batch_size=batch_size, class_mode="categorical")

#%% model oluşturma eğitim ve test
# Değişecek parametre değerleri
filter_factors = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125]
kernel_steps = [0, 2, 4]
stride_steps = [0, 1, 2]

results = []
min_accuracy = 0.5  # eğer doğruluk bunun altına inerse o modeli geç

for f_factor in filter_factors:
    layer1katman = int(96 * f_factor)
    layer2katman = int(256 * f_factor)
    layer3katman = int(384 * f_factor)
    layer4katman = int(384 * f_factor)
    layer5katman = int(256 * f_factor)

    dense1katman = int(4096 * f_factor)
    dense2katman = int(1024 * f_factor)

    for k_step in kernel_steps:
        kernel1 = (max(1, 11 - k_step), max(1, 11 - k_step))
        kernel2 = (max(1, 5 - k_step), max(1, 5 - k_step))
        kernel_rest = (max(1, 3 - k_step), max(1, 3 - k_step))

        for s_step in stride_steps:
            stride1 = (max(1, 4 - s_step), max(1, 4 - s_step))

            print(f"\n==== Filters: {f_factor}, KernelStep: {k_step}, StrideStep: {s_step} ====")

            # Model oluşturuluyor
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(layer1katman, kernel1, strides=stride1, activation='relu', input_shape=(image_width, image_height, 3)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(3, strides=(2, 2)),

                tf.keras.layers.Conv2D(layer2katman, kernel2, strides=(1, 1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(3, strides=(2, 2)),

                tf.keras.layers.Conv2D(layer3katman, kernel_rest, strides=(1, 1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(layer4katman, kernel_rest, strides=(1, 1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(layer5katman, kernel_rest, strides=(1, 1), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(3, strides=(2, 2)),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(dense1katman, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(dense2katman, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(len(class_names), activation='softmax')
            ])

            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, decay=1e-05)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            history = model.fit(train_ds, validation_data=validate_ds, epochs=epoch)

            start_test = datetime.now()
            test_result = model.evaluate(test_ds)
            end_test = datetime.now()

            accuracy = test_result[1]

            if accuracy < min_accuracy:
                print(f"❌ Doğruluk {accuracy:.4f}, 0.5'in altında olduğu için bu kombinasyon geçildi.")
                continue

            print("✅ Test Accuracy: {:.4f}".format(accuracy))
            print("Test Loss: {:.4f}".format(test_result[0]))
            print("Test Süresi:", end_test - start_test)

            results.append({
                'filters': f_factor,
                'kernel_step': k_step,
                'stride_step': s_step,
                'test_accuracy': accuracy,
                'test_loss': test_result[0],
                'kernel1': kernel1,
                'stride1': stride1
            })

# En iyi sonucu yazdır
if results:
    best_result = max(results, key=lambda x: x['test_accuracy'])
    print("\n🎯 En iyi kombinasyon (Test doğruluğu >= 0.5):")
    for key, val in best_result.items():
        print(f"{key}: {val}")
else:
    print("\n😕 Hiçbir model 0.5 doğruluk eşiğini geçemedi.")
