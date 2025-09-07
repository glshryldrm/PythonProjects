import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import nltk
from nltk.corpus import stopwords
#%%
# Veri yükleme
data = pd.read_csv('turkish_song_lyrics.csv') #https://www.kaggle.com/datasets/emreokcular/turkish-song-lyrics
print(data.head())
print(data.isnull().sum())  # Eksik veri kontrolü
#%%
# Gereksiz sütunları kaldırma
data = data.drop(columns=['album', 'singer', 'song'])
data = data.sample(n=1000, random_state=42)  # Sabit bir random_state kullanarak tekrarlanabilirlik sağlanır
data['lyrics_cleaned'] = data['lyrics'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
data = data.dropna(subset=['lyrics_cleaned'])
data = data[data['lyrics_cleaned'].str.strip() != '']  # Boş satırları kaldır

#%%
# Tokenizer işlemi
tokenizer = Tokenizer(num_words=5000)  # Kelime sayısını sınırlayın
tokenizer.fit_on_texts(data['lyrics_cleaned'])
#%%
# Metni dizilere dönüştürme
input_sequences = []
for line in data['lyrics_cleaned']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i+1])


nltk.download('stopwords')  # Stopwords listesini indirir

# Türkçe stopwords listesini al
stop_words = set(stopwords.words('turkish'))  # İngilizce için 'english' yazabilirsiniz

def remove_stopwords(text):
    if isinstance(text, str):  # Eğer metin bir string ise
        words = text.split()  # Metni kelimelere ayır
        filtered_words = [word for word in words if word not in stop_words]  # Stopword'leri çıkar
        return " ".join(filtered_words)  # Kelimeleri tekrar birleştir
    return text  # Eğer metin değilse olduğu gibi döndür


data['lyrics_cleaned'] = data['lyrics_cleaned'].apply(remove_stopwords)

#%%
# Pad sequences
max_sequence_len = max([len(seq) for seq in input_sequences])  # Maksimum uzunluk belirleme
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

X = input_sequences[:, :-1]  # Tüm satırların son eleman hariç kısmı
y = input_sequences[:, -1]   # Tüm satırların sadece son elemanı
from sklearn.model_selection import train_test_split

# Eğitim ve doğrulama verilerini ayırma
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#%%

total_words = len(tokenizer.word_index) + 1  # Tokenizer'ın tüm kelimelerini alın


from tensorflow.keras.layers import SimpleRNN

model = Sequential([
    Embedding(input_dim=total_words, output_dim=128),
    SimpleRNN(128, return_sequences=True),  # İlk RNN katmanı
    SimpleRNN(64),                          # İkinci RNN katmanı
    Dense(total_words, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Callbacks
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)

from sklearn.model_selection import train_test_split

# Eğitim, doğrulama ve test setine ayırma
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)  # %10 Test seti
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)  # %20 Doğrulama

print(f"Eğitim seti boyutu: {len(X_train)}")
print(f"Doğrulama seti boyutu: {len(X_val)}")
print(f"Test seti boyutu: {len(X_test)}")

# Modeli eğitme
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),  # Doğrulama verisi
    epochs=20,
    batch_size=256,
    callbacks=[early_stopping, reduce_lr]
)
model.save("song_lyrics_generator_model.h5")
# Test setinde değerlendirme
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

#%%
# Eğitim ve doğrulama kaybını inceleme
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Sonuçları yazdırma
print(f"Training Loss: {train_loss}")
print(f"Validation Loss: {val_loss}")


#%%
# Sıcaklık fonksiyonu
def apply_temperature(predictions, temperature=1.0):
    if temperature <= 0:
        raise ValueError("Temperature should be a positive value.")
    predictions = np.log(predictions + 1e-10) / temperature
    exp_preds = np.exp(predictions - np.max(predictions))  # Stabilizasyon için maksimum çıkar
    predictions = exp_preds / np.sum(exp_preds)  # Normalize
    return predictions

#%%
# Şarkı üretme fonksiyonu
def generate_lyrics(seed_text, next_words, model, tokenizer, max_sequence_len, temperature=1.0):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)[0]
        predicted = apply_temperature(predicted, temperature)
        try:
            predicted_word_index = np.random.choice(range(len(predicted)), p=predicted)
        except ValueError:
            break  # Olası hata durumunda döngüden çık
        predicted_word = tokenizer.index_word.get(predicted_word_index, "")
        if predicted_word == "":
            continue  # Bulunamayan kelimeleri atla
        seed_text += " " + predicted_word
    return seed_text

#%%
# Şarkı üretimi
seed_text = "aşk"
temperature = 1.0  # Daha yaratıcı olması için uygun bir sıcaklık
print(generate_lyrics(seed_text, 20, model, tokenizer, max_sequence_len, temperature))