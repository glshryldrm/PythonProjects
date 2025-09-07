from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import json

# Kaydedilen modeli yükle
model = load_model("song_lyrics_generator_model.h5")

# Tokenizer ve max_sequence_len'i yükle
with open('tokenizer.json') as f:
    tokenizer = tokenizer_from_json(json.load(f))

with open('max_sequence_len.txt') as f:
    max_sequence_len = int(f.read())

# Sıcaklık fonksiyonu
def apply_temperature(predictions, temperature=1.0):
    if temperature <= 0:
        raise ValueError("Temperature should be a positive value.")
    predictions = np.log(predictions + 1e-10) / temperature
    exp_preds = np.exp(predictions - np.max(predictions))
    predictions = exp_preds / np.sum(exp_preds)
    return predictions

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
            break
        predicted_word = tokenizer.index_word.get(predicted_word_index, "")
        if predicted_word == "":
            continue
        seed_text += " " + predicted_word
    return seed_text

# Örnek kullanım
seed_text = "aşk"
temperature = 0.8
print(generate_lyrics(seed_text, 20, model, tokenizer, max_sequence_len, temperature))
