import streamlit as st
import pandas as pd
import json
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from collections import Counter
import re

@st.cache_resource
def load_model_and_tokenizer():
    model = load_model('ta_model_revise_3.keras')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

@st.cache_data
def load_data():
    with open('dataset.json') as json_file:
        data = json.load(json_file)
    with open('datatest.json') as json_file2:
        data2 = json.load(json_file2)
    return data, data2

def label_prediction(prediksi):
    if prediksi < 0.5:
        return 'Baik-baik saja'
    elif prediksi < 0.9:
        return 'Kemungkinan butuh pertolongan'
    else:
        return 'Membutuhkan pertolongan'

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return words

def get_keyword_frequencies(texts, keywords):
    keyword_counts = Counter()
    for text in texts:
        words = preprocess_text(text)
        for word in words:
            if word in keywords:
                keyword_counts[word] += 1
    return keyword_counts.most_common()

def adjust_prediction_for_positive_words(prediction, text, positive_keywords):
    words = preprocess_text(text)
    positive_count = sum(1 for word in words if word in positive_keywords)
    # Adjust the prediction based on the number of positive words found
    if positive_count > 0:
        adjustment = positive_count * 0.02  # Example adjustment factor
        prediction = max(0, prediction - adjustment)
    return prediction

mental_health_keywords = [
    "anxiety", "disorder", "personality", "deficit", "hyperactivity", "adhd",
    "posttraumatic", "stress", "ptsd", "depression", "bipolar",
    "obsessive", "compulsive", "ocd", "autism", "spectrum",
    "schizophrenia", "schizo", "psychotic", "psychosis", "anorexia nervosa",
    "anorexia", "nervosa", "bulimia nervosa", "bulimia", "borderline",
    "bpd", "substance", "sud", "addiction", "drug", "alcoholism", "panic",
    "gad", "dissociative", "suicide", "suicidal", "threat", "die", "death",
    "selfharm", "possessions", "kill", "crisis", "useless", "helplessness",
    "guilt", "blame", "overwhelmed", "shame", "abandon", "forsake", "reject",
    "desperation", "anguish", "agony", "sacrifice", "void", "lost", "tired",
    "anxious", "pretending", "stigma", "loneliness", "pointless",
    "worthlessness", "hallucinations", "confusion", "paranoia", "neglect",
    "distrustful", "unworthy", "isolation", "difficult", "afraid", "fear",
    "mistake", "nothing", "underappreciate", "problem", "angry", "pressure",
    "overthinking", "doubt", "disappoint", "fail", "alone", "hate", "insult",
    "meaningless", "suffer", "stupid", "struggle", "abuse", "unhelpful",
    "emotional", "trauma", "injury", "inadequacy", "dead", "unhealthy",
    "betray", "negative", "issue", "tear", "intrusive", "hurt", 
    "uncomfortable", "cry", "terrible", "depressive",
]

positive_keywords = [
    "happy", "joy", "love", "excellent", "good", "great", "wonderful", 
    "awesome", "fantastic", "positive", "smile", "success", "achievement",
    "nice", "beautiful", "happy", "joyful", "satisfied", "happiness", "fine",
    "amazing", "brilliant", "cheerful", "delight", "ecstatic", "enjoy",
    "fabulous", "fortunate", "glad", "glorious", "hopeful", "incredible",
    "jubilant", "lucky", "marvelous", "outstanding", "perfect", "pleasant",
    "radiant", "remarkable", "spectacular", "splendid", "terrific", "thrilled",
    "triumphant", "uplifting", "wonder", "enthusiastic", "energetic",
    "lovely", "blissful", "grateful", "graceful", "kind", "benevolent",
    "compassionate", "courteous", "euphoric", "gorgeous", "heartwarming",
    "inspired", "magnificent", "refreshing",
    "remarkable", "respectful", "sensational", "stunning", "supportive",
    "sweet", "sympathetic", "tender", "thankful", "thoughtful", "vibrant",
    "warm", "welcoming", "accomplished", "affectionate",
    "blessed", "breathtaking", "charming",
    "empowered", "encouraged", "exhilarated", "faithful", "generous",
    "gentle", "honest", "hope", "impressive", "lively", "motivated",
    "optimistic", "peaceful", "playful", "proud", "thankful", "love", "loved"
]

model, tokenizer = load_model_and_tokenizer()
data, data2 = load_data()
# model.summary()

# MAIN CODE
st.header("APLIKASI IDENTIFIKASI ANOMALI KESEHATAN MENTAL PADA KOMENTAR")
st.markdown("**Developed By Salwa Ahmad Zanjabila | Informatika Universitas Pembangunan Jaya 2020**")
st.write("")
st.subheader("Dataset Komentar Dari YouTube")
comments = [comment['Preprocess'] for comment in data['data']]
dataset = pd.DataFrame({'Komentar': comments})
st.write(dataset)

st.subheader("Kata Kunci Yang Digunakan Pada Aplikasi")
col1, col2 = st.columns(2)
keyword_list_df_negatif = pd.DataFrame(mental_health_keywords, columns=['Kata Kunci'])
keyword_list_df_positif = pd.DataFrame(positive_keywords, columns=['Kata Kunci'])

with col1:    
    st.write("Kata Kunci Negatif")
    st.write(keyword_list_df_negatif)
with col2:
    st.write("Kata Kunci Positif")
    st.write(keyword_list_df_positif)

st.subheader("Data Test Komentar Dari YouTube")
comments2 = [comment2['Comments'] for comment2 in data2['data']]
datatest = pd.DataFrame({'Komentar': comments2})
st.write(datatest)

st.subheader("Kata Kunci Yang Paling Sering Muncul")
negatif_keywords_frequencies = get_keyword_frequencies(datatest['Komentar'], mental_health_keywords)
negatif_keywords_frequencies_df = pd.DataFrame(negatif_keywords_frequencies, columns=['Kata Kunci', 'Frekuensi'])

positif_keywords_frequencies = get_keyword_frequencies(datatest['Komentar'], positive_keywords)
positif_keywords_frequencies_df = pd.DataFrame(positif_keywords_frequencies, columns=['Kata Kunci', 'Frekuensi'])

col1, col2 = st.columns(2)
with col1:
    st.write(negatif_keywords_frequencies_df)
with col2:
    st.write(positif_keywords_frequencies_df)

if 'predictions' not in st.session_state:
    sequence = tokenizer.texts_to_sequences(datatest['Komentar'])
    padded = pad_sequences(sequence, maxlen=537)
    predictions = model.predict(padded)
    st.session_state['predictions'] = predictions
else:
    predictions = st.session_state['predictions']

df_predictions = pd.DataFrame({'Komentar': comments2, 'Label': predictions.flatten(), 'Prediksi': predictions.flatten()})

# Adjust predictions based on positive keywords
df_predictions['Prediksi'] = df_predictions.apply(
    lambda row: adjust_prediction_for_positive_words(row['Prediksi'], row['Komentar'], positive_keywords), axis=1
)

# PREDIKSI KOMENTAR
df_predictions['Label'] = df_predictions['Prediksi'].apply(label_prediction)
df_predictions['Prediksi (%)'] = df_predictions['Prediksi'] * 100
st.subheader("Hasil Identifikasi Data Test Komentar")
st.write(df_predictions)

st.subheader("Chart Hasil Identifikasi Berdasarkan Label")
label_counts = df_predictions.groupby('Label').size().reset_index(name='Jumlah')
st.bar_chart(label_counts.set_index('Label'))
st.write(label_counts)

st.subheader("Masukan Komentar Yang Ingin Di Identifikasi (Menggunakan Bahasa Inggris)")
with st.form(key='form_predict'):
    text_input = st.text_input(label='Masukkan Komentar')
    submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        # Preprocess the input text
        input_sequence = tokenizer.texts_to_sequences([text_input])
        input_padded = pad_sequences(input_sequence, maxlen=537)

        # Predict the input text
        input_prediction = model.predict(input_padded)[0][0]
        
        # Adjust prediction for positive words
        st.write(f"Komentar: {text_input}")
        # st.write(f"Prediksi Awal: {input_prediction:.2f}%")
        input_prediction = adjust_prediction_for_positive_words(input_prediction, text_input, positive_keywords)
        
        input_label = label_prediction(input_prediction)
        input_prediction_percent = input_prediction * 100

        # Display the result
        st.write(f"Hasil Prediksi: {input_prediction_percent:.2f}%")
        st.write(f"Label: {input_label}")

        # Display keyword frequencies in the input text
        col1, col2 = st.columns(2)
        with col1:
            input_keywords = get_keyword_frequencies([text_input], mental_health_keywords)
            st.write("Kata Kunci negatif dalam Komentar yang Dimasukkan:")
            st.write(pd.DataFrame(input_keywords, columns=['Kata Kunci', 'Frekuensi']))
        with col2:
            input_keywords = get_keyword_frequencies([text_input], positive_keywords)
            st.write("Kata Kunci positif dalam Komentar yang Dimasukkan:")
            st.write(pd.DataFrame(input_keywords, columns=['Kata Kunci', 'Frekuensi']))
