import re
import joblib
import nltk
import numpy as np
import scipy.sparse as sp
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Konfigurasi Halaman Web
st.set_page_config(
    page_title="IMDB Sentiment Detector", page_icon="🎬", layout="centered"
)


# Load dependensi dan model ke dalam memori cache agar web super cepat
@st.cache_resource
def load_models():
  nltk.download("stopwords", quiet=True)
  nltk.download("wordnet", quiet=True)

  tfidf = joblib.load("tfidf_vectorizer.pkl")
  model = joblib.load("linearsvc_model.pkl")
  analyzer = SentimentIntensityAnalyzer()
  stop_w = set(stopwords.words("english"))
  lemma = WordNetLemmatizer()

  return tfidf, model, analyzer, stop_w, lemma


tfidf, model, analyzer, STOP_WORDS, lemmatizer = load_models()


def preprocess_text(text):
  text = text.lower()
  text = re.sub(r"<.*?>", " ", text)
  text = re.sub(r"[^a-z\s]", "", text)
  tokens = text.split()
  tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS]
  return " ".join(tokens)


# Tampilan Antarmuka
st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.markdown("""
Aplikasi web ini menguji hasil riset NLP untuk mengklasifikasikan ulasan film berbahasa Inggris menggunakan kombinasi model **TF-IDF + LinearSVC** (dilengkapi *VADER Lexicon Score Stacking*).  
Dataset acuan: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
""")

st.divider()

st.subheader("📝 Masukkan Teks Ulasan Film")
user_input = st.text_area(
    "Ketikkan ulasan dalam Bahasa Inggris di bawah ini:",
    height=150,
    placeholder="Contoh: The movie was absolutely breathtaking! The plot twists kept me on the edge of my seat...",
)

if st.button("🔍 Analisis Sentimen", type="primary"):
  if not user_input.strip():
    st.warning("⚠️ Teks ulasan tidak boleh kosong.")
  else:
    with st.spinner("Menganalisis pola teks..."):
      # Pemrosesan teks input
      cleaned = preprocess_text(user_input)
      x_tfidf = tfidf.transform([cleaned])
      vader_comp = analyzer.polarity_scores(cleaned)["compound"]
      x_input = sp.hstack((x_tfidf, np.array([[vader_comp]])))

      # Prediksi & Konversi Jarak Margin ke Persentase Confidence
      pred = model.predict(x_input)[0]
      margin = model.decision_function(x_input)[0]
      confidence = 1 / (1 + np.exp(-margin))
      if pred == 0:
        confidence = 1 - confidence

    st.divider()
    st.subheader("📊 Hasil Deteksi")

    col1, col2 = st.columns(2)
    with col1:
      if pred == 1:
        st.success("### 🟢 **POSITIF** (Positive)")
        st.write("Penulis menyukai film ini.")
      else:
        st.error("### 🔴 **NEGATIF** (Negative)")
        st.write("Penulis mengkritik/tidak menyukai film ini.")

    with col2:
      st.metric("Tingkat Keyakinan Model", f"{confidence * 100:.2f}%")
      st.progress(float(confidence))

    with st.expander("⚙️ Rincian Teknis untuk Penguji"):
      st.write(f"**Teks Bersih**: `{cleaned}`")
      st.write(f"**Skor Compound VADER**: `{vader_comp}`")
      st.write(f"**Nilai Decision Function**: `{margin:.4f}`")