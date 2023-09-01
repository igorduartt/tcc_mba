import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
import spacy
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from textblob import TextBlob
from gensim.models import FastText

# Configurações iniciais
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
portuguese_stopwords = stopwords.words('portuguese')
nlp = spacy.load('pt_core_news_sm')
fasttext_model = FastText.load_fasttext_format('cc.pt.300.bin')

# Carregar dados
df = pd.read_excel('df.xlsx')

# Preparar dados
df = df.query('Quem == "Candidato"')
df['Texto_preprocessado'] = df['Texto'].apply(preprocess_text)
df['Entidades_nomeadas'] = df['Texto'].apply(get_named_entities)

# Definir tópicos
topics = ['educacao', 'saude', 'corrupcao', 'economia', 'emprego']

# Layout do aplicativo
st.set_page_config(page_title='Análise de Discursos de Candidatos', page_icon=':bar_chart:', layout='wide')
st.title('Análise de Discursos de Candidatos')

# Análise de tópicos
st.header('Análise de Tópicos')

# Mostrar texto pré-processado
if st.checkbox('Mostrar Texto Pré-processado'):
    st.write(df[['Nome', 'Texto_preprocessado']])

# Mostrar tópicos
if st.checkbox('Mostrar Tópicos'):
    topic_word_counts = count_topic_words(df, topics)
    for candidate in df['Nome'].unique():
        st.write(f"\n{candidate}:")
        for topic in topics:
            st.write(f"{topic}: {topic_word_counts[f'{candidate}_{topic}']}")

# Análise de sentimentos
st.header('Análise de Sentimentos')

# Mostrar scores de sentimento
if st.checkbox('Mostrar Scores de Sentimento'):
    plot_sentiment_scores(df)

# Análise de entidades nomeadas
st.header('Análise de Entidades Nomeadas')

# Mostrar entidades nomeadas
if st.checkbox('Mostrar Entidades Nomeadas'):
    show_filtered_entities = st.checkbox('Filtrar entidades', value=True)
    if show_filtered_entities:
        named_entities_counts_filtered = count_named_entities(df, filter_named_entities)
        top_named_entities_filtered = get_top_named_entities(named_entities_counts_filtered, top_n=10)
        for candidate in df['Nome'].unique():
            st.write(f"\n{candidate}:")
            for entity, count in top_named_entities_filtered[candidate]:
                st.write(f"{entity}: {count}")
else:
    named_entities_counts = count_named_entities(df)
    top_named_entities = get_top_named_entities(named_entities_counts, top_n=10)
    for candidate in df['Nome'].unique():
        st.write(f"\n{candidate}:")
        for entity, count in top_named_entities[candidate]:
            st.write(f"{entity}: {count}")
#Funções auxiliares

def preprocess_text(text):
    text = re.sub(r'[^a-zA-ZÀ-ú\s]', '', text)
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in portuguese_stopwords]
    lemmatized_words = [token.lemma_ for token in nlp(' '.join(words))]
    return 
    lemmatized_words

def get_named_entities(text):
    doc = nlp(text)
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]
    return 
    named_entities

def filter_named_entities(named_entities):
    ignore_list = ["Brasil", "R$", "Real", "reais", "Sabe", "Olha", "Cadê", "Tenho", "Sim", "Fui", "Bonner", "Vera"]
    filtered_entities = []
for entity, label in named_entities:
    if label != "PER" and entity not in ignore_list:
        filtered_entities.append((entity, label))
return 
        filtered_entities

def count_topic_words(df, topics, similarity_threshold=0.6):
    topic_word_counts = defaultdict(int)