!pip install pandas numpy matplotlib seaborn nltk spacy wordcloud gensim sklearn openpyxl

!python -m spacy download pt_core_news_sm

!pip install gensim

import spacy

nlp = spacy.load('pt_core_news_sm')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import spacy
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models
import openpyxl

df = pd.read_excel("df.xlsx")
df = df.query('Quem == "Candidato"')
df.head()

# Insira o código para análise exploratória aqui, como verificar valores ausentes, estatísticas descritivas, gráficos de frequência, etc.
# Exemplo: verificar valores ausentes
missing_values = df.isnull().sum()
missing_values.to_excel('missing_values.xlsx')

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from textblob import TextBlob

# Função para análise de sentimentos
def sentiment_analysis(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Função para plotar a análise de sentimentos
def plot_sentiment_analysis(candidate_sentiments):
    fig, ax = plt.subplots()
    ax.bar(candidate_sentiments.keys(), candidate_sentiments.values())
    ax.set_xlabel('Candidato')
    ax.set_ylabel('Sentimento')
    ax.set_title('Análise de Sentimentos por Candidato')
    plt.show()

# Função para modelagem de tópicos
def topic_modeling(df):
    count_vectorizer = CountVectorizer(stop_words='english')
    count_data = count_vectorizer.fit_transform(df['processed_text'])
    lda = LDA(n_components=5, random_state=42)
    lda.fit(count_data)
    return lda

# Função para exibir os tópicos
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

# Adicionar colunas ao dataframe para análise de sentimentos e entidades nomeadas
df['sentiment'] = df['Texto'].apply(sentiment_analysis)
df['named_entities'] = df['Texto'].apply(extract_named_entities)

# Criar um dicionário para armazenar a análise de sentimentos por candidato
candidate_sentiments = {'Jair Bolsonaro': 0,
                        'Ciro Gomes': 0,
                        'Lula': 0,
                        'Simone Tebet': 0,
                        'Soraya Thronicke': 0,
                        "Felipe D'Avila": 0,
                        }

# Percorrer todas as falas do dataset
for index, row in df.iterrows():
    # Adicionar a análise de sentimentos ao dicionário
    candidate_sentiments[row['Nome']] += row['sentiment']

# Dividir os valores de sentimento pelo total de falas de cada candidato para obter a média
total_candidate1_rows = len(df[df['Nome'] == 'Jair Bolsonaro'])
total_candidate2_rows = len(df[df['Nome'] == 'Ciro Gomes'])
total_candidate3_rows = len(df[df['Nome'] == 'Lula'])
total_candidate4_rows = len(df[df['Nome'] == 'Simone Tebet'])
total_candidate5_rows = len(df[df['Nome'] == 'Soraya Thronicke'])
total_candidate6_rows = len(df[df['Nome'] == "Felipe D'Avila"])
candidate_sentiments['Jair Bolsonaro'] /= total_candidate1_rows
candidate_sentiments['Ciro Gomes'] /= total_candidate2_rows
candidate_sentiments['Lula'] /= total_candidate3_rows
candidate_sentiments['Simone Tebet'] /= total_candidate4_rows
candidate_sentiments['Soraya Thronicke'] /= total_candidate5_rows
candidate_sentiments["Felipe D'Avila"] /= total_candidate6_rows

# Plotar a análise de sentimentos
plot_sentiment_analysis(candidate_sentiments)

# Aplicar pré-processamento de texto e armazenar em uma nova coluna
df['processed_text'] = df['Texto'].apply(preprocess_text)

# Modelagem de tópicos e exibição dos tópicos
lda_model = topic_modeling(df)
print_topics(lda_model, count_vectorizer, 10)

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
portuguese_stopwords = stopwords.words('portuguese')

from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import gensim.downloader as api

# Carregar o modelo pré-treinado FastText em português
fasttext_model = api.load('fasttext-wiki-news-subwords-300-pt')

# Função para contar a frequência das palavras relacionadas aos tópicos
def count_topic_words(dataset, topics, similarity_threshold=0.6):
    count_vectorizer = CountVectorizer(stop_words=portuguese_stopwords)
    count_data = count_vectorizer.fit_transform(dataset['processed_text'])
    word2index = {word: idx for idx, word in enumerate(count_vectorizer.get_feature_names_out())}
    
    topic_word_counts = defaultdict(int)
    
    for candidate in dataset['Nome'].unique():
        candidate_rows = dataset[dataset['Nome'] == candidate]
        candidate_count_data = count_vectorizer.transform(candidate_rows['processed_text'])
        
        for topic in topics:
            if topic not in fasttext_model.key_to_index:
                continue

            topic_vector = fasttext_model[topic]
            for word, idx in word2index.items():
                if word not in fasttext_model.key_to_index:
                    continue

                word_vector = fasttext_model[word]
                similarity = np.dot(topic_vector, word_vector) / (np.linalg.norm(topic_vector) * np.linalg.norm(word_vector))

                if similarity >= similarity_threshold:
                    topic_word_counts[f"{candidate}_{topic}"] += candidate_count_data[:, idx].sum()
    
    return topic_word_counts

# Definir os tópicos e palavras relacionadas
topics = ['educação', 'saúde', 'corrupção', 'economia', 'emprego']

# Contar a frequência das palavras relacionadas aos tópicos
topic_word_counts = count_topic_words(dataset, topics)

# Imprimir os resultados
for candidate in dataset['Nome'].unique():
    print(f"\n{candidate}:")
    for topic in topics:
        print(f"{topic}: {topic_word_counts[f'{candidate}_{topic}']}")