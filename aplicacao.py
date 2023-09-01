import streamlit as st
import pandas as pd
import re
from collections import defaultdict, Counter
from textblob import TextBlob
import spacy
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Inicializar o modelo spaCy para reconhecimento de entidades nomeadas
nlp = spacy.load('pt_core_news_sm')

# Stopwords em português
from spacy.lang.pt.stop_words import STOP_WORDS as stopwords

# Tópicos e palavras-chave relacionadas
topics = {
    'educacao': ['educação', 'escola', 'universidade', 'professor', 'ensino'],
    'saude': ['saúde', 'hospital', 'médico', 'remédio', 'paciente'],
    'seguranca': ['segurança', 'polícia', 'crime', 'violência', 'prisão'],
    'corrupcao': ['corrupção', 'corrupto', 'lavagem', 'propina', 'desvio'],
    'economia': ['economia', 'imposto', 'tributo', 'inflação', 'crescimento']
}

def preprocess_data(text):
    return re.sub(r'[^a-zA-ZÀ-ú\s]', '', text).lower()

def count_mentions(text):
    mentions = defaultdict(int)
    for topic, keywords in topics.items():
        for keyword in keywords:
            mentions[topic] += text.count(keyword)
    return mentions

def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positivo'
    elif analysis.sentiment.polarity == 0:
        return 'Neutro'
    else:
        return 'Negativo'

def extract_entities(text):
    doc = nlp(text)
    excluded_entities = ["Brasil", "Bonner", "R$", "Renata", "Sabe", "Olha", "Vera"]
    return [(ent.text, ent.label_) for ent in doc.ents if ent.text not in excluded_entities]

def plot_results(candidate_data, candidate_name):
    topics_list = list(candidate_data.keys())
    mentions = list(candidate_data.values())
    fig = px.bar(x=topics_list, y=mentions, labels={'x':'Tópicos', 'y':'Número de Menções'},
                 title=f'Tópicos mencionados por {candidate_name}', color=topics_list,
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig)

def generate_wordcloud(text):
    wordcloud = WordCloud(stopwords=stopwords, background_color="white", width=800, height=400).generate(text)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis('off')
    st.pyplot(fig)

def main():
    st.title("Análise de discursos dos candidatos à presidência do Brasil (2022)")

    uploaded_file = st.file_uploader("Carregue o dataset", type=['xlsx'])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        # Preprocessamento e análise dos dados
        df['Texto_preprocessed'] = df['Texto'].apply(preprocess_data)
        df['topic_mentions'] = df['Texto_preprocessed'].apply(count_mentions)
        df['sentiment'] = df['Texto'].apply(analyze_sentiment)
        df['entities'] = df['Texto'].apply(extract_entities)

        # Contagem de ocorrências para cada candidato
        topic_counts = defaultdict(lambda: defaultdict(int))
        entity_counts = defaultdict(Counter)
        sentiment_counts = defaultdict(lambda: defaultdict(int))
        
        for index, row in df.iterrows():
            for topic, count in row['topic_mentions'].items():
                topic_counts[row['Nome']][topic] += count
            for entity, _ in row['entities']:
                entity_counts[row['Nome']][entity] += 1
            sentiment_counts[row['Nome']][row['sentiment']] += 1

        # Mostrando a análise
        candidate = st.selectbox("Escolha um candidato", list(topic_counts.keys()))

        # Seção de resumo
        st.subheader('Resumo')
        with st.container():
            st.markdown(f"**Candidato(a) selecionado(a):** {candidate}")
            st.markdown(f"**Total de menções a tópicos:** {sum(topic_counts[candidate].values())}")
            st.markdown(f"**Sentimento predominante:** {max(sentiment_counts[candidate], key=sentiment_counts[candidate].get)}")
        
        # Análise dos tópicos
        with st.expander("Análise dos tópicos mais citados"):
            plot_results(topic_counts[candidate], candidate)

        # Análise de sentimentos
        with st.expander("Análise de sentimentos"):
            sentiments = list(sentiment_counts[candidate].keys())
            sentiment_values = list(sentiment_counts[candidate].values())
            fig = px.bar(x=sentiments, y=sentiment_values, labels={'x':'Sentimento', 'y':'Número de ocorrências'},
                     title=f'Distribuição de sentimentos por {candidate}', color=sentiments,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig)

        # Entidades nomeadas
        with st.expander("Entidades nomeadas"):
            top_entities = dict(sorted(entity_counts[candidate].items(), key=lambda item: item[1], reverse=True)[:10])
            entities_list = list(top_entities.keys())
            entity_counts_list = list(top_entities.values())
            fig = px.bar(y=entities_list, x=entity_counts_list, orientation='h', labels={'y':'Entidades', 'x':'Número de menções'},
                     title=f'Tópicos mais abordados pelo(a) candidato(a): {candidate}', color=entities_list,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig)

        # Word Cloud
        with st.expander("Word Cloud"):
            all_text = ' '.join(df[df['Nome'] == candidate]['Texto_preprocessed'].tolist())
            generate_wordcloud(all_text)

if __name__ == '__main__':
    main()
