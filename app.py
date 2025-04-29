# app.py

import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import emoji
import nltk
nltk.download('stopwords')

st.set_page_config(layout="wide")
st.title("📊 WhatsApp Chat Analyzer with NLP")

# File uploader
uploaded_file = st.file_uploader("Upload WhatsApp Chat (.txt)", type="txt")

if uploaded_file:
    raw_data = uploaded_file.read().decode("utf-8")

    # ----- Parsing -----
    pattern = r"(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2} (?:AM|PM)) - (.*?): (.+)"
    matches = re.findall(pattern, raw_data)
    df = pd.DataFrame(matches, columns=['Date', 'Time', 'User', 'Message'])
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %I:%M %p', errors='coerce')
    df.dropna(subset=['Datetime'], inplace=True)

    # Cleanup
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    stop_words = set(stopwords.words('english'))

    # ----- Sidebar Options -----
    st.sidebar.header("📌 Filters")
    users = df['User'].unique().tolist()
    selected_user = st.sidebar.selectbox("Select user (or All)", ["All"] + users)

    if selected_user != "All":
        df = df[df['User'] == selected_user]

    # ----- Stats -----
    st.subheader("🔢 Basic Stats")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Messages", len(df))
    with col2:
        st.metric("Unique Users", df['User'].nunique())
    with col3:
        st.metric("Total Words", df['Message'].apply(lambda x: len(x.split())).sum())

    # ----- Word Cloud -----
    st.subheader("☁️ Word Cloud")
    text = ' '.join(df['Message'].dropna())
    words = ' '.join([word for word in text.lower().split() if word not in stop_words])
    wc = WordCloud(width=800, height=400, background_color='white').generate(words)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # ----- Emoji Analysis -----
    st.subheader("😄 Emoji Usage")
    df['Emojis'] = df['Message'].apply(lambda msg: ''.join(c for c in msg if c in emoji.EMOJI_DATA))
    all_emojis = ''.join(df['Emojis'].tolist())
    emoji_freq = Counter(all_emojis)
    top_emojis = pd.DataFrame(emoji_freq.most_common(10), columns=['Emoji', 'Count'])

    st.dataframe(top_emojis)
    fig, ax = plt.subplots()
    sns.barplot(data=top_emojis, x='Emoji', y='Count', ax=ax)
    st.pyplot(fig)

    # ----- Sentiment Analysis -----
    st.subheader("🧠 Sentiment Analysis")
    analyzer = SentimentIntensityAnalyzer()
    df['Sentiment'] = df['Message'].apply(lambda m: analyzer.polarity_scores(m)['compound'])
    df['Label'] = df['Sentiment'].apply(lambda s: 'Positive' if s > 0.05 else 'Negative' if s < -0.05 else 'Neutral')

    sentiment_count = df['Label'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
    st.bar_chart(sentiment_count)

    # ----- Activity Plots -----
    st.subheader("📆 Activity Over Time")
    df['Hour'] = df['Datetime'].dt.hour
    df['Day'] = df['Datetime'].dt.day_name()

    col1, col2 = st.columns(2)

    with col1:
        hourly = df['Hour'].value_counts().sort_index()
        st.line_chart(hourly.rename("Messages by Hour"))

    with col2:
        daily = df['Day'].value_counts().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        st.bar_chart(daily.rename("Messages by Day"))
