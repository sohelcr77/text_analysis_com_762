import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob

st.set_page_config(page_title="E-commerce Sentiment Analysis", layout="wide")

# =========================================================
# 📌 TITLE
# =========================================================
st.title("🛒 E-commerce Customer Behaviour & Sentiment Analysis")

st.markdown("""
This app analyses customer reviews using NLP techniques including sentiment analysis,
word clouds, and behavioural trends.
""")

# =========================================================
# 📂 FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader("Upload Reviews CSV (or use sample)", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

if uploaded_file:
    df = load_data(uploaded_file)
else:
    st.warning("Using sample dataset (first 10k rows)")
    df = pd.read_csv("Reviews.csv", nrows=10000)

# =========================================================
# 🧹 TEXT CLEANING
# =========================================================
def clean_text(text):
    return re.sub(r'[^a-zA-Z ]', '', str(text)).lower()

df['clean_text'] = df['Text'].apply(clean_text)

# =========================================================
# 😊 SENTIMENT ANALYSIS
# =========================================================
df['sentiment'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

def label_sentiment(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'neutral'

df['label'] = df['sentiment'].apply(label_sentiment)

# =========================================================
# 📊 SENTIMENT DISTRIBUTION
# =========================================================
st.subheader("📊 Sentiment Distribution")
fig1, ax1 = plt.subplots()
df['label'].value_counts().plot(kind='bar', ax=ax1)
st.pyplot(fig1)

# =========================================================
# ⭐ RATINGS PATTERN
# =========================================================
st.subheader("⭐ Ratings Distribution")
fig2, ax2 = plt.subplots()
df['Score'].value_counts().sort_index().plot(kind='bar', ax=ax2)
st.pyplot(fig2)

# =========================================================
# 🔍 SENTIMENT vs RATING
# =========================================================
st.subheader("🔍 Sentiment vs Rating")
fig3, ax3 = plt.subplots()
sns.boxplot(x='Score', y='sentiment', data=df, ax=ax3)
st.pyplot(fig3)

# =========================================================
# 📈 TIME TREND
# =========================================================
st.subheader("📈 Reviews Over Time")

df['Time'] = pd.to_datetime(df['Time'], unit='s')
df['Year'] = df['Time'].dt.year

fig4, ax4 = plt.subplots()
df['Year'].value_counts().sort_index().plot(ax=ax4)
st.pyplot(fig4)

# =========================================================
# ☁️ WORD CLOUDS
# =========================================================
st.subheader("☁️ Word Clouds")

custom_stopwords = set(STOPWORDS).union({
    'br', 'product', 'one', 'like', 'taste', 'food'
})

col1, col2 = st.columns(2)

# Positive
with col1:
    st.markdown("### Positive Reviews")
    pos_text = " ".join(df[df['label'] == 'positive']['clean_text'])
    wc_pos = WordCloud(stopwords=custom_stopwords,
                       background_color="white").generate(pos_text)
    fig5, ax5 = plt.subplots()
    ax5.imshow(wc_pos)
    ax5.axis("off")
    st.pyplot(fig5)

# Negative
with col2:
    st.markdown("### Negative Reviews")
    neg_text = " ".join(df[df['label'] == 'negative']['clean_text'])
    wc_neg = WordCloud(stopwords=custom_stopwords,
                       background_color="white").generate(neg_text)
    fig6, ax6 = plt.subplots()
    ax6.imshow(wc_neg)
    ax6.axis("off")
    st.pyplot(fig6)

# =========================================================
# 📋 DATA PREVIEW
# =========================================================
st.subheader("📋 Sample Data")
st.dataframe(df.head())

# =========================================================
# ✅ FOOTER
# =========================================================
st.markdown("---")
st.markdown("Developed for NLP & Data Analysis Project")
