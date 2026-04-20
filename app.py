import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob

st.set_page_config(page_title="E-commerce Sentiment Analysis", layout="wide")

st.title("🛒 E-commerce Customer Behaviour & Sentiment Analysis")

# ==============================
# LOAD DATA
# ==============================
uploaded_file = st.file_uploader("Upload Reviews CSV", type=["csv"])

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file:
    df = load_data(uploaded_file)
else:
    df = pd.read_csv("Reviews.csv", nrows=20000)

# ==============================
# CLEAN TEXT
# ==============================
def clean_text(text):
    return re.sub(r'[^a-zA-Z ]', '', str(text)).lower()

df['clean_text'] = df['Text'].apply(clean_text)

# ==============================
# TIME PROCESSING
# ==============================
df['Time'] = pd.to_datetime(df['Time'], unit='s')
df['Year'] = df['Time'].dt.year


# ==============================
# SAFE TIME HANDLING
# ==============================
if 'Time' in df.columns:
    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    df['Year'] = df['Time'].dt.year
else:
    st.warning("⚠️ 'Time' column not found. Time-based analysis disabled.")
    df['Year'] = 0  # fallback
# ==============================
# 🎯 INTERACTIVE FILTERS
# ==============================

if 'Year' in df.columns and df['Year'].nunique() > 1:
    year_range = st.sidebar.slider(
        "Select Year Range",
        int(df['Year'].min()),
        int(df['Year'].max()),
        (int(df['Year'].min()), int(df['Year'].max()))
    )
else:
    year_range = (0, 9999)


st.sidebar.header("🔎 Filters")

# Year filter
year_range = st.sidebar.slider(
    "Select Year Range",
    int(df['Year'].min()),
    int(df['Year'].max()),
    (int(df['Year'].min()), int(df['Year'].max()))
)

# Rating filter
rating_range = st.sidebar.slider(
    "Select Rating Range",
    int(df['Score'].min()),
    int(df['Score'].max()),
    (1, 5)
)

# Apply filters
filtered_df = df[
    (df['Year'].between(year_range[0], year_range[1])) &
    (df['Score'].between(rating_range[0], rating_range[1]))
]

# ==============================
# SENTIMENT
# ==============================
filtered_df['sentiment'] = filtered_df['clean_text'].apply(
    lambda x: TextBlob(x).sentiment.polarity
)

def label_sentiment(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'neutral'

filtered_df['label'] = filtered_df['sentiment'].apply(label_sentiment)

# ==============================
# VISUALS
# ==============================

# Sentiment Distribution
st.subheader("📊 Sentiment Distribution")
fig1, ax1 = plt.subplots()
filtered_df['label'].value_counts().plot(kind='bar', ax=ax1)
st.pyplot(fig1)

# Ratings Distribution
st.subheader("⭐ Ratings Distribution")
fig2, ax2 = plt.subplots()
filtered_df['Score'].value_counts().sort_index().plot(kind='bar', ax=ax2)
st.pyplot(fig2)

# Sentiment vs Rating
st.subheader("🔍 Sentiment vs Rating")
fig3, ax3 = plt.subplots()
sns.boxplot(x='Score', y='sentiment', data=filtered_df, ax=ax3)
st.pyplot(fig3)

# ==============================
# WORD CLOUD
# ==============================
st.subheader("☁️ Word Cloud")

text = " ".join(filtered_df['clean_text'])
wc = WordCloud(stopwords=set(STOPWORDS),
               background_color="white").generate(text)

fig4, ax4 = plt.subplots()
ax4.imshow(wc)
ax4.axis("off")
st.pyplot(fig4)

# ==============================
# 🔴 REAL-TIME SENTIMENT INPUT
# ==============================
st.subheader("🧠 Real-Time Sentiment Prediction")

user_input = st.text_area("Enter a review:")

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        polarity = TextBlob(user_input).sentiment.polarity
        
        st.write(f"Sentiment Score: {round(polarity,3)}")
        
        if polarity > 0:
            st.success("✅ Positive Review")
        elif polarity < 0:
            st.error("⚠️ Negative Review")
        else:
            st.info("😐 Neutral Review")

# ==============================
# DATA PREVIEW
# ==============================
st.subheader("📋 Filtered Data Preview")
st.dataframe(filtered_df.head())
