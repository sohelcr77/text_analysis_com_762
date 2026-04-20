# 🛒 E-commerce Customer Behaviour & Sentiment Analysis (Streamlit App)

## 📌 Project Overview

This project presents an interactive web application built using Streamlit to analyse customer reviews from an e-commerce dataset. It applies Natural Language Processing (NLP) techniques to extract insights from textual data, focusing on sentiment analysis, word frequency patterns, and customer behaviour trends.

The application allows users to upload their own dataset or use a default sample dataset for analysis.

---

## 🚀 Features

* 📊 Sentiment Analysis (Positive, Negative, Neutral)
* ☁️ Word Cloud Visualization (Positive & Negative Reviews)
* ⭐ Rating Distribution Analysis
* 🔍 Sentiment vs Rating Comparison
* 📈 Review Trends Over Time
* 📋 Interactive Data Preview

---

## 🗂️ Dataset

* Source: Amazon Fine Food Reviews Dataset (Kaggle)
* Contains:

  * Review Text
  * Ratings (Score)
  * Timestamp
  * Product/User metadata

---

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash
git clone <your-repo-link>
cd <project-folder>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Application

```bash
streamlit run app.py
```

---

## 📦 Requirements

* streamlit
* pandas
* matplotlib
* seaborn
* wordcloud
* textblob

---

## 📊 How It Works

1. Text data is cleaned using regex.
2. Sentiment polarity is calculated using TextBlob.
3. Data is categorized into positive, negative, or neutral classes.
4. Visualizations are generated using Matplotlib and Seaborn.
5. Word clouds highlight frequently used terms in reviews.

---

## 📸 Output Visualizations

* Sentiment Distribution Bar Chart
* Ratings Distribution Chart
* Sentiment vs Rating Boxplot
* Review Trend Line Chart
* Positive & Negative Word Clouds

---

## ⚠️ Limitations

* Cannot detect sarcasm or contextual nuance
* Dependent on quality of input text
* Basic sentiment model (can be improved with ML models)

---

## 🔮 Future Improvements

* Implement TF-IDF + Machine Learning models
* Add real-time sentiment prediction API
* Improve UI/UX dashboard design
* Deploy on cloud platforms (Streamlit Cloud, AWS)

---

## 📚 References

1. Amazon Fine Food Reviews Dataset, Kaggle
2. Pang, B., & Lee, L. (2008). Opinion Mining and Sentiment Analysis

---

## 👨‍💻 Author

Developed as part of NLP & Data Analysis coursework project.
