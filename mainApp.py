import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import nltk
from nltk.stem import PorterStemmer, ISRIStemmer
import squarify  # for treemap

# Download resources
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("punkt_tab")   # <-- add this
# ----------------------
# Utility Functions
# ----------------------

porter = PorterStemmer()
arabic_stemmer = ISRIStemmer()

def preprocess_text_arabic(text):
    """Fix Arabic rendering for word cloud."""
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return bidi_text

def clean_and_stem(text):
    """Clean and apply stemming for Arabic + English"""
    if pd.isna(text):
        return ""
    text = re.sub(r"http\S+", " ", str(text))   # remove links
    text = re.sub(r"[^A-Za-zأ-ي\s]", " ", text) # keep only Arabic/English
    text = re.sub(r"\s+", " ", text)            # normalize spaces

    words = nltk.word_tokenize(text)
    stemmed_words = []
    for w in words:
        if re.search(r"[أ-ي]", w):  # Arabic word
            try:
                stemmed_words.append(arabic_stemmer.stem(w))
            except:
                stemmed_words.append(w)
        else:  # English word
            stemmed_words.append(porter.stem(w))
    return " ".join(stemmed_words)

def generate_wordcloud(text, title="Word Cloud"):
    """Generate word cloud after Arabic reshaping"""
    if not text.strip():
        return None
    text = preprocess_text_arabic(text)
    wordcloud = WordCloud(
        width=800, height=400,
        background_color="white",
        font_path="arial.ttf"  # use Arabic-compatible font if needed
    ).generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=16)
    return fig

def topic_modeling(docs, n_topics=3, n_top_words=8):
    """Run LDA topic modeling and return top words per topic"""
    if not docs or len(docs) < 5:
        return None

    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
    X = vectorizer.fit_transform(docs)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    try:
        lda.fit(X)
    except ValueError:
        return None

    words = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append({"Topic": f"Topic {idx+1}", "Top Words": ", ".join(top_words)})

    return pd.DataFrame(topics)

def plot_word_frequency_horizontal(series, title):
    """Plot horizontal bar chart for word frequency with labels"""
    counts = series.value_counts(normalize=True) * 100
    if counts.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    counts.head(10).plot(kind="barh", ax=ax, color="skyblue", edgecolor="black")
    ax.set_xlabel("Percentage (%)")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=14)
    plt.gca().invert_yaxis()  # highest value on top

    # Add percentage labels on bars
    for i, v in enumerate(counts.head(10).values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center")

    return fig

def plot_pie_chart(series, title):
    """Plot pie chart with percentages"""
    counts = series.value_counts()
    if counts.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=plt.cm.Paired.colors,
        textprops={"fontsize": 10}
    )
    ax.set_title(title, fontsize=14)
    return fig

# ----------------------
# Streamlit App
# ----------------------
st.set_page_config(page_title="Transcript Analyzer", layout="wide")
st.title("🌍 Transcript Analyzer with Stemming (Loc → Label → Topics)")

@st.cache_data
def load_data(file):
    return pd.read_excel(file)

uploaded_file = 'youtubeData.xlsx'  # change to st.file_uploader if needed

if uploaded_file:
    df = load_data(uploaded_file)

    # --- Pie chart for locations ---
    fig_loc_pie = plot_pie_chart(df["loc"].dropna(), "📊 Location Distribution")
    if fig_loc_pie:
        st.pyplot(fig_loc_pie)

    # Loop through each location
    locs = df["loc"].dropna().unique().tolist()

    for loc in locs:
        st.markdown(f"## 📍 Location: {loc}")
        loc_df = df[df["loc"] == loc]

        # --- Pie chart for labels in this location ---
        fig_label_pie = plot_pie_chart(loc_df["label"].dropna(), f"🏷 Label Distribution in {loc}")
        if fig_label_pie:
            st.pyplot(fig_label_pie)

        labels = loc_df["label"].dropna().unique().tolist()
        for label in labels:
            st.markdown(f"### 🏷 Label: {label}")
            label_df = loc_df[loc_df["label"] == label]

            # Apply stemming
            docs = [clean_and_stem(t) for t in label_df["Transcript"].dropna().astype(str).tolist()]
            text = " ".join(docs)

            # Word Cloud
            fig = generate_wordcloud(text, title=f"{loc} → {label}")
            if fig:
                st.pyplot(fig)

            # Topic Modeling
            topics_df = topic_modeling(docs, n_topics=5, n_top_words=15)
            if topics_df is not None:
                st.write("📑 Topics found:")
                st.dataframe(topics_df)
            else:
                st.warning("Not enough text for topic modeling in this group.")

            # Word Frequency for categorical columns
            for col in ["business_model", "target_segment", "kind_of_business"]:
                st.markdown(f"#### 📊 {col.replace('_', ' ').title()} Frequency")

                series = label_df[col].dropna()

                # Horizontal Bar Chart
                fig_barh = plot_word_frequency_horizontal(series, f"{col.replace('_', ' ').title()} Frequency")
                if fig_barh:
                    st.pyplot(fig_barh)
                else:
                    st.warning(f"No data available for {col}.")

