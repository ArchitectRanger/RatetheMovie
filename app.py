# -----------------------------
# 🎬 Movie Rating Prediction App with Tabs
# -----------------------------

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Movie Rating Predictor", page_icon="🎥", layout="wide")
st.title("🎬 Movie Rating Prediction App")
st.write("Predict average movie ratings using popularity, votes, and release year — powered by TMDB data.")

# -----------------------------
# API Config
# -----------------------------
API_KEY = "4194f61956693a7513861c1075ce59a4"
BASE_URL = "https://api.themoviedb.org/3"

@st.cache_data
def get_popular_movies(page=1):
    """Fetch popular movies from TMDB API."""
    url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&language=en-US&page={page}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["results"]
    else:
        st.error(f"Error fetching data: {response.status_code}")
        return []

@st.cache_data
def load_movie_data(pages=5):
    """Load and preprocess movie data."""
    all_movies = []
    for page in range(1, pages + 1):
        all_movies.extend(get_popular_movies(page))
    df = pd.DataFrame(all_movies)

    # Select useful columns
    df = df[["title", "popularity", "vote_average", "vote_count", "release_date", "original_language"]]
    df.dropna(inplace=True)

    # Convert release_date → year
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df.dropna(subset=["release_year"], inplace=True)

    # Map known languages, set all others to 'Other'
    language_map = {
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "ko": "Korean",
        "ja": "Japanese",
        "hi": "Hindi",
        "zh": "Chinese",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese"
    }
    df["original_language"] = df["original_language"].map(language_map).fillna("Other")

    # Encode languages
    df = pd.get_dummies(df, columns=["original_language"], drop_first=True)
    return df

# -----------------------------
# Automatically Load Data
# -----------------------------
df = load_movie_data()
if len(df) == 0:
    st.error("No data loaded. Check your API or mapping.")
    st.stop()

# -----------------------------
# Prepare Features
# -----------------------------
X = df[["popularity", "vote_count", "release_year"] +
       [col for col in df.columns if col.startswith("original_language_")]]
y = df["vote_average"]

if len(df) < 2:
    st.error("Not enough data to train the model.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📂 Data & Download", "📈 Visual Insights", "🎥 Predict Rating"])

# -----------------------------
# Tab 1: Data & Download
# -----------------------------
with tab1:
    st.header("Data Preview")
    st.dataframe(df.head(10))
    
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Movie Data (CSV)",
        data=csv,
        file_name="movie_data.csv",
        mime="text/csv"
    )
    
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
    col2.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.3f}")
    col3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

# -----------------------------
# Tab 2: Visual Insights
# -----------------------------
with tab2:
    st.header("Visual Insights")

    # Scatter plot
    fig1, ax1 = plt.subplots()
    ax1.scatter(y_test, y_pred, alpha=0.6)
    ax1.set_xlabel("Actual Ratings")
    ax1.set_ylabel("Predicted Ratings")
    ax1.set_title("Actual vs Predicted Movie Ratings")
    st.pyplot(fig1)

    # Correlation heatmap (numeric-only)
    numeric_df = df.select_dtypes(include=np.number)
    fig2, ax2 = plt.subplots()
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# -----------------------------
# Tab 3: Prediction Interface
# -----------------------------
with tab3:
    st.header("Predict a Movie Rating")
    
    popularity = st.slider("Popularity", float(df["popularity"].min()), float(df["popularity"].max()), float(df["popularity"].mean()))
    vote_count = st.slider("Vote Count", int(df["vote_count"].min()), int(df["vote_count"].max()), int(df["vote_count"].mean()))
    release_year = st.slider("Release Year", int(df["release_year"].min()), int(df["release_year"].max()), int(df["release_year"].mean()))
    
    languages = [col.replace("original_language_", "") for col in X.columns if col.startswith("original_language_")]
    selected_lang = st.selectbox("Movie Language", languages)
    
    user_input = pd.DataFrame({
        "popularity": [popularity],
        "vote_count": [vote_count],
        "release_year": [release_year]
    })
    for lang in languages:
        user_input[f"original_language_{lang}"] = 1 if lang == selected_lang else 0
    
    predicted_rating = model.predict(user_input)[0]
    st.success(f"⭐ Predicted Average Rating: **{predicted_rating:.2f} / 10**")
