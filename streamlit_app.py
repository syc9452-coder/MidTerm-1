import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# --- Page Configuration ---
st.set_page_config(
    page_title="Spotify Success Predictor",
    layout="wide",
)

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("SpotifyTracksDataset.csv")
    # Clean unnamed column if it exists
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("Spotify Artist Toolkit 🎸")
page = st.sidebar.selectbox("Go to", ["Introduction", "Data Visualization", "Popularity Predictor"])

# --- PAGE 1: INTRODUCTION ---

if page == "Introduction":


    intro_col1, intro_col2 = st.columns([2, 1])
    
    with intro_col1:
        st.title("Spotify Success Predictor")
        st.subheader("Helping Independent Artists Crack the Algorithm")
        st.markdown(f"""
        Welcome to the **Spotify Artist Toolkit**. 
        This is a tool that empowers independent creators to analyze their music through the lens of machine learning.
        
        Use the sidebar to explore visual insights or predict your next track's performance.
        """, unsafe_allow_html=True)
        
    with intro_col2:
        # Adding the image here makes it look like a professional logo/banner
        st.image("spotify-image.png", use_container_width=True)

    st.divider()


    
    col4, col5 = st.columns([1, 1])
    
    with col4:
        st.markdown("""
        ### :green[The Problem]
        Small independent artists often struggle to know if their music fits current trends. 
        Without a major label's data team, how do you know if your song's **Acousticness** is too high or if your **Tempo** is too slow for a hit?
        
        ### :green[The Goal]
        This app uses **Machine Learning (Linear Regression)** to:
        1. Identify which audio features drive track popularity.
        2. Help artists optimize their track characteristics before release.
        """)
        
    with col5:
        st.info("**Dataset Info:** 114,000 tracks across 114 genres. Each genre has exactly 1,000 songs.")

    st.divider()
    st.markdown("### Data Preview")
    st.dataframe(df.head(10))
    
    st.markdown("### Dataset Summary 📊 ")
    st.write(df.describe())








# --- PAGE 2: VISUALIZATION ---

elif page == "Data Visualization":
    st.title("📊 Key Insights from Spotify Data")
    st.markdown("We analyzed 114,000 tracks to identify the patterns that lead to success.")


    # Top Genres
    st.subheader("🏆 Most Popular Genres")
    st.markdown("Before diving into features, let's see which genres are currently trending.")
    # Calculate average popularity per genre and take the top 10
    top_genres = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=False).head(15).reset_index()
    top_genres = top_genres.set_index('track_genre')

    st.bar_chart(top_genres, color="#1DB954") # Spotify Green color
    st.divider()


    # Energy vs Genre
    st.subheader("1. Energy Score vs. Popularity & Genre")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image("energy_genre.png") # Make sure this filename matches your saved screenshot
    with col2:
        st.markdown("""
        **What this shows:** This graph tracks how the average energy of a song changes as popularity increases across different genres.
        
        **Key Takeaway:** For 'K-Pop' and 'Electronic' music, high energy is a requirement for popularity. However, for 'Sleep' and 'Study' music, the trend is reversed: lower energy levels are actually more popular.
        """)

    st.divider()


    # Loudness Trends
    st.subheader("2. Loudness vs. Popularity")
    col3, col4 = st.columns([2, 1])
    with col3:
        st.image("loudness_genre.png")
    with col4:
        st.markdown("""
        **What this shows:** Loudness (measured in dB) across popularity tiers. Note that -5dB is louder than -20dB.
        
        **Key Takeaway:** Loudness affects popularity differently by genre. For calmer genres like 'Sleep' and 'Sad', more popular songs tend to be quieter, which aligns with listener expectations for soft, emotional music. In contrast, 'Alt-Rock' and 'Techno' tracks are consistently loud, and their most popular songs are slightly louder still, suggesting that intensity boosts success in high-energy genres.
        """)

    st.divider()

    # Energy Bins (Balanced)
    st.subheader("3. Popularity by Energy Bins")
    col5, col6 = st.columns([2, 1])
    with col5:
        st.image("energy_bins.png")
    with col6:
        st.markdown("""
        **What this shows:** Average popularity score across five balanced levels of energy.
        
        **Key Takeaway:** There is a 'sweet spot' for energy. While very low energy tracks struggle, the highest popularity is often found in the mid-to-high energy range (around 0.6 to 0.8).
        """)

    st.divider()

    # Tempo sweet spot
    st.subheader("4. The Impact of Tempo (BPM)")
    col7, col8 = st.columns([2, 1])
    with col7:
        st.image("tempo_bins.png")
    with col8:
        st.markdown("""
        **What this shows:** How the Beats Per Minute (BPM) correlates with average popularity.
        
        **Key Takeaway:** Tracks with a tempo between **140 and 160 BPM** show a significant peak in popularity. This is the standard 'heartbeat' of modern hit radio and club music.
        """)

    st.divider()

    # Top Tracks snapshot
    st.subheader("5. Top 10 Tracks in the Dataset")

    col_left, col_mid, col_right = st.columns([1, 4, 1])
    with col_mid:
        st.image("top_tracks.png", use_container_width=True)
        
    st.markdown("""
    **Description:** This chart shows the 'ceiling' of our dataset. Tracks like 'Unholy' and 'Quevedo' represent the 100/100 popularity score. 
    Our goal with the prediction model is to see how close *your* song can get to these leaders.
    """)








# --- PAGE 3: PREDICTION ---
elif page == "Popularity Predictor":
    st.title("Predict Your Track's Popularity")
    
    # Create a copy of the dataframe to avoid modifying the original 'df'
    # .dropna() to ensure the model doesn't crash due to missing values
    df_model = df.copy().dropna()
    
    # Encode Genre: 
    # # LabelEncoder converts genre names  into numbers (0, 1, 2...)
    le = LabelEncoder()
    df_model['genre_encoded'] = le.fit_transform(df_model['track_genre'])
    
    # Features selection: 
    # Define the 'Independent Variables' (X) that we think influence popularity
    features = ['genre_encoded', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    X = df_model[features] # Features (The 'Cause')
    y = df_model['popularity'] # Target Variable (The 'Effect')
    
    # Split and Train: 
    # hiding 20% of the data from the model to 'test' it later.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression() # Linear Regression model
    model.fit(X_train, y_train)  # Finds the 'line of best fit' between audio features and popularity scores
    
    # Artist Input UI
    st.sidebar.header("Configure Your Track")
    user_inputs = {}

    # Gnere:
    # We get unique genres from the original dataframe
    available_genres = sorted(df['track_genre'].unique())
    selected_genre = st.sidebar.selectbox("Select Genre", available_genres)
    
    # Convert the selected text genre back to its encoded number for the model
    user_inputs['genre_encoded'] = le.transform([selected_genre])[0]



    for feat in features:
        if feat == 'genre_encoded':
            continue # Skip because we handled it above with the selectbox
        if feat == 'tempo':
            user_inputs[feat] = st.sidebar.slider(feat, 40, 250, 120) # BPM usually ranges from 40 to 250
        elif feat == 'loudness':
            user_inputs[feat] = st.sidebar.slider(feat, -60, 0, -10) # Loudness is measured in negative decibels (dB)
        elif feat in ['key', 'mode']:
            user_inputs[feat] = st.sidebar.number_input(feat, min_value=0, max_value=11, value=1)
        else:
            # Most Spotify features (Energy, Danceability) are normalized between 0.0 and 1.0
            user_inputs[feat] = st.sidebar.slider(feat, 0.0, 1.0, 0.5)
            

    # Make prediction:
    # Convert user inputs into a 1-row DataFrame for the model to read
    input_df = pd.DataFrame([user_inputs])
    prediction = model.predict(input_df)[0]
    
    # Display Result:
    st.metric("Predicted Popularity Score", f"{prediction:.2f}/100")
    
    if prediction > 50:
        st.success("This track has high hit potential!")
    else:
        st.warning("This might be a niche track. Consider adjusting your features.")



    # Model Evaluation
    # An expander keeps the technical data hidden unless the 'judges' want to see it
    with st.expander("Show Model Performance"):
        y_pred = model.predict(X_test)

        # R-squared
        st.write(f"R² Score: {metrics.r2_score(y_test, y_pred):.4f}")

        # MAE
        st.write(f"MAE: {metrics.mean_absolute_error(y_test, y_pred):.2f}")
        
        # Coefficients: show which feature has the biggest impact on the final score
        coef_df = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
        st.write("### Feature Importance (Impact on Popularity)")
        st.bar_chart(coef_df.set_index('Feature'))