import streamlit as st
import pandas as pd
import joblib
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
import time

# Set page config
st.set_page_config(page_title="Sentiment Analysis App", page_icon="📊", layout="wide")

# --- 1. LOAD THE MODEL PIPELINE ---
@st.cache_resource
def load_model_pipeline():
    """Loads the saved Logistic Regression pipeline."""
    try:
        model_path = Path('models') / 'model_lr.joblib'  
        pipeline = joblib.load(model_path)
        print("✅ Model pipeline loaded successfully.")
        return pipeline
    except Exception as e:
        st.error(f"❌ Could not load model pipeline: {e}")
        return None


# --- 2. LOAD THE DATASET ---
@st.cache_data
def load_data():
    """Loads the dataset for analysis."""
    try:
        data_path = Path('outputs') / 'tokopedia_sentiments.csv'   
        df = pd.read_csv(data_path)
        print("✅ Dataset loaded successfully.")
        return df
    except FileNotFoundError:
        st.error("❌ tokopedia_sentiments.csv not found in /outputs folder.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None

# Load model and data
model_pipeline = load_model_pipeline()

df_latih = load_data()

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("📊 Sentiment Analysis NLP App")
page = st.sidebar.selectbox("Choose a page:", 
                           ["Home (Prediction)", "Dataset & Analysis", "KMeans Clustering", "About"])

# --- 4. PAGE: HOME (PREDICTION) ---
if page == "Home (Prediction)":
    st.title("📊 Indonesian Sentiment Analysis")
    st.markdown("Enter a text (like a Tokopedia review or a Tweet) to predict its sentiment.")
    
    text_input = st.text_area("Input Text:", height=150, placeholder="Tulis ulasan atau teks Anda di sini...")

    if st.button("Analyze Sentiment", type="primary"):
        if model_pipeline is None:
            st.error("Model not loaded. Please check your model file.")
        elif not text_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner('Analyzing...'):
                time.sleep(1)
                try:
                    prediction = model_pipeline.predict([text_input])[0]

                    st.subheader("Analysis Result:")
                    if prediction == 'positive':
                        st.success("**Predicted Sentiment: Positive 😊**")
                    elif prediction == 'negative':
                        st.error("**Predicted Sentiment: Negative 😠**")
                    else:
                        st.info("**Predicted Sentiment: Neutral 😐**")

                    st.write("---")
                    st.write("**Input Text:**")
                    st.write(f"_{text_input}_")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

# --- 5. PAGE: DATASET & ANALYSIS ---
elif page == "Dataset & Analysis":
    st.header("📈 Dataset & Sentiment Analysis")
    if df_latih is None:
        st.warning("Dataset not found. Please make sure 'tokopedia_sentiments.csv' is in the outputs folder.")
    else:
        st.subheader("Dataset Preview")
        st.dataframe(df_latih.head())
        st.write(f"Total entries: **{len(df_latih)}**")

        st.subheader("Sentiment Distribution")
        if 'sentiment_final' in df_latih.columns:
            st.bar_chart(df_latih['sentiment_final'].value_counts())
        else:
            st.error("Column 'sentiment_final' not found in dataset.")

        st.subheader("Word Cloud")
        try:
            all_text = ' '.join(df_latih['review'])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not generate word cloud: {e}")

# --- 6. PAGE: KMEANS CLUSTERING ---
elif page == "KMeans Clustering":
    st.header("🔬 KMeans Clustering Analysis")

    if df_latih is None or model_pipeline is None:
        st.error("Dataset or model not loaded. Cannot perform clustering.")
    else:
        # Clustering parameters
        st.subheader("Clustering Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=8, value=3)
        
        with col2:
            max_iter = st.slider("Max Iterations", min_value=100, max_value=500, value=300)
        
        if st.button("Run Clustering", type="primary"):
            with st.spinner("Clustering in progress..."):
                try:
                    if 'tfidf' in model_pipeline.named_steps:
                        # Transform text to TF-IDF vectors
                        tfidf_vectors = model_pipeline.named_steps['tfidf'].transform(df_latih['review'])
                        
                        # Perform KMeans clustering
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=max_iter)
                        kmeans.fit(tfidf_vectors)
                        
                        # Calculate silhouette score
                        silhouette_avg = silhouette_score(tfidf_vectors, kmeans.labels_)
                        
                        # Display results
                        st.subheader("Clustering Results")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
                        with col2:
                            st.metric("Number of Clusters", n_clusters)
                        with col3:
                            st.metric("Total Samples", len(df_latih))
                        
                        # Cluster distribution
                        cluster_counts = pd.Series(kmeans.labels_).value_counts().sort_index()
                        st.subheader("Cluster Distribution")
                        st.bar_chart(cluster_counts)
                        
                        # Show sample reviews from each cluster
                        st.subheader("Sample Reviews by Cluster")
                        for cluster_id in range(n_clusters):
                            cluster_reviews = df_latih[kmeans.labels_ == cluster_id]
                            if len(cluster_reviews) > 0:
                                with st.expander(f"Cluster {cluster_id} ({len(cluster_reviews)} reviews)"):
                                    sample_reviews = cluster_reviews['review'].head(3)
                                    for i, review in enumerate(sample_reviews, 1):
                                        st.write(f"{i}. {review}")
                        
                    else:
                        st.error("The pipeline does not contain a 'tfidf' step.")
                        
                except Exception as e:
                    st.error(f"An error occurred during clustering: {e}")

# --- 7. PAGE: ABOUT ---
elif page == "About":
    st.header("About This App")
    st.markdown("""
    This Streamlit app performs **Indonesian Sentiment Analysis** using a pre-trained Logistic Regression model.

    **Project Structure:**
    - `models/model_lr.joblib` → Trained pipeline  
    - `outputs/tokopedia_sentiments.csv` → Dataset for analysis  
    - `py/app.py` → This Streamlit dashboard  

    **Features:**
    - **Home:** Predict sentiment from input text  
    - **Dataset & Analysis:** Visualize dataset and word cloud  
    - **KMeans Clustering:** Topic clustering via TF-IDF vectors  
    """)