import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Credit Card Customer Segmentation",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
from src.database import (
    init_db,
    save_customers_to_db,
    get_customers_from_db,
    save_segmentation_results,
    get_segmentation_history
)
from src.config import (
    APP_TITLE,
    APP_DESCRIPTION,
    NUMERICAL_FEATURES,
    MIN_CLUSTERS,
    MAX_CLUSTERS,
    DEFAULT_CLUSTERS,
    RANDOM_STATE,
    CUSTOM_CSS
)

# Initialize database
init_db()

# Add custom CSS
st.markdown(f'<style>{CUSTOM_CSS}</style>', unsafe_allow_html=True)

# App title and description
st.title(APP_TITLE)
st.markdown(APP_DESCRIPTION)

# Sidebar
st.sidebar.header("Controls")

# Data input section
st.header("Data Input")
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("Data loaded successfully!")
        
        # Save to database
        if st.button("Save to Database"):
            if save_customers_to_db(data):
                st.success("Data saved to database successfully!")
            else:
                st.error("Failed to save data to database.")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

# Load data from database
st.header("Load Data from Database")
if st.button("Load Data"):
    data = get_customers_from_db()
    if data is not None:
        st.success("Data loaded from database!")
        st.write(data.head())
    else:
        st.error("No data found in database.")

# Clustering section
st.header("Customer Segmentation")
n_clusters = st.slider("Number of Clusters", MIN_CLUSTERS, MAX_CLUSTERS, DEFAULT_CLUSTERS)

if st.button("Run Clustering"):
    data = get_customers_from_db()
    if data is not None:
        # Prepare data
        X = data[NUMERICAL_FEATURES].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
        labels = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        sil_score = silhouette_score(X_scaled, labels)
        
        # Save results
        preprocessing_steps = ["StandardScaler", "Missing values filled with 0"]
        if save_segmentation_results(data, labels, sil_score, preprocessing_steps):
            st.success("Segmentation results saved!")
        
        # Visualize results
        data['Cluster'] = labels
        fig = px.scatter(data, x='BALANCE', y='PURCHASES', color='Cluster',
                        title='Customer Segments')
        st.plotly_chart(fig)
        
        # Show metrics
        st.metric("Silhouette Score", f"{sil_score:.3f}")
    else:
        st.error("Please load data first.")

# View history
st.header("Segmentation History")
if st.button("View History"):
    history = get_segmentation_history()
    if history is not None:
        st.write(history)
    else:
        st.info("No segmentation history found.") 