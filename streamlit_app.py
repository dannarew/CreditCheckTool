import streamlit as st
import pandas as pd
import numpy as np
from src.config import *
from src.data_loader import get_data
from src.preprocess import preprocess_data
from src.segmenter import (
    find_optimal_k,
    perform_clustering,
    analyze_clusters,
    get_cluster_recommendations
)
from src.visualizer import (
    plot_elbow_curve,
    plot_cluster_scatter,
    plot_cluster_radar,
    plot_feature_distributions,
    plot_cluster_sizes,
    plot_feature_importance,
    create_visualization_section
)
from src.reporter import generate_report, save_report
from src.database import init_db, save_segmentation_results, get_customers_from_db, get_segmentation_history
from src.data_entry import create_data_entry_form, create_batch_data_generator
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
from typing import Tuple, List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize database
init_db()

# Page configuration
st.set_page_config(
    page_title="Credit Card Customer Segmentation",
    page_icon="💳",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stProgress .st-bo {
        background-color: #00cc00;
    }
    </style>
    """, unsafe_allow_html=True)

def perform_clustering(data: pd.DataFrame, n_clusters: int) -> Tuple[np.ndarray, float]:
    """Perform K-means clustering on the data."""
    # Select numerical features for clustering
    features = ['BALANCE', 'PURCHASES', 'PAYMENTS', 'CREDIT_LIMIT', 'MINIMUM_PAYMENTS']
    X = data[features].fillna(data[features].mean())
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    # Calculate silhouette score
    silhouette = silhouette_score(X_scaled, labels)
    
    return labels, silhouette

def plot_elbow_curve(inertias: List[float], silhouette_scores: List[float]) -> go.Figure:
    """Create an elbow curve plot."""
    fig = go.Figure()
    
    # Plot inertia
    fig.add_trace(go.Scatter(
        y=inertias,
        name='Inertia',
        mode='lines+markers'
    ))
    
    # Plot silhouette score on secondary y-axis
    fig.add_trace(go.Scatter(
        y=silhouette_scores,
        name='Silhouette Score',
        mode='lines+markers',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Elbow Curve and Silhouette Score',
        xaxis_title='Number of Clusters',
        yaxis_title='Inertia',
        yaxis2=dict(
            title='Silhouette Score',
            overlaying='y',
            side='right'
        )
    )
    
    return fig

def main():
    st.title("Credit Card Customer Segmentation")
    st.write("Analyze and segment credit card customers based on their spending behavior.")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Data Entry", "Analysis & Segmentation", "History & Reports"]
    )
    
    if page == "Data Entry":
        st.header("Data Entry")
        tab1, tab2 = st.tabs(["Manual Entry", "Batch Generation"])
        
        with tab1:
            create_data_entry_form()
        
        with tab2:
            create_batch_data_generator()
    
    elif page == "Analysis & Segmentation":
        st.header("Analysis & Segmentation")
        
        # Load data from database
        data = get_customers_from_db()
        
        if data is not None and not data.empty:
            # Data preprocessing options
            st.subheader("Data Preprocessing")
            handle_missing = st.checkbox("Handle Missing Values", value=True)
            
            if handle_missing:
                data = data.fillna(data.mean())
            
            # Clustering parameters
            st.subheader("Clustering Parameters")
            min_clusters = 2
            max_clusters = 8
            n_clusters = st.slider("Number of Clusters", min_clusters, max_clusters, 4)
            
            if st.button("Perform Clustering"):
                with st.spinner("Performing clustering analysis..."):
                    # Calculate elbow curve
                    inertias = []
                    silhouette_scores = []
                    
                    for k in range(min_clusters, max_clusters + 1):
                        labels, silhouette = perform_clustering(data, k)
                        inertias.append(sum((data[col] - data[col].mean()) ** 2 for col in data.columns))
                        silhouette_scores.append(silhouette)
                    
                    # Plot elbow curve
                    elbow_fig = plot_elbow_curve(inertias, silhouette_scores)
                    st.plotly_chart(elbow_fig)
                    
                    # Perform final clustering with selected number of clusters
                    labels, silhouette = perform_clustering(data, n_clusters)
                    st.success(f"Clustering completed! Silhouette Score: {silhouette:.3f}")
                    
                    # Save segmentation results
                    save_segmentation_results(data, labels, silhouette)
                    
                    # Display visualizations
                    create_visualization_section(data, labels)
        else:
            st.warning("No data available. Please enter or generate data in the Data Entry section.")
    
    else:  # History & Reports
        st.header("Segmentation History & Reports")
        history = get_segmentation_history()
        
        if history is not None and not history.empty:
            st.dataframe(history)
        else:
            st.info("No segmentation history available yet.")

if __name__ == "__main__":
    main() 