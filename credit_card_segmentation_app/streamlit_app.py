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
from typing import Tuple, List, Dict, Any
import os
from dotenv import load_dotenv
from datetime import datetime

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

def generate_recommendations(data: pd.DataFrame, labels: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Generate detailed recommendations for each cluster based on their credit card usage patterns."""
    recommendations = {}
    
    # Add cluster labels to data
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = labels
    
    # Calculate credit utilization percentage
    data_with_clusters['Credit_Utilization'] = (data_with_clusters['BALANCE'] / data_with_clusters['CREDIT_LIMIT']) * 100
    
    # Calculate payment ratio
    data_with_clusters['Payment_Ratio'] = data_with_clusters['PAYMENTS'] / data_with_clusters['PURCHASES'].replace(0, 1)
    
    # Calculate risk score (higher balance and lower payments = higher risk)
    data_with_clusters['Risk_Score'] = (data_with_clusters['BALANCE'] / data_with_clusters['CREDIT_LIMIT']) * (1 - data_with_clusters['Payment_Ratio'].clip(0, 1))
    
    # Generate recommendations for each cluster
    for cluster_id in range(len(np.unique(labels))):
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster_id]
        
        # Calculate average metrics for the cluster
        avg_balance = cluster_data['BALANCE'].mean()
        avg_purchases = cluster_data['PURCHASES'].mean()
        avg_payments = cluster_data['PAYMENTS'].mean()
        avg_credit_limit = cluster_data['CREDIT_LIMIT'].mean()
        avg_utilization = cluster_data['Credit_Utilization'].mean()
        avg_risk_score = cluster_data['Risk_Score'].mean()
        
        # Determine risk level
        if avg_risk_score < 0.2:
            risk_level = "Low"
        elif avg_risk_score < 0.5:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Determine eligible products based on cluster characteristics
        eligible_products = []
        
        # Premium card (high credit limit, high purchases, low risk)
        if avg_credit_limit > 15000 and avg_purchases > 5000 and avg_risk_score < 0.3:
            eligible_products.append("Premium Rewards Card")
        
        # Standard card (moderate credit limit, moderate purchases)
        if avg_credit_limit > 5000 and avg_purchases > 1000:
            eligible_products.append("Standard Rewards Card")
        
        # Basic card (lower credit limit, lower purchases, or higher risk)
        if avg_credit_limit > 1000:
            eligible_products.append("Basic Card")
        
        # Generate specific recommendations
        specific_recommendations = []
        
        # Credit utilization recommendations
        if avg_utilization > 80:
            specific_recommendations.append("High credit utilization detected. Consider reducing balance or requesting a credit limit increase.")
        elif avg_utilization > 50:
            specific_recommendations.append("Moderate credit utilization. Maintain current spending patterns.")
        else:
            specific_recommendations.append("Low credit utilization. Consider increasing spending to maximize rewards.")
        
        # Payment recommendations
        if avg_payments < avg_purchases * 0.8:
            specific_recommendations.append("Payment amount is less than purchases. Consider increasing payments to avoid interest charges.")
        else:
            specific_recommendations.append("Good payment habits. Continue paying more than the minimum payment.")
        
        # Risk-based recommendations
        if risk_level == "High":
            specific_recommendations.append("High risk profile. Consider debt consolidation or financial counseling.")
        elif risk_level == "Medium":
            specific_recommendations.append("Medium risk profile. Focus on reducing outstanding balance.")
        else:
            specific_recommendations.append("Low risk profile. Eligible for premium credit products.")
        
        # Store recommendations for this cluster
        recommendations[str(cluster_id)] = {
            "risk_level": risk_level,
            "credit_utilization": f"{avg_utilization:.1f}%",
            "eligible_products": eligible_products,
            "specific_recommendations": specific_recommendations,
            "cluster_size": len(cluster_data),
            "avg_balance": f"${avg_balance:.2f}",
            "avg_purchases": f"${avg_purchases:.2f}",
            "avg_payments": f"${avg_payments:.2f}",
            "avg_credit_limit": f"${avg_credit_limit:.2f}"
        }
    
    return recommendations

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
            
            # Ensure all data is numeric
            numeric_columns = ['BALANCE', 'PURCHASES', 'PAYMENTS', 'CREDIT_LIMIT', 'MINIMUM_PAYMENTS']
            for col in numeric_columns:
                if col in data.columns:
                    # Convert to numeric, coercing errors to NaN
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            if handle_missing:
                # Fill NaN values with mean of each column
                for col in numeric_columns:
                    if col in data.columns:
                        data[col] = data[col].fillna(data[col].mean())
            
            # Clustering parameters
            st.subheader("Clustering Parameters")
            min_clusters = 2
            max_clusters = 8
            n_clusters = st.slider("Number of Clusters", min_clusters, max_clusters, 4)
            
            # Move the button outside of any form context
            run_analysis = st.button("Perform Clustering")
            
            if run_analysis:
                with st.spinner("Performing clustering analysis..."):
                    # Ensure we only use numeric columns for clustering
                    clustering_data = data[numeric_columns].copy()
                    
                    # Calculate elbow curve
                    inertias = []
                    silhouette_scores = []
                    
                    for k in range(min_clusters, max_clusters + 1):
                        labels, silhouette = perform_clustering(clustering_data, k)
                        inertias.append(sum((clustering_data[col] - clustering_data[col].mean()) ** 2 for col in clustering_data.columns))
                        silhouette_scores.append(silhouette)
                    
                    # Plot elbow curve
                    elbow_fig = plot_elbow_curve(inertias, silhouette_scores)
                    st.plotly_chart(elbow_fig)
                    
                    # Perform final clustering with selected number of clusters
                    labels, silhouette = perform_clustering(clustering_data, n_clusters)
                    st.success(f"Clustering completed! Silhouette Score: {silhouette:.3f}")
                    
                    # Generate comprehensive recommendations
                    recommendations = generate_recommendations(clustering_data, labels)
                    
                    # Save segmentation results with recommendations
                    save_segmentation_results(
                        analysis_date=datetime.now(),
                        n_clusters=n_clusters,
                        silhouette_score=silhouette,
                        preprocessing_steps=['standardization', 'missing_value_imputation'],
                        cluster_profiles={str(i): {
                            'size': len(clustering_data[labels == i]),
                            'mean_balance': float(clustering_data[labels == i]['BALANCE'].mean()),
                            'mean_purchases': float(clustering_data[labels == i]['PURCHASES'].mean()),
                            'mean_payments': float(clustering_data[labels == i]['PAYMENTS'].mean()),
                            'mean_credit_limit': float(clustering_data[labels == i]['CREDIT_LIMIT'].mean())
                        } for i in range(n_clusters)},
                        recommendations=recommendations
                    )
                    
                    # Display comprehensive analysis
                    st.header("Comprehensive Analysis")
                    
                    # Create tabs for different analysis views
                    analysis_tabs = st.tabs(["Overview", "Cluster Profiles", "Visualizations", "Recommendations"])
                    
                    with analysis_tabs[0]:  # Overview
                        st.subheader("Analysis Overview")
                        st.write(f"Total Customers: {len(data)}")
                        st.write(f"Number of Segments: {n_clusters}")
                        st.write(f"Silhouette Score: {silhouette:.3f}")
                        
                        # Display cluster sizes
                        cluster_sizes = pd.Series(labels).value_counts().sort_index()
                        fig = go.Figure(data=[go.Bar(
                            x=[f"Cluster {i}" for i in cluster_sizes.index],
                            y=cluster_sizes.values,
                            text=cluster_sizes.values,
                            textposition='auto',
                        )])
                        fig.update_layout(title="Cluster Sizes")
                        st.plotly_chart(fig)
                    
                    with analysis_tabs[1]:  # Cluster Profiles
                        st.subheader("Detailed Cluster Profiles")
                        for cluster_id in range(n_clusters):
                            with st.expander(f"Cluster {cluster_id} Profile"):
                                cluster_rec = recommendations[str(cluster_id)]
                                
                                # Display metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Cluster Size", cluster_rec["cluster_size"])
                                    st.metric("Risk Level", cluster_rec["risk_level"])
                                with col2:
                                    st.metric("Avg Balance", cluster_rec["avg_balance"])
                                    st.metric("Avg Purchases", cluster_rec["avg_purchases"])
                                with col3:
                                    st.metric("Avg Payments", cluster_rec["avg_payments"])
                                    st.metric("Avg Credit Limit", cluster_rec["avg_credit_limit"])
                                
                                # Display credit utilization gauge
                                utilization = float(cluster_rec["credit_utilization"].replace("%", ""))
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=utilization,
                                    title={'text': "Credit Utilization"},
                                    gauge={'axis': {'range': [0, 100]},
                                           'bar': {'color': "darkblue"},
                                           'steps': [
                                               {'range': [0, 30], 'color': "lightgreen"},
                                               {'range': [30, 70], 'color': "yellow"},
                                               {'range': [70, 100], 'color': "red"}
                                           ]}
                                ))
                                st.plotly_chart(fig)
                    
                    with analysis_tabs[2]:  # Visualizations
                        st.subheader("Data Visualizations")
                        create_visualization_section(clustering_data, labels)
                    
                    with analysis_tabs[3]:  # Recommendations
                        st.subheader("Strategic Recommendations")
                        for cluster_id in range(n_clusters):
                            with st.expander(f"Recommendations for Cluster {cluster_id}"):
                                cluster_rec = recommendations[str(cluster_id)]
                                
                                # Display eligible products
                                st.write("### Eligible Products")
                                for product in cluster_rec["eligible_products"]:
                                    st.info(product)
                                
                                # Display specific recommendations
                                st.write("### Specific Recommendations")
                                for rec in cluster_rec["specific_recommendations"]:
                                    st.write(f"- {rec}")
                                
                                # Display risk assessment
                                st.write("### Risk Assessment")
                                st.write(f"Risk Level: {cluster_rec['risk_level']}")
                                st.write(f"Credit Utilization: {cluster_rec['credit_utilization']}")
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