import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
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

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Credit Card Customer Segmentation",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

def generate_recommendations(data, labels):
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

def display_cluster_profiles(data, labels, recommendations):
    """Display detailed profiles and recommendations for each cluster."""
    st.header("Cluster Profiles and Recommendations")
    
    # Add cluster labels to data
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = labels
    
    # Create tabs for each cluster
    tabs = st.tabs([f"Cluster {i}" for i in range(len(np.unique(labels)))])
    
    for i, tab in enumerate(tabs):
        with tab:
            cluster_id = str(i)
            cluster_rec = recommendations[cluster_id]
            
            # Display cluster metrics
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
            
            # Display credit utilization
            st.subheader("Credit Utilization")
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
            
            # Display eligible products
            st.subheader("Eligible Products")
            for product in cluster_rec["eligible_products"]:
                st.info(product)
            
            # Display recommendations
            st.subheader("Recommendations")
            for rec in cluster_rec["specific_recommendations"]:
                st.write(f"- {rec}")
            
            # Display customer distribution
            st.subheader("Customer Distribution")
            cluster_data = data_with_clusters[data_with_clusters['Cluster'] == i]
            
            # Create a scatter plot of customers in this cluster
            fig = px.scatter(cluster_data, x='BALANCE', y='PURCHASES',
                            title=f'Customer Distribution in Cluster {i}',
                            labels={'BALANCE': 'Balance ($)', 'PURCHASES': 'Purchases ($)'})
            st.plotly_chart(fig)

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
if st.button("Run Clustering"):
    with st.spinner("Running clustering analysis..."):
        # Load and preprocess data
        data = get_customers_from_db()
        if data is not None:
            data = data[NUMERICAL_FEATURES].fillna(0)
            
            # Run clustering
            n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4)
            kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
            labels = kmeans.fit_predict(data)
            data['Cluster'] = labels
            
            # Calculate silhouette score
            sil_score = silhouette_score(data, labels)
            
            # Generate cluster profiles
            cluster_profiles = {}
            for cluster_id in range(n_clusters):
                cluster_data = data[data['Cluster'] == cluster_id]
                profile = {
                    'size': len(cluster_data),
                    'mean_balance': float(cluster_data['BALANCE'].mean()),
                    'mean_purchases': float(cluster_data['PURCHASES'].mean()),
                    'mean_payments': float(cluster_data['PAYMENTS'].mean()),
                    'mean_credit_limit': float(cluster_data['CREDIT_LIMIT'].mean()),
                    'mean_minimum_payments': float(cluster_data['MINIMUM_PAYMENTS'].mean())
                }
                cluster_profiles[str(cluster_id)] = profile
            
            # Generate detailed recommendations using the comprehensive function
            recommendations = generate_recommendations(data, labels)
            
            # Save results to database
            success = save_segmentation_results(
                analysis_date=datetime.now(),
                n_clusters=n_clusters,
                silhouette_score=sil_score,
                preprocessing_steps=['standardization', 'missing_value_imputation'],
                cluster_profiles=cluster_profiles,
                recommendations=recommendations
            )
            
            if success:
                st.success("Clustering analysis completed and results saved successfully!")
            else:
                st.error("Error saving clustering results to database.")
            
            # Visualize results
            fig = px.scatter(data, x='BALANCE', y='PURCHASES', color='Cluster',
                            title='Customer Segments')
            st.plotly_chart(fig)
            
            # Show metrics
            st.metric("Silhouette Score", f"{sil_score:.3f}")
            
            # Display cluster profiles and recommendations
            display_cluster_profiles(data, labels, recommendations)
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