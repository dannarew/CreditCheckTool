import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple, Dict, List

def find_optimal_k(data: np.ndarray, max_k: int = 10) -> Tuple[int, List[float]]:
    """Find optimal number of clusters using elbow method and silhouette score."""
    inertias = []
    silhouette_scores = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    
    # Find elbow point using the second derivative
    inertias = np.array(inertias)
    diffs = np.diff(inertias)
    diffs_r = np.diff(diffs)
    k_optimal = np.argmax(diffs_r) + 2
    
    return k_optimal, silhouette_scores

def perform_clustering(data: np.ndarray, 
                      n_clusters: int,
                      random_state: int = 42) -> Tuple[np.ndarray, Dict]:
    """Perform K-means clustering on the data."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(data)
    
    # Calculate cluster centers
    centers = kmeans.cluster_centers_
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(data, labels)
    
    # Prepare clustering info
    clustering_info = {
        'n_clusters': n_clusters,
        'inertia': kmeans.inertia_,
        'silhouette_score': silhouette_avg,
        'cluster_centers': centers
    }
    
    return labels, clustering_info

def analyze_clusters(df: pd.DataFrame, 
                    labels: np.ndarray, 
                    feature_columns: List[str]) -> pd.DataFrame:
    """Analyze and profile each cluster."""
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = labels
    
    # Calculate mean values for each feature in each cluster
    cluster_profiles = df_with_clusters.groupby('Cluster')[feature_columns].mean()
    
    # Calculate size of each cluster
    cluster_sizes = df_with_clusters['Cluster'].value_counts().sort_index()
    
    # Calculate percentage of total
    cluster_percentages = (cluster_sizes / len(df) * 100).round(2)
    
    # Combine all metrics
    cluster_analysis = pd.DataFrame({
        'Size': cluster_sizes,
        'Percentage': cluster_percentages
    })
    
    # Add feature means
    for col in feature_columns:
        cluster_analysis[f'{col}_mean'] = cluster_profiles[col]
    
    return cluster_analysis

def get_cluster_recommendations(cluster_analysis: pd.DataFrame, 
                              feature_columns: List[str]) -> Dict[int, str]:
    """Generate recommendations for each cluster based on their characteristics."""
    recommendations = {}
    
    for cluster in cluster_analysis.index:
        profile = cluster_analysis.loc[cluster]
        
        # Analyze spending patterns
        if profile['BALANCE_mean'] > profile['CREDIT_LIMIT_mean'] * 0.8:
            risk_level = "High"
        elif profile['BALANCE_mean'] > profile['CREDIT_LIMIT_mean'] * 0.5:
            risk_level = "Medium"
        else:
            risk_level = "Low"
            
        # Generate recommendation
        recommendation = f"""
        This segment represents {profile['Percentage']}% of customers.
        Risk Level: {risk_level}
        
        Key Characteristics:
        - Average Balance: ${profile['BALANCE_mean']:.2f}
        - Average Purchases: ${profile['PURCHASES_mean']:.2f}
        - Credit Utilization: {(profile['BALANCE_mean']/profile['CREDIT_LIMIT_mean']*100):.1f}%
        
        Recommended Actions:
        - {'Consider credit limit increase' if risk_level == 'Low' else 'Monitor credit utilization'}
        - {'Offer premium rewards program' if profile['PURCHASES_mean'] > profile['PURCHASES_mean'].mean() else 'Focus on basic rewards'}
        - {'Implement risk management strategies' if risk_level == 'High' else 'Focus on growth opportunities'}
        """
        
        recommendations[cluster] = recommendation 