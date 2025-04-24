import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Tuple
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def plot_elbow_curve(inertias: List[float], silhouette_scores: List[float]) -> go.Figure:
    """Plot elbow curve and silhouette scores."""
    fig = go.Figure()
    
    # Add elbow curve
    fig.add_trace(go.Scatter(
        x=list(range(2, len(inertias) + 2)),
        y=inertias,
        name='Inertia',
        line=dict(color='blue')
    ))
    
    # Add silhouette scores
    fig.add_trace(go.Scatter(
        x=list(range(2, len(silhouette_scores) + 2)),
        y=silhouette_scores,
        name='Silhouette Score',
        line=dict(color='red'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Elbow Curve and Silhouette Score',
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Inertia',
        yaxis2=dict(
            title='Silhouette Score',
            overlaying='y',
            side='right'
        ),
        showlegend=True
    )
    
    return fig

def plot_cluster_scatter(df: pd.DataFrame, 
                        labels: np.ndarray,
                        feature1: str,
                        feature2: str) -> go.Figure:
    """Create scatter plot of clusters."""
    fig = px.scatter(
        df,
        x=feature1,
        y=feature2,
        color=labels,
        title=f'Cluster Distribution: {feature1} vs {feature2}',
        labels={'color': 'Cluster'},
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title=feature1,
        yaxis_title=feature2,
        showlegend=True
    )
    
    return fig

def plot_cluster_radar(cluster_centers: np.ndarray,
                      feature_names: List[str],
                      cluster_idx: int) -> go.Figure:
    """Create radar chart for cluster characteristics."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=cluster_centers[cluster_idx],
        theta=feature_names,
        fill='toself',
        name=f'Cluster {cluster_idx}'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, np.max(cluster_centers)]
            )),
        showlegend=True,
        title=f'Cluster {cluster_idx} Characteristics'
    )
    
    return fig

def plot_feature_distributions(df: pd.DataFrame,
                             labels: np.ndarray,
                             feature_columns: List[str]) -> None:
    """Plot feature distributions for each cluster."""
    for feature in feature_columns:
        fig = px.box(
            df,
            x=labels,
            y=feature,
            title=f'Distribution of {feature} by Cluster',
            labels={'x': 'Cluster', 'y': feature}
        )
        st.plotly_chart(fig)

def plot_cluster_sizes(cluster_sizes: pd.Series) -> go.Figure:
    """Plot cluster sizes as a bar chart."""
    fig = px.bar(
        x=cluster_sizes.index,
        y=cluster_sizes.values,
        title='Cluster Sizes',
        labels={'x': 'Cluster', 'y': 'Number of Customers'}
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title='Cluster',
        yaxis_title='Number of Customers'
    )
    
    return fig

def plot_feature_importance(cluster_centers: np.ndarray,
                          feature_names: List[str]) -> go.Figure:
    """Plot feature importance based on cluster center variations."""
    # Calculate standard deviation of each feature across clusters
    feature_std = np.std(cluster_centers, axis=0)
    
    # Create bar chart
    fig = px.bar(
        x=feature_names,
        y=feature_std,
        title='Feature Importance in Clustering',
        labels={'x': 'Feature', 'y': 'Standard Deviation'}
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title='Feature',
        yaxis_title='Standard Deviation'
    )
    
    return fig

class CreditCardVisualizer:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def plot_feature_distribution(self, feature: str) -> None:
        """Create a distribution plot for a specific feature."""
        fig = px.histogram(
            self.data,
            x=feature,
            title=f'Distribution of {feature}',
            nbins=30
        )
        fig.update_layout(
            xaxis_title=feature,
            yaxis_title='Count',
            showlegend=False
        )
        st.plotly_chart(fig)

    def plot_correlation_heatmap(self) -> None:
        """Create a correlation heatmap for numerical features."""
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = self.data[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title='Feature Correlation Heatmap',
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig)

    def plot_segment_distribution(self, segment_labels: np.ndarray) -> None:
        """Plot the distribution of customer segments."""
        segment_counts = pd.Series(segment_labels).value_counts()
        
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title='Customer Segment Distribution'
        )
        st.plotly_chart(fig)

    def plot_segment_characteristics(self, segment_labels: np.ndarray, features: List[str]) -> None:
        """Create box plots showing feature distributions across segments."""
        df_with_segments = self.data.copy()
        df_with_segments['Segment'] = segment_labels
        
        for feature in features:
            fig = px.box(
                df_with_segments,
                x='Segment',
                y=feature,
                title=f'{feature} Distribution by Segment'
            )
            st.plotly_chart(fig)

    def plot_radar_chart(self, segment_labels: np.ndarray, features: List[str]) -> None:
        """Create a radar chart showing segment profiles."""
        df_with_segments = self.data.copy()
        df_with_segments['Segment'] = segment_labels
        
        # Calculate mean values for each feature by segment
        segment_profiles = df_with_segments.groupby('Segment')[features].mean()
        
        # Scale the features between 0 and 1 for better visualization
        scaler = lambda x: (x - x.min()) / (x.max() - x.min())
        segment_profiles_scaled = segment_profiles.apply(scaler)
        
        fig = go.Figure()
        for segment in segment_profiles_scaled.index:
            fig.add_trace(go.Scatterpolar(
                r=segment_profiles_scaled.loc[segment],
                theta=features,
                name=f'Segment {segment}'
            ))
            
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title='Segment Profiles'
        )
        st.plotly_chart(fig)

    def plot_3d_scatter(self, segment_labels: np.ndarray, features: List[str]) -> None:
        """Create a 3D scatter plot using three selected features."""
        if len(features) >= 3:
            df_plot = self.data.copy()
            df_plot['Segment'] = segment_labels
            
            fig = px.scatter_3d(
                df_plot,
                x=features[0],
                y=features[1],
                z=features[2],
                color='Segment',
                title='3D Segment Visualization'
            )
            st.plotly_chart(fig)
        else:
            st.warning("Need at least 3 features for 3D visualization")

    def create_segment_summary(self, segment_labels: np.ndarray) -> pd.DataFrame:
        """Generate a summary of segment characteristics."""
        df_with_segments = self.data.copy()
        df_with_segments['Segment'] = segment_labels
        
        summary = df_with_segments.groupby('Segment').agg({
            'BALANCE': ['mean', 'std'],
            'PURCHASES': ['mean', 'std'],
            'PAYMENTS': ['mean', 'std'],
            'CREDIT_LIMIT': ['mean', 'std'],
            'MINIMUM_PAYMENTS': ['mean', 'std']
        }).round(2)
        
        return summary

def create_visualization_section(data: pd.DataFrame, segment_labels: np.ndarray = None) -> None:
    """Create the visualization section in the Streamlit app."""
    st.header("Data Visualization and Analysis")
    
    visualizer = CreditCardVisualizer(data)
    
    # Feature Distribution Analysis
    st.subheader("Feature Distribution Analysis")
    feature_to_plot = st.selectbox(
        "Select feature to visualize:",
        options=data.select_dtypes(include=['float64', 'int64']).columns
    )
    visualizer.plot_feature_distribution(feature_to_plot)
    
    # Correlation Analysis
    st.subheader("Correlation Analysis")
    visualizer.plot_correlation_heatmap()
    
    if segment_labels is not None:
        # Segment Analysis
        st.subheader("Segment Analysis")
        visualizer.plot_segment_distribution(segment_labels)
        
        # Feature Distribution by Segment
        st.subheader("Feature Distribution by Segment")
        features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        selected_features = st.multiselect(
            "Select features to analyze:",
            options=features,
            default=features[:3]
        )
        
        if selected_features:
            visualizer.plot_segment_characteristics(segment_labels, selected_features)
            visualizer.plot_radar_chart(segment_labels, selected_features)
            
            if len(selected_features) >= 3:
                st.subheader("3D Segment Visualization")
                visualizer.plot_3d_scatter(segment_labels, selected_features)
        
        # Segment Summary
        st.subheader("Segment Summary Statistics")
        summary = visualizer.create_segment_summary(segment_labels)
        st.dataframe(summary) 