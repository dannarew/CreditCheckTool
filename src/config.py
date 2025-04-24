import os

"""
Configuration settings for the Credit Card Customer Segmentation application.
"""

# App Configuration
APP_TITLE = "Credit Card Customer Segmentation"
APP_DESCRIPTION = """
This application helps you segment credit card customers based on their transaction patterns
using K-means clustering algorithm.
"""

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'credit_segmentation',
    'user': 'postgres',
    'password': ''
}

# Feature Configuration
NUMERICAL_FEATURES = [
    'BALANCE',
    'PURCHASES',
    'PAYMENTS',
    'CREDIT_LIMIT',
    'MINIMUM_PAYMENTS'
]

# Clustering Configuration
MIN_CLUSTERS = 2
MAX_CLUSTERS = 8
DEFAULT_CLUSTERS = 4
RANDOM_STATE = 42

# Data Generation Configuration
SAMPLE_SIZE_RANGE = (100, 10000)
DEFAULT_SAMPLE_SIZE = 1000

FEATURE_RANGES = {
    'BALANCE': (0, 25000),
    'PURCHASES': (0, 15000),
    'PAYMENTS': (0, 15000),
    'CREDIT_LIMIT': (1000, 30000),
    'MINIMUM_PAYMENTS': (0, 1000)
}

# Visualization Configuration
PLOT_CONFIG = {
    'color_sequence': [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f'   # Gray
    ],
    'plot_height': 500,
    'plot_width': 800
}

# Application Theme
THEME = {
    'primary_color': '#1f77b4',
    'secondary_color': '#ff7f0e',
    'background_color': '#ffffff',
    'text_color': '#333333',
    'font_family': 'sans-serif'
}

# Custom CSS
CUSTOM_CSS = """
.main {
    padding: 2rem;
}
.stButton>button {
    width: 100%;
}
.stProgress .st-bo {
    background-color: #00cc00;
}
.reportview-container {
    background-color: #ffffff;
}
.sidebar .sidebar-content {
    background-color: #f8f9fa;
}
"""

# Error Messages
ERROR_MESSAGES = {
    'no_data': 'No data available. Please enter or generate data first.',
    'invalid_input': 'Invalid input. Please check your entries.',
    'db_connection': 'Could not connect to the database. Please check your configuration.',
    'clustering_error': 'An error occurred during clustering. Please try again.',
    'save_error': 'Could not save the results. Please try again.'
}

# Success Messages
SUCCESS_MESSAGES = {
    'data_saved': 'Data successfully saved to the database.',
    'clustering_complete': 'Clustering analysis completed successfully.',
    'report_generated': 'Analysis report generated successfully.'
}

# File Paths
PATHS = {
    'data_dir': 'data',
    'raw_data': 'data/raw',
    'processed_data': 'data/processed',
    'reports': 'reports',
    'assets': 'assets'
}

# Report Templates
REPORT_TEMPLATES = {
    'header': """
# Credit Card Customer Segmentation Analysis Report
Generated on: {date}

## Overview
This report presents the results of customer segmentation analysis performed on credit card transaction data.
""",
    'preprocessing': """
## Data Preprocessing
- Number of customers analyzed: {n_customers}
- Features used: {features}
- Preprocessing steps applied: {preprocessing_steps}
""",
    'clustering': """
## Clustering Results
- Number of clusters: {n_clusters}
- Silhouette score: {silhouette_score:.3f}
- Clustering algorithm: K-means
""",
    'recommendations': """
## Cluster Insights and Recommendations
{recommendations}
"""
}

# Data Processing
SCALING_METHODS = {
    'StandardScaler': 'Standard Scaler (z-score)',
    'MinMaxScaler': 'Min-Max Scaler',
    'RobustScaler': 'Robust Scaler'
}

# Report Settings
REPORT_TEMPLATE = """
# Customer Segmentation Report

## Overview
- Total Customers: {total_customers}
- Number of Segments: {n_segments}
- Analysis Date: {date}

## Segment Profiles
{segment_profiles}

## Recommendations
{recommendations}
""" 