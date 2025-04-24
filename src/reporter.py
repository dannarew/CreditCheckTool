"""
Module for generating reports based on credit card customer segmentation analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
from .config import REPORT_TEMPLATES, PATHS
import os

def generate_summary_stats(df: pd.DataFrame, 
                         feature_columns: List[str]) -> Dict:
    """Generate summary statistics for the dataset."""
    summary_stats = {}
    
    # Basic dataset info
    summary_stats['total_customers'] = len(df)
    summary_stats['total_features'] = len(feature_columns)
    
    # Feature statistics
    for feature in feature_columns:
        summary_stats[feature] = {
            'mean': df[feature].mean(),
            'median': df[feature].median(),
            'std': df[feature].std(),
            'min': df[feature].min(),
            'max': df[feature].max()
        }
    
    return summary_stats

def generate_cluster_profile(cluster_data: pd.DataFrame) -> str:
    """Generate a profile description for a cluster based on its characteristics."""
    profile = []
    
    # Calculate statistics
    mean_balance = cluster_data['BALANCE'].mean()
    mean_purchases = cluster_data['PURCHASES'].mean()
    mean_payments = cluster_data['PAYMENTS'].mean()
    mean_credit_limit = cluster_data['CREDIT_LIMIT'].mean()
    mean_min_payments = cluster_data['MINIMUM_PAYMENTS'].mean()
    
    # Determine spending behavior
    if mean_purchases > mean_payments * 1.2:
        spending = "high-spending"
    elif mean_purchases < mean_payments * 0.8:
        spending = "conservative"
    else:
        spending = "balanced"
    
    # Determine credit utilization
    credit_utilization = (mean_balance / mean_credit_limit) * 100
    if credit_utilization > 70:
        credit_status = "high credit utilization"
    elif credit_utilization < 30:
        credit_status = "low credit utilization"
    else:
        credit_status = "moderate credit utilization"
    
    # Build profile description
    profile.append(f"- Average Balance: ${mean_balance:,.2f}")
    profile.append(f"- Average Purchases: ${mean_purchases:,.2f}")
    profile.append(f"- Average Payments: ${mean_payments:,.2f}")
    profile.append(f"- Average Credit Limit: ${mean_credit_limit:,.2f}")
    profile.append(f"- Credit Utilization: {credit_utilization:.1f}%")
    profile.append(f"- Spending Behavior: {spending.title()}")
    profile.append(f"- Credit Status: {credit_status.title()}")
    
    return "\n".join(profile)

def generate_recommendations(cluster_data: pd.DataFrame) -> str:
    """Generate recommendations based on cluster characteristics."""
    recommendations = []
    
    # Calculate key metrics
    credit_utilization = (cluster_data['BALANCE'].mean() / cluster_data['CREDIT_LIMIT'].mean()) * 100
    payment_ratio = cluster_data['PAYMENTS'].mean() / cluster_data['PURCHASES'].mean()
    
    # Generate recommendations based on metrics
    if credit_utilization > 70:
        recommendations.append("- Consider offering balance transfer options or debt consolidation programs")
        recommendations.append("- Provide financial education about credit utilization")
    elif credit_utilization < 30:
        recommendations.append("- Opportunity for credit limit increase offers")
        recommendations.append("- Promote rewards programs to encourage more card usage")
    
    if payment_ratio < 0.8:
        recommendations.append("- Monitor for potential payment difficulties")
        recommendations.append("- Consider offering flexible payment plans")
    elif payment_ratio > 1.2:
        recommendations.append("- Potential for premium card upgrades")
        recommendations.append("- Offer investment or savings products")
    
    if cluster_data['PURCHASES'].mean() > 10000:
        recommendations.append("- Target with premium rewards programs")
        recommendations.append("- Offer exclusive shopping benefits")
    elif cluster_data['PURCHASES'].mean() < 1000:
        recommendations.append("- Implement activation campaigns")
        recommendations.append("- Consider cashback offers to increase card usage")
    
    return "\n".join(recommendations)

def generate_report(
    data: pd.DataFrame,
    labels: np.ndarray,
    silhouette_score: float,
    preprocessing_steps: List[str]
) -> str:
    """Generate a comprehensive analysis report."""
    # Create report sections
    sections = []
    
    # Add header
    header = REPORT_TEMPLATES['header'].format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    sections.append(header)
    
    # Add preprocessing information
    preprocessing = REPORT_TEMPLATES['preprocessing'].format(
        n_customers=len(data),
        features=", ".join(data.columns),
        preprocessing_steps=", ".join(preprocessing_steps)
    )
    sections.append(preprocessing)
    
    # Add clustering results
    clustering = REPORT_TEMPLATES['clustering'].format(
        n_clusters=len(np.unique(labels)),
        silhouette_score=silhouette_score
    )
    sections.append(clustering)
    
    # Add cluster profiles and recommendations
    sections.append("## Cluster Profiles and Recommendations")
    
    for cluster_id in range(len(np.unique(labels))):
        cluster_data = data[labels == cluster_id]
        sections.append(f"\n### Cluster {cluster_id} ({len(cluster_data)} customers)")
        
        # Add cluster profile
        sections.append("#### Profile")
        sections.append(generate_cluster_profile(cluster_data))
        
        # Add recommendations
        sections.append("\n#### Recommendations")
        sections.append(generate_recommendations(cluster_data))
        
        sections.append("\n---")
    
    # Combine all sections
    report = "\n\n".join(sections)
    
    return report

def save_report(report: str, filename: str = None) -> str:
    """Save the generated report to a file."""
    if filename is None:
        filename = f"segmentation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    # Create reports directory if it doesn't exist
    os.makedirs(PATHS['reports'], exist_ok=True)
    
    # Save report
    report_path = os.path.join(PATHS['reports'], filename)
    with open(report_path, 'w') as f:
        f.write(report)
    
    return report_path 