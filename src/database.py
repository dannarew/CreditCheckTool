"""
Module for handling database operations in the credit card segmentation application.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from .config import DB_CONFIG
import streamlit as st
import psycopg

# Load environment variables
load_dotenv()

# Create SQLAlchemy base class
Base = declarative_base()

class Customer(Base):
    """Customer data model."""
    __tablename__ = 'customers'
    
    id = Column(Integer, primary_key=True)
    customer_id = Column(String(50), unique=True, nullable=False)
    balance = Column(Float, nullable=False)
    purchases = Column(Float, nullable=False)
    payments = Column(Float, nullable=False)
    credit_limit = Column(Float, nullable=False)
    minimum_payments = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

class SegmentationResult(Base):
    """Segmentation results model."""
    __tablename__ = 'segmentation_results'
    
    id = Column(Integer, primary_key=True)
    analysis_date = Column(DateTime, nullable=False)
    n_clusters = Column(Integer, nullable=False)
    silhouette_score = Column(Float, nullable=False)
    preprocessing_steps = Column(JSON, nullable=False)
    cluster_profiles = Column(JSON, nullable=False)
    recommendations = Column(JSON, nullable=False)

def get_database_url() -> str:
    """Get database URL from environment variables or config."""
    db_user = os.getenv('DB_USER', DB_CONFIG['user'])
    db_password = os.getenv('DB_PASSWORD', DB_CONFIG['password'])
    db_host = os.getenv('DB_HOST', DB_CONFIG['host'])
    db_port = os.getenv('DB_PORT', DB_CONFIG['port'])
    db_name = os.getenv('DB_NAME', DB_CONFIG['database'])
    
    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

def get_db_connection():
    """Get a database connection using psycopg3."""
    try:
        conn = psycopg.connect(get_database_url())
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        return None

def init_db() -> None:
    """Initialize the database and create tables."""
    engine = create_engine(get_database_url())
    Base.metadata.create_all(engine)

def get_session():
    """Create a new database session."""
    engine = create_engine(get_database_url())
    Session = sessionmaker(bind=engine)
    return Session()

def save_customers_to_db(data: pd.DataFrame) -> bool:
    """Save customer data to the database."""
    try:
        session = get_session()
        
        for _, row in data.iterrows():
            customer = Customer(
                customer_id=str(row['CUSTOMER_ID']),
                balance=float(row['BALANCE']),
                purchases=float(row['PURCHASES']),
                payments=float(row['PAYMENTS']),
                credit_limit=float(row['CREDIT_LIMIT']),
                minimum_payments=float(row['MINIMUM_PAYMENTS']) if pd.notna(row['MINIMUM_PAYMENTS']) else None
            )
            session.add(customer)
        
        session.commit()
        return True
    
    except Exception as e:
        st.error(f"Error saving customers to database: {str(e)}")
        session.rollback()
        return False
    
    finally:
        session.close()

def get_customers_from_db() -> Optional[pd.DataFrame]:
    """Retrieve customer data from the database."""
    try:
        session = get_session()
        
        # Query all customers
        customers = session.query(Customer).all()
        
        if not customers:
            return None
        
        # Convert to DataFrame
        data = []
        for customer in customers:
            data.append({
                'CUSTOMER_ID': customer.customer_id,
                'BALANCE': customer.balance,
                'PURCHASES': customer.purchases,
                'PAYMENTS': customer.payments,
                'CREDIT_LIMIT': customer.credit_limit,
                'MINIMUM_PAYMENTS': customer.minimum_payments
            })
        
        return pd.DataFrame(data)
    
    except Exception as e:
        st.error(f"Error retrieving customers from database: {str(e)}")
        return None
    
    finally:
        session.close()

def save_segmentation_results(
    data: pd.DataFrame,
    labels: np.ndarray,
    silhouette_score: float,
    preprocessing_steps: List[str] = None
) -> bool:
    """Save segmentation results to the database."""
    try:
        session = get_session()
        
        # Create cluster profiles
        cluster_profiles = {}
        for cluster_id in range(len(np.unique(labels))):
            cluster_data = data[labels == cluster_id]
            profile = {
                'size': len(cluster_data),
                'mean_balance': float(cluster_data['BALANCE'].mean()),
                'mean_purchases': float(cluster_data['PURCHASES'].mean()),
                'mean_payments': float(cluster_data['PAYMENTS'].mean()),
                'mean_credit_limit': float(cluster_data['CREDIT_LIMIT'].mean()),
                'mean_minimum_payments': float(cluster_data['MINIMUM_PAYMENTS'].mean())
            }
            cluster_profiles[str(cluster_id)] = profile
        
        # Create recommendations (placeholder)
        recommendations = {
            str(cluster_id): f"Recommendations for cluster {cluster_id}"
            for cluster_id in range(len(np.unique(labels)))
        }
        
        # Create segmentation result
        result = SegmentationResult(
            analysis_date=datetime.now(),
            n_clusters=len(np.unique(labels)),
            silhouette_score=float(silhouette_score),
            preprocessing_steps=preprocessing_steps or [],
            cluster_profiles=cluster_profiles,
            recommendations=recommendations
        )
        
        session.add(result)
        session.commit()
        return True
    
    except Exception as e:
        st.error(f"Error saving segmentation results: {str(e)}")
        session.rollback()
        return False
    
    finally:
        session.close()

def get_segmentation_history() -> Optional[pd.DataFrame]:
    """Retrieve segmentation history from the database."""
    try:
        session = get_session()
        
        # Query all segmentation results
        results = session.query(SegmentationResult).all()
        
        if not results:
            return None
        
        # Convert to DataFrame
        data = []
        for result in results:
            data.append({
                'analysis_date': result.analysis_date,
                'n_clusters': result.n_clusters,
                'silhouette_score': result.silhouette_score,
                'preprocessing_steps': result.preprocessing_steps,
                'cluster_profiles': result.cluster_profiles,
                'recommendations': result.recommendations
            })
        
        return pd.DataFrame(data)
    
    except Exception as e:
        st.error(f"Error retrieving segmentation history: {str(e)}")
        return None
    
    finally:
        session.close() 