import pandas as pd
import streamlit as st
from typing import Optional, Tuple
import numpy as np

def load_sample_data() -> pd.DataFrame:
    """Load sample credit card data."""
    # This is a placeholder - you should replace with actual sample data
    data = {
        'CUSTOMER_ID': range(1, 1001),
        'BALANCE': np.random.normal(1500, 500, 1000),
        'PURCHASES': np.random.normal(1000, 300, 1000),
        'PAYMENTS': np.random.normal(800, 200, 1000),
        'CREDIT_LIMIT': np.random.normal(2000, 500, 1000),
        'MINIMUM_PAYMENTS': np.random.normal(100, 30, 1000)
    }
    return pd.DataFrame(data)

def load_user_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Load data from user uploaded file."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Please upload a CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def validate_data(df: pd.DataFrame, required_columns: list) -> Tuple[bool, str]:
    """Validate if the dataframe contains required columns."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    return True, "Data validation successful"

def get_data() -> Optional[pd.DataFrame]:
    """Main function to handle data loading and validation."""
    st.sidebar.header("Data Input")
    
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Upload Data", "Use Sample Data"]
    )
    
    if data_source == "Upload Data":
        uploaded_file = st.sidebar.file_uploader(
            "Upload your credit card data (CSV or Excel)",
            type=['csv', 'xlsx', 'xls']
        )
        
        if uploaded_file is not None:
            df = load_user_data(uploaded_file)
        else:
            return None
    else:
        df = load_sample_data()
    
    if df is not None:
        # Display basic data info
        st.sidebar.write("Data Preview:")
        st.sidebar.write(f"Number of rows: {len(df)}")
        st.sidebar.write(f"Number of columns: {len(df.columns)}")
        
        # Validate data
        is_valid, message = validate_data(df, FEATURE_COLUMNS)
        if not is_valid:
            st.error(message)
            return None
            
        return df
    
    return None 