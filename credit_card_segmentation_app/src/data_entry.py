import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from .database import save_customers_to_db, get_customers_from_db

def create_data_entry_form() -> Optional[pd.DataFrame]:
    """Create a form for manual data entry."""
    st.header("Manual Data Entry")
    
    # Initialize session state for form data
    if 'form_data' not in st.session_state:
        st.session_state.form_data = []
    
    # Form for single customer entry
    with st.form("customer_entry_form"):
        st.subheader("Enter Customer Data")
        
        # Input fields
        customer_id = st.text_input("Customer ID", "")
        balance = st.number_input("Balance", min_value=0.0, value=0.0, step=100.0)
        purchases = st.number_input("Purchases", min_value=0.0, value=0.0, step=100.0)
        payments = st.number_input("Payments", min_value=0.0, value=0.0, step=100.0)
        credit_limit = st.number_input("Credit Limit", min_value=0.0, value=0.0, step=100.0)
        minimum_payments = st.number_input("Minimum Payments", min_value=0.0, value=0.0, step=10.0)
        
        # Submit button
        submitted = st.form_submit_button("Add Customer")
        
        if submitted and customer_id:
            # Add to session state
            st.session_state.form_data.append({
                'CUSTOMER_ID': customer_id,
                'BALANCE': balance,
                'PURCHASES': purchases,
                'PAYMENTS': payments,
                'CREDIT_LIMIT': credit_limit,
                'MINIMUM_PAYMENTS': minimum_payments
            })
            st.success(f"Customer {customer_id} added successfully!")
            # Clear form
            st.experimental_rerun()
    
    # Display entered data
    if st.session_state.form_data:
        st.subheader("Entered Data")
        df = pd.DataFrame(st.session_state.form_data)
        st.dataframe(df)
        
        # Save to database button (outside form)
        save_to_db = st.button("Save to Database")
        if save_to_db:
            if save_customers_to_db(df):
                st.success("Data saved to database successfully!")
                # Clear session state
                st.session_state.form_data = []
                st.experimental_rerun()
            else:
                st.error("Failed to save data to database.")
    
    # Bulk data entry
    st.subheader("Bulk Data Entry")
    st.write("Upload a CSV file with multiple customer records")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            # Save to database button (outside form)
            save_uploaded = st.button("Save Uploaded Data to Database")
            if save_uploaded:
                if save_customers_to_db(df):
                    st.success("Data saved to database successfully!")
                else:
                    st.error("Failed to save data to database.")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # View database data
    st.subheader("View Database Data")
    view_data = st.button("Load Data from Database")
    if view_data:
        df = get_customers_from_db()
        if df is not None and not df.empty:
            st.dataframe(df)
        else:
            st.info("No data found in the database.")
    
    return None

def create_batch_data_generator() -> Optional[pd.DataFrame]:
    """Create a form for generating batch data."""
    st.header("Generate Sample Data")
    
    # Form for batch data generation
    with st.form("batch_data_form"):
        st.subheader("Generate Multiple Customer Records")
        
        # Input fields
        num_customers = st.number_input("Number of Customers", min_value=1, max_value=1000, value=100, step=10)
        
        # Distribution parameters
        st.write("Distribution Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            balance_mean = st.number_input("Balance Mean", min_value=0.0, value=1500.0, step=100.0)
            balance_std = st.number_input("Balance Std", min_value=0.0, value=500.0, step=50.0)
            purchases_mean = st.number_input("Purchases Mean", min_value=0.0, value=1000.0, step=100.0)
            purchases_std = st.number_input("Purchases Std", min_value=0.0, value=300.0, step=50.0)
        
        with col2:
            payments_mean = st.number_input("Payments Mean", min_value=0.0, value=800.0, step=100.0)
            payments_std = st.number_input("Payments Std", min_value=0.0, value=200.0, step=50.0)
            credit_limit_mean = st.number_input("Credit Limit Mean", min_value=0.0, value=2000.0, step=100.0)
            credit_limit_std = st.number_input("Credit Limit Std", min_value=0.0, value=500.0, step=50.0)
        
        minimum_payments_mean = st.number_input("Minimum Payments Mean", min_value=0.0, value=100.0, step=10.0)
        minimum_payments_std = st.number_input("Minimum Payments Std", min_value=0.0, value=30.0, step=5.0)
        
        # Submit button
        submitted = st.form_submit_button("Generate Data")
        
        if submitted:
            # Generate data
            data = {
                'CUSTOMER_ID': [f"CUST_{i+1}" for i in range(num_customers)],
                'BALANCE': np.random.normal(balance_mean, balance_std, num_customers),
                'PURCHASES': np.random.normal(purchases_mean, purchases_std, num_customers),
                'PAYMENTS': np.random.normal(payments_mean, payments_std, num_customers),
                'CREDIT_LIMIT': np.random.normal(credit_limit_mean, credit_limit_std, num_customers),
                'MINIMUM_PAYMENTS': np.random.normal(minimum_payments_mean, minimum_payments_std, num_customers)
            }
            
            # Ensure all values are positive
            for key in ['BALANCE', 'PURCHASES', 'PAYMENTS', 'CREDIT_LIMIT', 'MINIMUM_PAYMENTS']:
                data[key] = np.abs(data[key])
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Display generated data
            st.subheader("Generated Data")
            st.dataframe(df.head())
            
            # Save to database button (outside form)
            save_generated = st.button("Save Generated Data to Database")
            if save_generated:
                if save_customers_to_db(df):
                    st.success("Data saved to database successfully!")
                else:
                    st.error("Failed to save data to database.")
            
            return df
    
    return None 