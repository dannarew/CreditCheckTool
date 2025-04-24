"""
Module for data preprocessing in the credit card segmentation application.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from typing import Tuple, List, Dict, Any
from .config import NUMERICAL_FEATURES

def validate_data(data: pd.DataFrame) -> Tuple[bool, str]:
    """Validate the input data for required columns and data types."""
    required_columns = set(NUMERICAL_FEATURES)
    
    # Check for required columns
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check data types
    for column in required_columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            return False, f"Column {column} must be numeric"
    
    return True, "Data validation successful"

def handle_missing_values(
    data: pd.DataFrame,
    strategy: str = 'mean'
) -> Tuple[pd.DataFrame, List[str]]:
    """Handle missing values in the dataset."""
    df = data.copy()
    steps = []
    
    for column in NUMERICAL_FEATURES:
        missing_count = df[column].isnull().sum()
        if missing_count > 0:
            if strategy == 'mean':
                df[column].fillna(df[column].mean(), inplace=True)
                steps.append(f"Filled {missing_count} missing values in {column} with mean")
            elif strategy == 'median':
                df[column].fillna(df[column].median(), inplace=True)
                steps.append(f"Filled {missing_count} missing values in {column} with median")
            elif strategy == 'zero':
                df[column].fillna(0, inplace=True)
                steps.append(f"Filled {missing_count} missing values in {column} with zero")
    
    return df, steps

def handle_outliers(
    data: pd.DataFrame,
    method: str = 'iqr',
    threshold: float = 1.5
) -> Tuple[pd.DataFrame, List[str]]:
    """Handle outliers in the dataset."""
    df = data.copy()
    steps = []
    
    for column in NUMERICAL_FEATURES:
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
            if len(outliers) > 0:
                df.loc[df[column] < lower_bound, column] = lower_bound
                df.loc[df[column] > upper_bound, column] = upper_bound
                steps.append(f"Capped {len(outliers)} outliers in {column} using IQR method")
        
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = df[z_scores > threshold][column]
            if len(outliers) > 0:
                df.loc[z_scores > threshold, column] = df[column].mean()
                steps.append(f"Replaced {len(outliers)} outliers in {column} using Z-score method")
    
    return df, steps

def scale_features(
    data: pd.DataFrame,
    method: str = 'standard'
) -> Tuple[pd.DataFrame, List[str]]:
    """Scale numerical features using the specified method."""
    df = data.copy()
    steps = []
    
    if method == 'standard':
        scaler = StandardScaler()
        steps.append("Applied StandardScaler normalization")
    elif method == 'robust':
        scaler = RobustScaler()
        steps.append("Applied RobustScaler normalization")
    elif method == 'minmax':
        scaler = MinMaxScaler()
        steps.append("Applied MinMaxScaler normalization")
    else:
        return df, ["No scaling applied"]
    
    # Scale numerical features
    df[NUMERICAL_FEATURES] = scaler.fit_transform(df[NUMERICAL_FEATURES])
    
    return df, steps

def preprocess_data(
    data: pd.DataFrame,
    handle_missing: bool = True,
    handle_outliers_flag: bool = True,
    scaling_method: str = 'standard'
) -> Tuple[pd.DataFrame, List[str]]:
    """Preprocess the data by applying various cleaning and transformation steps."""
    df = data.copy()
    preprocessing_steps = []
    
    # Validate data
    is_valid, message = validate_data(df)
    if not is_valid:
        raise ValueError(message)
    
    # Handle missing values
    if handle_missing:
        df, missing_steps = handle_missing_values(df)
        preprocessing_steps.extend(missing_steps)
    
    # Handle outliers
    if handle_outliers_flag:
        df, outlier_steps = handle_outliers(df)
        preprocessing_steps.extend(outlier_steps)
    
    # Scale features
    if scaling_method != 'none':
        df, scaling_steps = scale_features(df, scaling_method)
        preprocessing_steps.extend(scaling_steps)
    
    return df, preprocessing_steps 