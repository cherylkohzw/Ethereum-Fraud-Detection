"""
Feature mapping module for Ethereum fraud detection.
This module provides functionality to standardize and process features from different Ethereum datasets.

Key responsibilities:
1. Maps features from different data sources to standardized names
2. Validates feature types and values
3. Calculates statistics for features
4. Processes and merges transaction and features datasets

The module is designed to handle two types of input datasets:
- Transaction dataset: Contains transaction-related features
- Features dataset: Contains address-based features
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

class FeatureMapper:
    """
    Maps standard fraud detection features to Ethereum features.
    
    This class handles the standardization of feature names across different datasets
    and provides functionality for feature validation and statistics calculation.
    
    The FEATURE_MAPPINGS dictionary defines how features from different sources
    map to standardized names. Each key is the standardized name, and the value
    is a list containing [transaction_dataset_name, features_dataset_name].
    """
    
    # Standard feature mappings between transaction and features datasets
    FEATURE_MAPPINGS = {
        # Core fraud detection features
        'fraud_label': ['FLAG', 'flag'],  # Binary indicator of fraudulent activity
        'transaction_count': ['total transactions', 'totalTransactions'],  # Number of transactions
        'account_balance': ['total ether balance', 'totalEtherBalance'],  # Current balance
        
        # Transaction patterns - Temporal features
        'transaction_frequency_sent': ['Avg min between sent tnx', 'avgTimeBetweenSentTnx'],  # Time between sent transactions
        'transaction_frequency_received': ['Avg min between received tnx', 'avgTimeBetweenRecTnx'],  # Time between received transactions
        'unique_contacts_sent': ['Unique Sent To Addresses', 'numUniqSentAddress'],  # Number of unique addresses sent to
        'unique_contacts_received': ['Unique Received From Addresses', 'numUniqRecAddress'],  # Number of unique addresses received from
        
        # Transaction values - Statistical features
        'min_transaction_sent': ['min val sent', 'minValSent'],  # Minimum value sent
        'max_transaction_sent': ['max val sent', 'maxValSent'],  # Maximum value sent
        'avg_transaction_sent': ['avg val sent', 'avgValSent'],  # Average value sent
        'min_transaction_received': ['min value received', 'minValReceived'],  # Minimum value received
        'max_transaction_received': ['max value received', 'maxValReceived'],  # Maximum value received
        'avg_transaction_received': ['avg val received', 'avgValReceived'],  # Average value received
        
        # Activity metrics - Behavioral features
        'total_sent': ['total Ether sent', 'totalEtherSent'],  # Total amount sent
        'total_received': ['total ether received', 'totalEtherReceived'],  # Total amount received
        'contract_interaction': ['total ether sent contracts', 'totalEtherSentContracts'],  # Interaction with smart contracts
        'contract_creation': ['Number of Created Contracts', 'createdContracts']  # Number of contracts created
    }

    def __init__(self):
        """
        Initialize the FeatureMapper.
        
        Creates empty dictionaries to store:
        - mapped_features: Mapping between original and standardized feature names
        - feature_stats: Statistics calculated for each feature
        """
        self.mapped_features = {}
        self.feature_stats = {}

    def map_features(self, df: pd.DataFrame, feature_type: str) -> pd.DataFrame:
        """
        Map features from original dataset to standardized names.
        
        This method performs the following steps:
        1. Creates a new empty DataFrame for standardized features
        2. Iterates through the FEATURE_MAPPINGS dictionary
        3. Maps features from the source dataset to standardized names
        4. Preserves the original data while renaming columns
        
        Args:
            df (pd.DataFrame): Input DataFrame with original feature names
            feature_type (str): Type of dataset ('transaction' or 'features')
            
        Returns:
            pd.DataFrame: DataFrame with standardized feature names
        """
        standardized_df = pd.DataFrame()
        
        for new_col, [trans_col, feat_col] in self.FEATURE_MAPPINGS.items():
            source_col = trans_col if feature_type == 'transaction' else feat_col
            if source_col in df.columns:
                standardized_df[new_col] = df[source_col]
                
        return standardized_df

    def get_feature_stats(self, df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive statistics for each feature.
        
        This method analyzes each feature and calculates:
        1. Basic information: data type, missing values, unique values
        2. Numerical statistics: min, max, mean, standard deviation
        
        These statistics are useful for:
        - Data quality assessment
        - Feature distribution understanding
        - Identifying potential issues or anomalies
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Dict: Dictionary containing feature statistics with structure:
                {feature_name: {
                    'type': data_type,
                    'missing': count_of_missing_values,
                    'unique_values': count_of_unique_values,
                    'min': minimum_value,  # for numeric features
                    'max': maximum_value,  # for numeric features
                    'mean': mean_value,    # for numeric features
                    'std': std_deviation   # for numeric features
                }}
        """
        stats = {}
        for col in df.columns:
            col_stats = {
                'type': str(df[col].dtype),
                'missing': df[col].isnull().sum(),
                'unique_values': df[col].nunique()
            }
            
            if df[col].dtype in ['int64', 'float64']:
                desc = df[col].describe()
                col_stats.update({
                    'min': desc['min'],
                    'max': desc['max'],
                    'mean': desc['mean'],
                    'std': desc['std']
                })
                
        return stats

    def validate_features(self, df: pd.DataFrame) -> List[str]:
        """
        Validate features in the dataset for data quality and consistency.
        
        Performs the following validations:
        1. Checks for presence of required features
        2. Validates data types (e.g., fraud_label should be boolean or integer)
        3. Checks value ranges (e.g., non-negative values for most features)
        
        This validation helps identify:
        - Missing critical features
        - Incorrect data types
        - Invalid value ranges
        - Potential data quality issues
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            List[str]: List of validation messages, empty if all validations pass
        """
        validation_messages = []
        
        # Check for required features
        required_features = ['fraud_label', 'transaction_count', 'account_balance']
        missing_required = [f for f in required_features if f not in df.columns]
        if missing_required:
            validation_messages.append(f"Missing required features: {missing_required}")
            
        # Check data types
        for col in df.columns:
            if col in ['fraud_label']:
                if df[col].dtype not in ['int64', 'bool']:
                    validation_messages.append(f"Invalid type for {col}: {df[col].dtype}")
            elif df[col].dtype not in ['int64', 'float64']:
                validation_messages.append(f"Non-numeric type for {col}: {df[col].dtype}")
                
        # Check value ranges
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if df[col].min() < 0 and 'balance' not in col:
                validation_messages.append(f"Negative values in {col}")
                
        return validation_messages

    def process_dataset(self, transaction_df: pd.DataFrame, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Process both transaction and features datasets to create a standardized dataset.
        
        This is the main processing pipeline that:
        1. Maps features from both datasets to standardized names
        2. Merges the standardized datasets
        3. Validates the merged dataset
        4. Calculates comprehensive feature statistics
        
        The process ensures:
        - Consistent feature naming across datasets
        - Data quality through validation
        - Statistical analysis of features
        - Proper merging of different data sources
        
        Args:
            transaction_df (pd.DataFrame): Transaction dataset
            features_df (pd.DataFrame): Features dataset
            
        Returns:
            Tuple[pd.DataFrame, Dict]: 
                - Standardized and merged DataFrame
                - Dictionary of feature statistics
        """
        # Map features from both datasets
        trans_standardized = self.map_features(transaction_df, 'transaction')
        feat_standardized = self.map_features(features_df, 'features')
        
        # Merge datasets
        merged_df = pd.concat([trans_standardized, feat_standardized], axis=0, ignore_index=True)
        
        # Calculate statistics
        stats = self.get_feature_stats(merged_df)
        
        # Validate features
        validation_messages = self.validate_features(merged_df)
        if validation_messages:
            print("Validation warnings:", validation_messages)
            
        return merged_df, stats 