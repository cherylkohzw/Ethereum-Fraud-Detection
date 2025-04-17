"""
Script to process Ethereum fraud detection datasets using the feature mapper.

This script serves as the main data processing pipeline that:
1. Loads raw transaction and features datasets
2. Standardizes features using the FeatureMapper
3. Performs data quality checks and validation
4. Saves the processed dataset for model training

The script handles:
- Path management using pathlib for cross-platform compatibility
- Data loading and saving operations
- Feature standardization and mapping
- Data quality validation and statistics calculation
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from pathlib import Path
from features.feature_mapping import FeatureMapper

def load_datasets(transaction_path: str, features_path: str) -> tuple:
    """
    Load transaction and features datasets from CSV files.
    
    This function:
    1. Reads the transaction dataset containing transaction-level information
    2. Reads the features dataset containing address-level features
    3. Performs basic validation of file existence and readability
    
    Args:
        transaction_path (str): Path to transaction dataset CSV file
        features_path (str): Path to features dataset CSV file
        
    Returns:
        tuple: (transaction_df, features_df)
            - transaction_df: DataFrame containing transaction data
            - features_df: DataFrame containing address features
            
    Raises:
        FileNotFoundError: If either file doesn't exist
        pd.errors.EmptyDataError: If either file is empty
    """
    transaction_df = pd.read_csv(transaction_path)
    features_df = pd.read_csv(features_path)
    return transaction_df, features_df

def save_processed_data(df: pd.DataFrame, output_path: str):
    """
    Save processed DataFrame to CSV file.
    
    This function:
    1. Creates the output directory if it doesn't exist
    2. Saves the processed DataFrame to CSV format
    3. Handles path creation and file writing operations
    
    The saved dataset will be used for:
    - Model training and validation
    - Feature analysis and selection
    - Data distribution analysis
    
    Args:
        df (pd.DataFrame): Processed DataFrame to save
        output_path (str): Path where the CSV file will be saved
        
    Note:
        The function creates any missing directories in the output path
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

def main():
    """
    Main execution function for the data processing pipeline.
    
    This function orchestrates the entire data processing workflow:
    1. Sets up file paths using pathlib for cross-platform compatibility
    2. Loads raw datasets from the data/raw directory
    3. Initializes the FeatureMapper for standardization
    4. Processes and validates the datasets
    5. Prints feature statistics for analysis
    6. Saves the processed dataset to data/processed directory
    
    The pipeline ensures:
    - Consistent feature naming across datasets
    - Data quality through validation
    - Proper feature mapping and standardization
    - Safe file operations with proper directory structure
    """
    # Define file paths
    base_dir = Path(__file__).parent.parent.parent
    transaction_path = base_dir / "data/raw/transaction_dataset.csv"
    features_path = base_dir / "data/raw/eth_illicit_features.csv"
    output_path = base_dir / "data/processed/standardized_fraud_detection.csv"
    
    # Load datasets
    print("Loading datasets...")
    transaction_df, features_df = load_datasets(str(transaction_path), str(features_path))
    
    # Initialize feature mapper
    mapper = FeatureMapper()
    
    # Process datasets
    print("Processing datasets...")
    processed_df, stats = mapper.process_dataset(transaction_df, features_df)
    
    # Print feature statistics
    print("\nFeature Statistics:")
    for feature, feature_stats in stats.items():
        print(f"\n{feature}:")
        for stat_name, stat_value in feature_stats.items():
            print(f"  {stat_name}: {stat_value}")
    
    # Save processed data
    save_processed_data(processed_df, str(output_path))
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main() 