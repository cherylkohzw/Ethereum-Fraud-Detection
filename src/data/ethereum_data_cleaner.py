"""
Ethereum transaction data cleaning module for fraud detection.

This module:
1. Defines the EthereumDataCleaner class for data preprocessing
2. Provides functionality to clean Ethereum transaction and feature data:
   - Missing value imputation for transaction and account features
   - Balancing fraudulent vs non-fraudulent transaction distributions
   - Outlier treatment for transaction amounts and metrics
3. Includes data quality assessment and benchmarking
4. Includes script to execute the cleaning pipeline on the dataset
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

class EthereumDataCleaner:
    """
    Specialized data cleaning class for Ethereum fraud detection datasets.
    Handles transaction and account feature preprocessing including:
    - Missing blockchain data imputation
    - Fraudulent transaction class balancing
    - Transaction amount outlier handling
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the EthereumDataCleaner.
        
        Args:
            random_state (int): Random seed for reproducibility in SMOTE and sampling
        """
        self.random_state = random_state
        self.imputer_dict = {}  # Store imputers for each blockchain feature
        self.feature_stats = {}  # Store transaction statistics for transformations
    
    def handle_missing_blockchain_data(self, df: pd.DataFrame, strategy_dict: Dict[str, str] = None) -> pd.DataFrame:
        """
        Handle missing values in blockchain transaction and account data.
        
        Args:
            df (pd.DataFrame): Input DataFrame with blockchain features
            strategy_dict (Dict[str, str]): Dictionary mapping feature names to imputation strategies
                                          ('mean', 'median', 'most_frequent', or 'drop')
                                          If None, uses 'mean' for numeric and 'most_frequent' for categorical
        
        Returns:
            pd.DataFrame: DataFrame with imputed blockchain data
        """
        df_processed = df.copy()
        
        if strategy_dict is None:
            strategy_dict = {}
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
            
            for col in numeric_cols:
                strategy_dict[col] = 'mean'
            for col in categorical_cols:
                strategy_dict[col] = 'most_frequent'
        
        # Handle columns marked for dropping
        drop_cols = [col for col, strategy in strategy_dict.items() if strategy == 'drop']
        if drop_cols:
            df_processed = df_processed.drop(columns=drop_cols)
        
        # Impute remaining columns
        for col, strategy in strategy_dict.items():
            if strategy != 'drop' and col in df_processed.columns:
                imputer = SimpleImputer(strategy=strategy)
                df_processed[col] = imputer.fit_transform(df_processed[[col]])
                self.imputer_dict[col] = imputer
        
        return df_processed
    
    def balance_fraud_classes(self, X: pd.DataFrame, y: pd.Series,
                          sampling_strategy: float = 1.0) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance fraudulent and non-fraudulent transaction classes using SMOTE.
        
        Args:
            X (pd.DataFrame): Transaction and account features
            y (pd.Series): Fraud labels
            sampling_strategy (float): Desired ratio of fraudulent to non-fraudulent transactions
                                     1.0 means equal distribution
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Balanced features and fraud labels
        """
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=y.name)
    
    def handle_transaction_outliers(self, df: pd.DataFrame, method: str = 'iqr',
                                threshold: float = 1.5, exclude_cols: List[str] = None) -> pd.DataFrame:
        """
        Handle outliers in transaction amounts and metrics using a more robust approach.
        
        Args:
            df (pd.DataFrame): Input DataFrame with transaction features
            method (str): Method to handle outliers
                         'iqr': Remove based on IQR (default)
                         'zscore': Remove based on z-score
            threshold (float): Threshold for outlier detection (default 1.5 for standard IQR rule)
            exclude_cols (List[str]): Features to exclude from outlier handling
        
        Returns:
            pd.DataFrame: DataFrame with handled transaction outliers
        """
        df_processed = df.copy()
        if exclude_cols is None:
            exclude_cols = []
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        cols_to_process = [col for col in numeric_cols if col not in exclude_cols]
        
        if method == 'iqr':
            # Process each column independently
            for col in cols_to_process:
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Remove outliers instead of capping
                df_processed = df_processed[
                    (df_processed[col] >= lower_bound) &
                    (df_processed[col] <= upper_bound)
                ]
        
        elif method == 'zscore':
            # Remove outliers based on z-score
            for col in cols_to_process:
                z_scores = np.abs(stats.zscore(df_processed[col]))
                df_processed = df_processed[z_scores < threshold]
        
        return df_processed
    
    def clean_ethereum_data(self, df: pd.DataFrame, target_col: str = 'fraud_label',
                        imputation_strategy: Dict[str, str] = None,
                        sampling_strategy: float = 1.0,
                        outlier_method: str = 'iqr',
                        outlier_threshold: float = 1.5,
                        exclude_from_outlier: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete Ethereum data cleaning pipeline.
        
        Args:
            df (pd.DataFrame): Raw Ethereum transaction and feature data
            target_col (str): Name of fraud label column
            imputation_strategy (Dict[str, str]): Imputation strategy for blockchain features
            sampling_strategy (float): SMOTE sampling strategy for fraud classes
            outlier_method (str): Method for handling outliers ('iqr' or 'zscore')
            outlier_threshold (float): Threshold for outlier detection (1.5 for IQR rule)
            exclude_from_outlier (List[str]): Features to exclude from outlier handling
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Cleaned features and fraud labels
        """
        # Separate features and fraud labels
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle missing blockchain data first
        X = self.handle_missing_blockchain_data(X, imputation_strategy)
        
        # Handle outliers before SMOTE to ensure clean base data
        X_cleaned = self.handle_transaction_outliers(
            X, 
            method=outlier_method,
            threshold=outlier_threshold,
            exclude_cols=exclude_from_outlier
        )
        
        # Update y to match cleaned X
        y_cleaned = y.loc[X_cleaned.index]
        
        # Apply SMOTE after cleaning
        X_balanced, y_balanced = self.balance_fraud_classes(X_cleaned, y_cleaned, sampling_strategy)
        
        return X_balanced, y_balanced

    def assess_data_quality(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame,
                          target_col: str = 'fraud_label') -> Dict:
        """
        Assess the quality of data cleaning by comparing original and cleaned datasets.
        
        Evaluates:
        1. Missing value statistics
        2. Class balance metrics
        3. Outlier statistics
        4. Distribution changes
        5. Feature correlations
        
        Args:
            original_df (pd.DataFrame): Original dataset before cleaning
            cleaned_df (pd.DataFrame): Cleaned dataset after preprocessing
            target_col (str): Name of the target column (fraud label)
            
        Returns:
            Dict: Dictionary containing quality metrics and comparisons
        """
        quality_report = {}
        
        # 1. Missing Value Analysis
        quality_report['missing_values'] = {
            'original': original_df.isnull().sum().to_dict(),
            'cleaned': cleaned_df.isnull().sum().to_dict(),
            'improvement': {
                col: {
                    'original_pct': (original_df[col].isnull().sum() / len(original_df)) * 100,
                    'cleaned_pct': (cleaned_df[col].isnull().sum() / len(cleaned_df)) * 100
                }
                for col in original_df.columns
            }
        }
        
        # 2. Class Balance Analysis
        quality_report['class_balance'] = {
            'original': original_df[target_col].value_counts(normalize=True).to_dict(),
            'cleaned': cleaned_df[target_col].value_counts(normalize=True).to_dict()
        }
        
        # 3. Outlier Analysis using same threshold as cleaning
        def count_outliers(df: pd.DataFrame, col: str) -> int:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            return len(df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)])
        
        numeric_cols = original_df.select_dtypes(include=['int64', 'float64']).columns
        quality_report['outliers'] = {
            col: {
                'original_count': count_outliers(original_df, col),
                'cleaned_count': count_outliers(cleaned_df, col),
                'reduction_pct': ((count_outliers(original_df, col) - count_outliers(cleaned_df, col)) / 
                                count_outliers(original_df, col) * 100) if count_outliers(original_df, col) > 0 else 0
            }
            for col in numeric_cols if col != target_col
        }
        
        # 4. Distribution Analysis
        quality_report['distribution_stats'] = {
            col: {
                'original': {
                    'mean': original_df[col].mean(),
                    'std': original_df[col].std(),
                    'skew': original_df[col].skew(),
                    'kurtosis': original_df[col].kurtosis()
                },
                'cleaned': {
                    'mean': cleaned_df[col].mean(),
                    'std': cleaned_df[col].std(),
                    'skew': cleaned_df[col].skew(),
                    'kurtosis': cleaned_df[col].kurtosis()
                }
            }
            for col in numeric_cols if col != target_col
        }
        
        # 5. Correlation Analysis
        original_corr = original_df.corr()[target_col].sort_values(ascending=False)
        cleaned_corr = cleaned_df.corr()[target_col].sort_values(ascending=False)
        
        quality_report['feature_correlations'] = {
            'original': original_corr.to_dict(),
            'cleaned': cleaned_corr.to_dict(),
            'changes': {
                col: {
                    'original_corr': original_corr[col],
                    'cleaned_corr': cleaned_corr[col],
                    'difference': cleaned_corr[col] - original_corr[col]
                }
                for col in original_corr.index if col != target_col
            }
        }
        
        return quality_report

    def plot_quality_metrics(self, quality_report: Dict, output_dir: str):
        """
        Generate visualizations for data quality metrics.
        
        Args:
            quality_report (Dict): Quality metrics from assess_data_quality
            output_dir (str): Directory to save the plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Missing Values Plot
        plt.figure(figsize=(12, 6))
        missing_data = pd.DataFrame(quality_report['missing_values']['improvement']).T
        missing_data[['original_pct', 'cleaned_pct']].plot(kind='bar')
        plt.title('Missing Values: Original vs Cleaned')
        plt.ylabel('Percentage Missing')
        plt.tight_layout()
        plt.savefig(output_path / 'missing_values.png')
        plt.close()
        
        # 2. Class Distribution Plot
        plt.figure(figsize=(8, 6))
        class_dist = pd.DataFrame({
            'Original': quality_report['class_balance']['original'],
            'Cleaned': quality_report['class_balance']['cleaned']
        })
        class_dist.plot(kind='bar')
        plt.title('Class Distribution: Original vs Cleaned')
        plt.ylabel('Proportion')
        plt.tight_layout()
        plt.savefig(output_path / 'class_distribution.png')
        plt.close()
        
        # 3. Outlier Reduction Plot
        plt.figure(figsize=(12, 6))
        outlier_data = pd.DataFrame(quality_report['outliers']).T
        outlier_data[['original_count', 'cleaned_count']].plot(kind='bar')
        plt.title('Outlier Counts: Original vs Cleaned')
        plt.ylabel('Number of Outliers')
        plt.tight_layout()
        plt.savefig(output_path / 'outlier_reduction.png')
        plt.close()
        
        # 4. Correlation Changes Plot
        plt.figure(figsize=(12, 8))
        corr_changes = pd.DataFrame(quality_report['feature_correlations']['changes']).T
        sns.heatmap(corr_changes[['original_corr', 'cleaned_corr']], 
                   annot=True, cmap='RdYlBu', center=0)
        plt.title('Feature Correlations: Original vs Cleaned')
        plt.tight_layout()
        plt.savefig(output_path / 'correlation_changes.png')
        plt.close()

def main():
    """
    Main execution function for Ethereum data cleaning pipeline.
    """
    # Define file paths
    base_dir = Path(__file__).parent.parent.parent
    input_path = base_dir / "data/processed/merged_dataset.csv"
    output_path = base_dir / "data/processed/cleaned_ethereum_fraud_data.csv"
    quality_report_path = base_dir / "data/processed/quality_assessment"
    
    # Load merged Ethereum dataset
    print("Loading Ethereum transaction and feature data...")
    eth_data = pd.read_csv(input_path)
    
    # Print missing value information
    print("\nMissing Value Analysis:")
    missing_counts = eth_data.isnull().sum()
    missing_percentages = (eth_data.isnull().sum() / len(eth_data)) * 100
    
    print("\nColumns with missing values:")
    for col in eth_data.columns:
        if missing_counts[col] > 0:
            print(f"{col}: {missing_counts[col]} missing values ({missing_percentages[col]:.2f}%)")
    
    if missing_counts.sum() == 0:
        print("No missing values found in any column!")
    
    # Initialize Ethereum data cleaner
    cleaner = EthereumDataCleaner(random_state=42)
    
    # Define cleaning strategies for blockchain features
    blockchain_cleaning_strategy = {
        'transaction_count': 'mean',
        'account_balance': 'mean',
        'transaction_frequency_sent': 'mean',
        'transaction_frequency_received': 'mean',
        'unique_contacts_sent': 'mean',
        'unique_contacts_received': 'mean',
        'min_transaction_sent': 'mean',
        'max_transaction_sent': 'mean',
        'avg_transaction_sent': 'mean',
        'min_transaction_received': 'mean',
        'max_transaction_received': 'mean',
        'avg_transaction_received': 'mean',
        'total_sent': 'mean',
        'total_received': 'mean',
        'contract_interaction': 'most_frequent',  # Changed to most_frequent for binary
        'contract_creation': 'most_frequent'      # Changed to most_frequent for binary
    }
    
    # Features to exclude from outlier treatment (binary and count-based features)
    exclude_from_outliers = [
        'fraud_label',
        'contract_creation',
        'contract_interaction',
        'transaction_count',
        'unique_contacts_sent',
        'unique_contacts_received',
        'transaction_frequency_sent',
        'transaction_frequency_received'
    ]
    
    print("Cleaning Ethereum transaction data...")
    # Apply cleaning pipeline with improved parameters
    X_cleaned, y_cleaned = cleaner.clean_ethereum_data(
        df=eth_data,
        target_col='fraud_label',
        imputation_strategy=blockchain_cleaning_strategy,
        sampling_strategy=1.0,  # Equal distribution of fraud/non-fraud
        outlier_method='iqr',  # Use IQR-based removal
        outlier_threshold=1.5,  # Standard IQR rule
        exclude_from_outlier=exclude_from_outliers
    )
    
    # Combine cleaned features and fraud labels
    cleaned_eth_data = pd.concat([X_cleaned, y_cleaned], axis=1)
    
    # Assess data quality
    print("\nAssessing data quality...")
    quality_report = cleaner.assess_data_quality(eth_data, cleaned_eth_data)
    
    # Generate quality assessment plots
    print("Generating quality assessment visualizations...")
    cleaner.plot_quality_metrics(quality_report, str(quality_report_path))
    
    # Save cleaned dataset
    print("Saving cleaned Ethereum dataset...")
    cleaned_eth_data.to_csv(output_path, index=False)
    
    # Print cleaning summary with quality metrics
    print("\nEthereum Data Cleaning Summary:")
    print(f"Original dataset shape: {eth_data.shape}")
    print(f"Cleaned dataset shape: {cleaned_eth_data.shape}")
    print(f"Fraud distribution in cleaned dataset:\n{y_cleaned.value_counts(normalize=True)}")
    
    print("\nQuality Assessment Highlights:")
    print("1. Missing Values:")
    for col, stats in quality_report['missing_values']['improvement'].items():
        if stats['original_pct'] > 0:
            print(f"  - {col}: {stats['original_pct']:.2f}% → {stats['cleaned_pct']:.2f}%")
    
    print("\n2. Outlier Reduction:")
    for col, stats in quality_report['outliers'].items():
        if stats['reduction_pct'] > 0:
            print(f"  - {col}: {stats['reduction_pct']:.2f}% reduction")
    
    print("\n3. Feature Correlation Changes:")
    for feat, stats in quality_report['feature_correlations']['changes'].items():
        if abs(stats['difference']) > 0.1:
            print(f"  - {feat}: {stats['original_corr']:.3f} → {stats['cleaned_corr']:.3f}")
    
    print(f"\nDetailed quality assessment plots saved to: {quality_report_path}")
    print("\nEthereum data cleaning complete!")

if __name__ == "__main__":
    main() 