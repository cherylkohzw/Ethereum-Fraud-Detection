# Ethereum Fraud Detection

This project implements a machine learning pipeline for detecting fraudulent transactions on the Ethereum blockchain.

## Project Structure

```
Ethereum-Fraud-Detection/
├── data/
│   ├── raw/             # Raw transaction and features datasets
│   └── processed/       # Processed datasets ready for modeling
├── notebooks/           # Jupyter notebooks for analysis
├── src/
│   ├── data/           # Data processing scripts
│   ├── features/       # Feature engineering code
│   └── models/         # Model training and evaluation code
└── requirements.txt    # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Processing Pipeline:
```bash
# Step 1: Feature standardization and initial processing
python src/data/process_data1.py

# Step 2: Data cleaning and preparation for modeling
python src/data/ethereum_data_cleaner.py
```

2. Analysis:
- Open Jupyter Notebook:
```bash
jupyter notebook
```
- Navigate to `notebooks/` directory and open the analysis notebook

## Data Processing Pipeline

The data processing pipeline consists of two main steps:

### Step 1: Feature Standardization (`process_data1.py`)
This script serves as the single source of truth for initial data processing.

#### Input Data
- Transaction dataset: `data/raw/transaction_dataset.csv`
- Features dataset: `data/raw/eth_illicit_features.csv`

#### Processing Steps
1. **Data Loading**: 
   - Loads transaction and features datasets
   - Performs basic validation of file existence and readability

2. **Feature Standardization**:
   - Uses the `FeatureMapper` class to standardize feature names
   - Maps Ethereum-specific features to standard fraud detection features
   - Examples:
     - `FLAG/flag` → `fraud_label`
     - `total_ether_balance` → `account_balance`
     - `Unique Sent To Addresses` → `unique_contacts_sent`

3. **Data Quality Checks**:
   - Validates data types
   - Checks for missing values
   - Identifies duplicate records
   - Calculates feature statistics

4. **Output**:
   - Saves the processed dataset as `data/processed/merged_dataset.csv`
   - This standardized dataset is used as input for Step 2

### Step 2: Data Cleaning (`ethereum_data_cleaner.py`)
This script prepares the standardized data for modeling.

#### Processing Steps
1. **Missing Value Handling**:
   - Imputes missing blockchain data using appropriate strategies
   - Uses mean for numeric features and most frequent for categorical features

2. **Class Balancing**:
   - Balances fraudulent vs non-fraudulent transaction distributions using SMOTE
   - Ensures balanced training data for model development

3. **Outlier Treatment**:
   - Handles outliers in transaction amounts and metrics
   - Uses IQR or z-score based methods
   - Preserves important fraud indicators while removing noise

4. **Quality Assessment**:
   - Generates data quality metrics
   - Compares original and cleaned datasets
   - Validates the cleaning process

#### Output
- The final cleaned dataset ready for model training

### Feature Mapping Overview
The pipeline standardizes various types of features:
- Core fraud detection features (e.g., fraud label, account balance)
- Transaction patterns (e.g., frequency, unique contacts)
- Transaction values (e.g., min/max/avg amounts)
- Activity metrics (e.g., total sent/received, contract interactions)

### Note
While there is an exploratory notebook (`notebooks/01_feature_mapping.ipynb`) that shows the development process, all data processing should use the standardized pipeline (`process_data1.py` followed by `ethereum_data_cleaner.py`) to ensure consistency.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 