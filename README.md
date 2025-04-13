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

1. Data Processing:
```bash
python src/data/process_data.py
```

2. Analysis:
- Open Jupyter Notebook:
```bash
jupyter notebook
```
- Navigate to `notebooks/` directory and open the analysis notebook

## Data Processing Pipeline

The data processing pipeline includes:
1. Loading transaction and features datasets
2. Feature engineering using the FeatureMapper class
3. Data cleaning and preprocessing
4. Saving processed datasets for modeling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 