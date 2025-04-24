from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import seaborn as sns

class BaseModel:
    def __init__(self):
        self.model = None
        self.model_name = None
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_prob)
        else:
            auc_score = None
        
        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'auc_score': auc_score
        }
    
    def plot_confusion_matrix(self, conf_matrix):
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

class LogisticRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = LogisticRegression(**kwargs)
        self.model_name = 'Logistic Regression'

class RandomForestModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = RandomForestClassifier(**kwargs)
        self.model_name = 'Random Forest'

class SVMModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = SVC(**kwargs)
        self.model_name = 'SVM'

class KNNModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = KNeighborsClassifier(**kwargs)
        self.model_name = 'KNN'

class NeuralNetworkModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 32], dropout_rate=0.3):
        super(NeuralNetworkModel, self).__init__()
        self.model_name = 'Neural Network'
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(GNNModel, self).__init__()
        self.model_name = 'Graph Neural Network'
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)

def train_model(model, X_train, y_train, X_test, y_test):
    """
    Train and evaluate a model.
    
    Args:
        model: An instance of BaseModel or its subclasses
        X_train, y_train: Training data
        X_test, y_test: Test data
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    evaluation = model.evaluate(X_test, y_test)
    
    # Plot confusion matrix
    model.plot_confusion_matrix(evaluation['confusion_matrix'])
    
    return evaluation 