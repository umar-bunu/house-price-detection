import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor

from functions.removeOutliers import remove_outliers
from functions.create_xgboost_model import create_xgboost_model
from functions.create_ensemble_model import create_ensemble_model
from functions.tune_hyperparameters import tune_hyperparameters

def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path)

def prepare_data(data):
    """Separate features and target."""
    X = data.drop('price', axis=1)
    y = data['price']
    return X, y

def create_preprocessor():
    """Create a preprocessor for numeric and categorical data."""
    numeric_features = ['floor', 'buildingAge', 'floorCount', 'room2', 'Baths', 'Size Area (m2)']
    categorical_features = ['area', 'city', 'Property Type']
    
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])


    
def create_model(preprocessor):
    """Create a pipeline with preprocessor and random forest regressor."""
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

def within_range(y_true, y_pred, range_percent=10):
    """Calculate percentage of predictions within ±range_percent of actual values."""
    lower_bound = y_true * (1 - range_percent/100)
    upper_bound = y_true * (1 + range_percent/100)
    within_range = np.logical_and(y_pred >= lower_bound, y_pred <= upper_bound)
    return np.mean(within_range) * 100

def evaluate_model_cv(model, X, y, cv=5):
    """Evaluate the model using cross-validation and print metrics."""
    # Use cross_val_predict to get predictions for all samples
    y_pred = cross_val_predict(model, X, y, cv=cv)
    
    # Calculate MSE and R-squared
    mse = np.mean((y - y_pred)**2)
    r2 = 1 - (np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2))
    
    # Calculate percentage within range
    percent_within_range = within_range(y, y_pred)
    
    print(f"Cross-validation results:")
    print(f"Mean MSE: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    print(f"Percentage within 10% range: {percent_within_range:.2f}%")
    
    return mse, r2, percent_within_range

def main():
    # Load data
    data = load_data('./datasets/dataset-utf8.csv')
    
    # Prepare data
    X, y = prepare_data(data)
    X, y = remove_outliers(X, y, ['Size Area (m2)'])
    # Create preprocessor and model
    preprocessor = create_preprocessor()
    model = create_ensemble_model(preprocessor)
    
    param_grid = {
        'ensemble__rf__n_estimators': [100, 200],
        'ensemble__xgb__n_estimators': [100, 500, 1000],
        'ensemble__xgb__learning_rate': [0.01, 0.05, 0.1],
        'ensemble__lasso__alpha': [0.01, 0.1, 1]
    }
    
    best_model = tune_hyperparameters(model, param_grid, X, y)

    
    # Evaluate model using cross-validation
    mse, r2, percent_within_range = evaluate_model_cv(model, X, y)
    
    # Train the final model on all data
    model.fit(X, y)
    
    return model, mse, r2, percent_within_range

if __name__ == "__main__":
    model, mse, r2, percent_within_range = main()