import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

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
    numeric_features = ['area', 'floor', 'buildingAge', 'floorCount', 'room2', 'Baths', 'Size Area (m2)']
    categorical_features = ['city', 'Property Type']
    
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

def compare_models(X, y):
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Elastic Net': ElasticNet(random_state=42),
        'SVR': SVR(kernel='rbf')
    }
    
    preprocessor = create_preprocessor()
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-scores)
        print(f"{name} - Mean RMSE: {rmse.mean():.4f} (+/- {rmse.std() * 2:.4f})")

def main():
    # Load and prepare data as before
    data = load_data('./datasets/dataset-utf8.csv')
    X, y = prepare_data(data)
    
    # Compare different models
    compare_models(X, y)
    
    # Choose the best model based on the results and proceed with further analysis
    # ...

if __name__ == "__main__":
    main()
