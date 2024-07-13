import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from xgboost import XGBRegressor

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Convert categorical data to numerical if necessary
    data = pd.get_dummies(data, columns=['area'], drop_first=True)
    
    # Handle outliers in the target variable
    q1 = data['price'].quantile(0.25)
    q3 = data['price'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data = data[(data['price'] >= lower_bound) & (data['price'] <= upper_bound)]
    
    X = data.drop(columns=['id', 'price'])
    y = data['price']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Split data into train and test sets
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Hyperparameter tuning using GridSearchCV
def tune_hyperparameters(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best parameters for", type(model).__name__, grid_search.best_params_)
    best_model = grid_search.best_estimator_
    return best_model

# Train and evaluate a single model
def train_and_evaluate_model(model, param_grid, X_train, y_train, X_test, y_test):
    best_model = tune_hyperparameters(model, param_grid, X_train, y_train)
    
    # Cross-validation
    scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(f"Cross-validation RMSE scores for {type(model).__name__}:", rmse_scores)
    print(f"Average RMSE for {type(model).__name__}:", rmse_scores.mean())
    
    # Train the final model on the entire training set
    best_model.fit(X_train, y_train)
    
    # Evaluate the model on the test set
    y_pred = best_model.predict(X_test)
    
    # Calculating metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    within_25_percent = np.abs((y_test - y_pred) / y_test) <= 0.25
    accuracy_within_25_percent = np.mean(within_25_percent)
    error_rate = 1 - accuracy_within_25_percent
    
    # Printing metrics
    print(f"Test R^2 for {type(model).__name__}: {r2}")
    print(f"Test MAE for {type(model).__name__}: {mae}")
    print(f"Test MSE for {type(model).__name__}: {mse}")
    print(f"Test RMSE for {type(model).__name__}: {rmse}")
    print(f"Accuracy for {type(model).__name__}: {accuracy_within_25_percent * 100:.2f}%")
    print(f"Error rate for {type(model).__name__}: {error_rate * 100:.2f}%")
    
    return best_model

# Main function to run the process
def main(file_path):
    data = load_data(file_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Define the models and their parameter grids
    models_and_params = [
        (RandomForestRegressor(random_state=42), {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        }),
        (GradientBoostingRegressor(random_state=42), {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.01],
            'max_depth': [3, 5]
        }),
        (XGBRegressor(random_state=42), {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.01],
            'max_depth': [3, 5]
        })
    ]
    
    # Train and evaluate each model individually
    for model, param_grid in models_and_params:
        print(f"Training and evaluating {type(model).__name__}")
        train_and_evaluate_model(model, param_grid, X_train, y_train, X_test, y_test)
        print("=" * 50)

# Execute the main function
file_path = './datasets/dataset-utf8.csv'
main(file_path)
