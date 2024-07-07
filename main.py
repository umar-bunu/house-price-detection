import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from xgboost import XGBRegressor
import seaborn as sns

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

# Train model using cross-validation with tuned hyperparameters
def train_model(X_train, y_train):
    rf = RandomForestRegressor(random_state=42)
    gb = GradientBoostingRegressor(random_state=42)
    xgb = XGBRegressor(random_state=42)

    # Hyperparameter tuning
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.01],
        'max_depth': [3, 5]
    }
    xgb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.01],
        'max_depth': [3, 5]
    }

    rf_best = tune_hyperparameters(rf, rf_params, X_train, y_train)
    gb_best = tune_hyperparameters(gb, gb_params, X_train, y_train)
    xgb_best = tune_hyperparameters(xgb, xgb_params, X_train, y_train)
    
    # Voting Regressor with tuned models
    voting_reg = VotingRegressor([('rf', rf_best), ('gb', gb_best), ('xgb', xgb_best)])
    
    # Cross-validation
    scores = cross_val_score(voting_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print("Cross-validation RMSE scores:", rmse_scores)
    print("Average RMSE:", rmse_scores.mean())
    
    # Train the final model on the entire training set
    voting_reg.fit(X_train, y_train)
    return voting_reg

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Test RMSE:", rmse)
    
    # Calculate the percentage of predictions within 25% of the actual values
    within_25_percent = np.abs((y_test - y_pred) / y_test) <= 0.25
    accuracy_within_25_percent = np.mean(within_25_percent)
    print(f"Accuracy within 25%: {accuracy_within_25_percent * 100:.2f}%")
    
    return rmse, accuracy_within_25_percent

# Main function to run the process
def main(file_path):
    data = load_data(file_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

# Execute the main function
file_path = './datasets/dataset-utf8.csv'
main(file_path)
