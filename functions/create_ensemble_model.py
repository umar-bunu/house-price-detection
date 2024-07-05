from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Lasso

def create_ensemble_model(preprocessor):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=42)
    lasso = Lasso(alpha=0.1, random_state=42, max_iter=10000, tol=1e-4)
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('ensemble', VotingRegressor([
            ('rf', rf),
            ('xgb', xgb),
            ('lasso', lasso)
        ]))
    ])