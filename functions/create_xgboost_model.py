from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

def create_xgboost_model(preprocessor):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=42))
    ])