def remove_outliers(X, y, columns, factor=1.5):
    for col in columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (factor * IQR)
        upper_bound = Q3 + (factor * IQR)
        mask = (X[col] >= lower_bound) & (X[col] <= upper_bound)
        X = X.loc[mask]
        y = y.loc[mask]
    return X, y