from sklearn.impute import SimpleImputer
import numpy as np

def simple_imputation(df, method='delete', fill_value=0, selected_columns=None):
    dfc = df.copy()
    if method == 'delete':
        dfc = df.dropna(subset=selected_columns)
    elif method in ['mean', 'median']:
        imputer = SimpleImputer(strategy=method)
        dfc[selected_columns] = imputer.fit_transform(dfc[selected_columns])
    elif method == 'most_frequent':
        imputer = SimpleImputer(strategy='most_frequent')
        dfc[selected_columns] = imputer.fit_transform(dfc[selected_columns])
    elif method in ['ffill', 'bfill']:
        dfc[selected_columns] = df[selected_columns].fillna(method=method)
    elif method == 'constant':
        imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
        dfc[selected_columns] = imputer.fit_transform(dfc[selected_columns])

    return dfc