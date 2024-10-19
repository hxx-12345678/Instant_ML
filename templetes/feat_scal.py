from sklearn.preprocessing import StandardScaler, MinMaxScaler

def feat_scal(df, methods, col_names):
    for method, col in zip(methods, col_names):
        if method == "Standardization":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[[col]])
    return df