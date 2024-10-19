def dp_col(df, drop_columns):
    df = df.drop(drop_columns, axis=1)
    return df