from sklearn.preprocessing import OrdinalEncoder

def ordinal_encoding(df, columns):
    encoder = OrdinalEncoder()
    df[columns] = encoder.fit_transform(df[columns])
    return df