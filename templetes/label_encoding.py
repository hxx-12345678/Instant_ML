from sklearn.preprocessing import LabelEncoder

def label_encoding(df, columns):
    label_encoder = LabelEncoder()
    for col in columns:
        df[col] = label_encoder.fit_transform(df[col])
    return df