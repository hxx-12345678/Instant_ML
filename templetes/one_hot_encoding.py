from sklearn.preprocessing import OneHotEncoder

def one_hot_encoding(df, columns):
	try:
		encoder = OneHotEncoder(sparse_output=False)
		encoded_data = encoder.fit_transform(df[columns])
		df_encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(columns))

		df = df.drop(columns, axis=1)
		df = pd.concat([df, df_encoded], axis=1)
	except:
		df = pd.get_dummies(df, columns=columns)
    
    return df