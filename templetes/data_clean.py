def data_clean(df, drop_columns=None, imputation_params=None, encoding_params=None, scaling_columns=None):
    if drop_columns:
        df = dp_col(df, drop_columns)
    if imputation_params:
        df = simple_imputation(df, **imputation_params)
    if encoding_params:
        if encoding_params['method'] == 'label_encoding':
            df = label_encoding(df, encoding_params['columns'])
        elif encoding_params['method'] == 'ordinal_encoding':
            df = ordinal_encoding(df, encoding_params['columns'])
        elif encoding_params['method'] == 'one_hot_encoding':
            df = one_hot_encoding(df, encoding_params['columns'])
    if scaling_columns:
        df = feat_scal(df, **scaling_columns)

    return df