from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN

def apply_oversampling(X, y, method):
    if method == "Random Oversampling":
        resampler = RandomOverSampler()
    elif method == "SMOTE":
        resampler = SMOTE()
    elif method == "Borderline-SMOTE":
        resampler = BorderlineSMOTE()
    elif method == "ADASYN":
        resampler = ADASYN()
    
    return resampler.fit_resample(X, y)

print("Balancing Dataset using Oversampling...")
X = df.drop("{target}", axis=1, inplace=False)
y = df["{target}"]
X, y = apply_oversampling(X, y, method="{method}")
df = pd.concat([X, y], axis=1)
print("Dataset Balanced!")