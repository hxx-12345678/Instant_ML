from imblearn.combine import SMOTETomek, SMOTEENN

def apply_hybrid(X, y, method):
    if method == "SMOTETomek":
        resampler = SMOTETomek()
    elif method == "SMOTE-ENN":
        resampler = SMOTEENN()
    
    return resampler.fit_resample(X, y)

print("Balancing Dataset using Hybrid Algorithm...")
X = df.drop("{target}", axis=1, inplace=False)
y = df["{target}"]
X, y = apply_hybrid(X, y, method="{method}")
df = pd.concat([X, y], axis=1)
print("Dataset Balanced!")