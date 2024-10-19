from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks

def apply_undersampling(X, y, method):
    if method == "Random Undersampling":
        resampler = RandomUnderSampler()
    elif method == "NearMiss":
        resampler = NearMiss()
    elif method == "Tomek Links":
        resampler = TomekLinks()
    
    return resampler.fit_resample(X, y)

print("Balancing Dataset using Undersampling...")
X = df.drop("{target}", axis=1, inplace=False)
y = df["{target}"]
X, y = apply_undersampling(X, y, method="{method}")
df = pd.concat([X, y], axis=1)
print("Dataset Balanced!")