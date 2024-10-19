import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

X = df.drop("{target}", axis=1, inplace = False)
y = df["{target}"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state = 101)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"train accuracy: {{train_score*100:.4f}} %")
print(f"test accuracy: {{test_score*100:.4f}} %")