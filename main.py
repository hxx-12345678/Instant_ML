import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, roc_curve, auc, precision_recall_curve
from io import BytesIO
import pickle
from fpdf import FPDF
import tempfile
import datetime

st.set_page_config(
    page_title="Instant ML",
    page_icon="media_files/icon.png",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'mailto:shahdishank24@gmail.com',
        'Report a bug': "mailto:shahdishank24@gmail.com",
        'About': "Make your model."
    }
)


lt = st.empty()
with lt.container():
	st.markdown("""
	<h1 style='text-align:center;'>Instant ML</h1>
	""", unsafe_allow_html=True)
	st.write("")

	col1, col2, col3 = st.columns([0.2, 0.5, 0.2])
	with col2:
		img_path = "media_files/home_img.svg"
		with open(img_path, 'r') as f:
			img = f.read()
		st.image(img, use_column_width=True)

	
	st.write("")
	st.write("")
	st.markdown("""
	<p style='font-size:20px; text-align:center'>
	Build Machine Learning models in seconds. Open the sidebar and <strong style='color:dodgerblue'>Get Started!<strong></p>
	""",unsafe_allow_html=True)


def get_data(df, target):
	y = df[target]
	X = df.drop(target, axis=1, inplace=False)
	return X,y


def params_clf(model_name):
	params = dict()
	if model_name == "Logistic Regression":
		params["solver"] = st.sidebar.selectbox(
			"solver",
			("lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga")
			)
		params["penalty"] = st.sidebar.selectbox(
			"penalty",
			("l2", "l1", "elasticnet")
			)
		params["C"] = st.sidebar.slider("C", 0.01, 1.0, 0.9)
	elif model_name == "KNN":
		params["n_neighbors"] = st.sidebar.slider("n_neighbors", 2, 20, 5)
		params["weights"] = st.sidebar.selectbox(
			"weights",
			("uniform", "distance")
			)
		params["metric"] = st.sidebar.selectbox(
			"metric",
			("minkowski", "euclidean", "manhattan")
			)
	elif model_name == "SVM":
		params["C"] = st.sidebar.slider("C", 0.1, 100.0, 1.0)
		params["gamma"] = st.sidebar.select_slider(
			"gamma",
			options=[0.0001, 0.001, 0.01, 0.1, 1, 10]
			)
		params["kernel"] = st.sidebar.selectbox(
			"kernel",
			("rbf", "linear", "sigmoid", "poly")
			)
		params["degree"] = 3
		if params["kernel"] == "poly":
			params["degree"] = st.sidebar.slider("degree", 2, 6, 3)
	elif model_name == "Naive Bayes":
		# params["var_smoothing"] = np.log(st.sidebar.slider("var_smoothing", -9, 1, -9))
		pass
	elif model_name == "Decision Tree":
		params["max_depth"] = st.sidebar.slider("max_depth", 3, 15, 3)
		params["min_samples_leaf"] = st.sidebar.slider("min_samples_leaf", 3, 20, 3)
		params["min_samples_split"] = st.sidebar.select_slider(
			"min_samples_split",
			options = [8, 10, 12, 14, 16, 18, 20]
			)
		params["criterion"] = st.sidebar.selectbox(
			"criterion",
			("gini", "entropy")
			)
	elif model_name == "Random Forest":
		params["n_estimators"] = st.sidebar.slider("n_estimators", 25, 150, 100)
		params["max_depth"] = st.sidebar.slider("max_depth", 1, 10, 1)
		params["max_features"] = st.sidebar.selectbox(
			"max_features",
			("sqrt", "log2", None)
			)
		params["max_leaf_nodes"] = st.sidebar.slider("max_leaf_nodes", 3, 9, 3)
	return params

@st.cache_resource
def model_clf(model_name, params):
	model = None
	if model_name == "Logistic Regression":
		model = LogisticRegression(solver = params["solver"], penalty = params["penalty"], C = params["C"])
	elif model_name == "KNN":
		model = KNeighborsClassifier(n_neighbors = params["n_neighbors"], weights = params["weights"], metric = params["metric"])
	elif model_name == "SVM":
		model = SVC(C = params["C"], gamma = params["gamma"], kernel = params["kernel"], degree = params["degree"])
	elif model_name == "Naive Bayes":
		model = GaussianNB()
		st.sidebar.caption("No need to tune the Parameters")
		st.sidebar.write(model.get_params())
	elif model_name == "Decision Tree":
		model = DecisionTreeClassifier(criterion = params["criterion"], max_depth = params["max_depth"], min_samples_split = params["min_samples_split"], min_samples_leaf = params["min_samples_leaf"])
	elif model_name == "Random Forest":
		model = RandomForestClassifier(n_estimators = params["n_estimators"], max_leaf_nodes = params["max_leaf_nodes"], max_depth = params["max_depth"], max_features = params["max_features"])
	return model

auto = ""

def grid_search_cv_clf(model_name):
	model = None
	if model_name == "Logistic Regression":
		params = [{"solver" : ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"], "penalty" : ["l2", "l1", "elasticnet"], "C" : [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1]}]
		model = GridSearchCV(LogisticRegression(), params, cv = 5, scoring = 'accuracy')
	elif model_name == "KNN":
		params = [{"n_neighbors" : np.arange(2, 30, 1), "weights" : ['uniform', 'distance'], "metric" : ["minkowski", "euclidean", "manhattan"]}]
		model = GridSearchCV(KNeighborsClassifier(), params, cv = 5, scoring = 'accuracy')
	elif model_name == "SVM":
		params = [{"C" : [0.1, 1, 10, 100], "gamma" : [0.0001, 0.001, 0.01, 0.1, 1, 10], "kernel" : ["rbf", "linear", "sigmoid", "poly"], "degree" : [2, 3, 4, 5, 6]}]
		model = GridSearchCV(SVC(), params, cv = 5, scoring = 'accuracy')
	elif model_name == "Naive Bayes":
		params = [{"var_smoothing" : np.logspace(1, -9, 100)}]
		model = GridSearchCV(GaussianNB(), params, cv = 5, scoring = 'accuracy')
	elif model_name == "Decision Tree":
		params = [{"max_depth" : [3, 6, 9], "min_samples_split" : [8, 12, 16, 20], "min_samples_leaf" : [3, 6, 9, 12, 15], "criterion" : ["gini", "entropy"]}]
		model = GridSearchCV(DecisionTreeClassifier(), params, cv = 5, scoring = 'accuracy')
	elif model_name == "Random Forest":
		params = [{"n_estimators" : [25, 50, 100, 150], "max_depth" : [3, 6, 9], "max_features" : ["sqrt", "log2", None], "max_leaf_nodes" : [3, 6, 9]}]
		model = GridSearchCV(RandomForestClassifier(), params, cv = 5, scoring = 'accuracy')
	return model


def params_reg(model_name):
	params = dict()
	if model_name == "Linear Regression":
		params["fit_intercept"] = st.sidebar.selectbox("fit_intercept", (True, False))
		params["copy_X"] = st.sidebar.selectbox("copy_X", (True, False))
	elif model_name == "Ridge Regression":
		params["alpha"] = st.sidebar.slider("alpha", 0.0, 10.0, 0.5)
		params["fit_intercept"] = st.sidebar.selectbox("fit_intercept", (True, False))
		params["solver"] = st.sidebar.selectbox("solver", ("auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"))
	elif model_name == "Lasso Regression":
		params["alpha"] = st.sidebar.slider("alpha", 0.0, 10.0, 0.5)
		params["fit_intercept"] = st.sidebar.selectbox("fit_intercept", (True, False))
		params["selection"] = st.sidebar.selectbox("selection", ("cyclic", "random"))
	elif model_name == "Elastic Net":
		params["alpha"] = st.sidebar.slider("alpha", 0.0, 10.0, 0.5)
		params["fit_intercept"] = st.sidebar.selectbox("fit_intercept", (True, False))
		params["l1_ratio"] = st.sidebar.slider("l1_ratio", 0.0, 1.0, 0.5)
	elif model_name == "KNN":
		params["n_neighbors"] = st.sidebar.slider("n_neighbors", 2, 20, 5)
		params["weights"] = st.sidebar.selectbox(
			"weights",
			("uniform", "distance")
			) 
	elif model_name == "SVM":
		params["C"] = st.sidebar.slider("C", 0.1, 100.0, 1.0)
		params["gamma"] = st.sidebar.selectbox(
			"gamma",
			("scale", "auto")
			)
		params["kernel"] = st.sidebar.selectbox(
			"kernel",
			("rbf", "linear", "sigmoid", "poly")
			)
		params["degree"] = 3
		if params["kernel"] == "poly":
			params["degree"] = st.sidebar.slider("degree", 2, 6, 3)
	elif model_name == "Decision Tree":
		params["criterion"] = st.sidebar.selectbox("criterion", ("squared_error", "friedman_mse", "absolute_error", "poisson"))
		params["splitter"] = st.sidebar.selectbox("splitter", ("best", "random"))
		params["min_samples_leaf"] = st.sidebar.slider("min_samples_leaf", 1, 20, 1)
		params["min_samples_split"] = st.sidebar.select_slider(
			"min_samples_split",
			options = [2, 8, 10, 12, 14, 16, 18, 20]
			)
	elif model_name == "Random Forest":
		params["n_estimators"] = st.sidebar.slider("n_estimators", 50, 200, 100)
		params["max_features"] = st.sidebar.selectbox(
			"max_features",
			("sqrt", "log2", None)
			)
		params["min_samples_leaf"] = st.sidebar.slider("min_samples_leaf", 1, 20, 1)
		params["min_samples_split"] = st.sidebar.select_slider(
			"min_samples_split",
			options = [2, 8, 10, 12, 14, 16, 18, 20]
			)
	return params

@st.cache_resource
def model_reg(model_name, params):
	model = None
	if model_name == "Linear Regression":
		model = LinearRegression(fit_intercept = params["fit_intercept"], copy_X = params["copy_X"])
	elif model_name == "Ridge Regression":
		model = Ridge(alpha = params["alpha"], fit_intercept = params["fit_intercept"], solver = params["solver"])
	elif model_name == "Lasso Regression":
		model = Lasso(alpha = params["alpha"], fit_intercept = params["fit_intercept"], selection = params["selection"])
	elif model_name == "Elastic Net":
		model = ElasticNet(alpha = params["alpha"], fit_intercept = params["fit_intercept"], l1_ratio = params["l1_ratio"])
	elif model_name == "KNN":
		model = KNeighborsRegressor(n_neighbors = params["n_neighbors"], weights = params["weights"])
	elif model_name == "SVM":
		model = SVR(C = params["C"], gamma = params["gamma"], kernel = params["kernel"], degree = params["degree"])
	elif model_name == "Decision Tree":
		model = DecisionTreeRegressor(criterion = params["criterion"], splitter = params["splitter"], min_samples_split = params["min_samples_split"], min_samples_leaf = params["min_samples_leaf"])
	elif model_name == "Random Forest":
		model = RandomForestRegressor(n_estimators = params["n_estimators"], max_features = params["max_features"], min_samples_split = params["min_samples_split"], min_samples_leaf = params["min_samples_leaf"])
	return model


def grid_search_cv_reg(model_name):
	model = None
	if model_name == "Linear Regression":
		params = [{"fit_intercept" : [True, False], "copy_X" : [True, False]}]
		model = GridSearchCV(LinearRegression(), params, cv = 5)
	elif model_name == "Ridge Regression":
		params = [{"alpha" : [0, 0.5, 1, 1.5, 2], "fit_intercept" : [True, False], "solver" : ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]}]
		model = GridSearchCV(Ridge(), params, cv = 5)
	elif model_name == "Lasso Regression":
		params = [{"alpha" : [0, 0.5, 1, 1.5, 2], "fit_intercept" : [True, False], "selection" : ["cyclic", "random"]}]
		model = GridSearchCV(Lasso(), params, cv = 5)
	elif model_name == "Elastic Net":
		params = [{"alpha" : [0, 0.5, 1, 1.5, 2], "fit_intercept" : [True, False], "l1_ratio" : [0, 0.2, 0.5, 0.8, 1]}]
		model = GridSearchCV(ElasticNet(), params, cv = 5)
	elif model_name == "KNN":
		params = [{"n_neighbors" : np.arange(2, 20, 1), "weights" : ["uniform", "distance"]}]
		model = GridSearchCV(KNeighborsRegressor(), params, cv = 5)
	elif model_name == "SVM":
		params = [{"C" : [0.1, 1, 10, 100], "gamma" : ["scale", "auto"], "kernel" : ["rbf", "linear", "sigmoid", "poly"], "degree" : [2, 3, 4, 5, 6]}]
		model = GridSearchCV(SVR(), params, cv = 5)
	elif model_name == "Decision Tree":
		params = [{"splitter" : ["best", "random"], "min_samples_split" : [2, 5, 8, 12, 16, 20], "min_samples_leaf" : [1, 3, 6, 9, 12, 15], "criterion" : ["squared_error", "friedman_mse", "absolute_error", "poisson"]}]
		model = GridSearchCV(DecisionTreeRegressor(), params, cv = 5)
	elif model_name == "Random Forest":
		params = [{"n_estimators" : [50, 100, 150, 200], "max_features" : ["sqrt", "log2", None], "min_samples_split" : [2, 5, 8, 12, 16, 20], "min_samples_leaf" : [1, 3, 6, 9, 12, 15]}]
		model = GridSearchCV(RandomForestRegressor(), params, cv = 5)
	return model



model_select = ""

def classification():
	global model_select
	model_select = st.sidebar.selectbox(
	'Select a model',
	('Logistic Regression', 'KNN', 'SVM', 'Naive Bayes', 'Decision Tree', 'Random Forest')
	)
	tune_choice = st.sidebar.selectbox(
	'Hyperparameter Tuning',
	('Manually', 'Automatically')
	)
	if tune_choice == "Manually":
		params = params_clf(model_select)
		model = model_clf(model_select, params)
	else:
		model = grid_search_cv_clf(model_select)
		global auto
		auto = "auto"
	return model


def regression():
	global model_select
	model_select = st.sidebar.selectbox(
	'Select a model',
	('Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net', 'KNN', 'SVM', 'Decision Tree', 'Random Forest')
	)
	tune_choice = st.sidebar.selectbox(
	'Hyperparameter Tuning',
	('Manually', 'Automatically')
	)
	if tune_choice == "Manually":
		params = params_reg(model_select)
		model = model_reg(model_select, params)
	else:
		model = grid_search_cv_reg(model_select)
		global auto
		auto = "auto"
	return model

def show_data(df):
	st.subheader(f"Shape of the Dataset: {df.shape}")
	st.write("")
	st.write("")
	st.caption("Data Overview")
	st.dataframe(df.head(), hide_index=True)
	st.write("")
	st.write("")
	st.caption("Some Statistics")
	st.table(df.describe())


def fetch_code(fname):
	with open(f"templetes/{fname}.py", "r") as f:
		data = f.read()
	return data


def get_code(algo_type, f_var, params):
	if algo_type == "Classification":
		if model_select == "Logistic Regression":
			data = fetch_code("clf_logistic_reg")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], solver = params["solver"], penalty = params["penalty"], C = params["C"])
		elif model_select == "KNN":
			data = fetch_code("clf_knn")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], n_neighbors = params["n_neighbors"], weights = params["weights"], metric = params["metric"])
		elif model_select == "SVM":
			data = fetch_code("clf_svm")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], C = params["C"], gamma = params["gamma"], kernel = params["kernel"], degree = params["degree"])
		elif model_select == "Naive Bayes":
			data = fetch_code("clf_naive_bayes")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"])
		elif model_select == "Decision Tree":
			data = fetch_code("clf_decision_tree")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], criterion = params["criterion"], max_depth = params["max_depth"], min_samples_split = params["min_samples_split"], min_samples_leaf = params["min_samples_leaf"])
		elif model_select == "Random Forest":
			data = fetch_code("clf_random_forest")
			if params["max_features"] is None:
				data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], n_estimators = params["n_estimators"], max_leaf_nodes = params["max_leaf_nodes"], max_depth = params["max_depth"], max_features = params["max_features"])
			else:
				max_f = "\""+params["max_features"]+"\""
				data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], n_estimators = params["n_estimators"], max_leaf_nodes = params["max_leaf_nodes"], max_depth = params["max_depth"], max_features = max_f)
	elif algo_type == "Regression":
		if model_select == "Linear Regression":
			data = fetch_code("reg_linear")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], fit_intercept = params["fit_intercept"], copy_X = params["copy_X"])
		elif model_select == "Ridge Regression":
			data = fetch_code("reg_ridge")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], alpha = params["alpha"], fit_intercept = params["fit_intercept"], solver = params["solver"])
		elif model_select == "Lasso Regression":
			data = fetch_code("reg_lasso")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], alpha = params["alpha"], fit_intercept = params["fit_intercept"], selection = params["selection"])
		elif model_select == "Elastic Net":
			data = fetch_code("reg_elastic_net")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], alpha = params["alpha"], fit_intercept = params["fit_intercept"], l1_ratio = params["l1_ratio"])
		elif model_select == "KNN":
			data = fetch_code("reg_knn")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], n_neighbors = params["n_neighbors"], weights = params["weights"])
		elif model_select == "SVM":
			data = fetch_code("reg_svm")
			data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], C = params["C"], gamma = params["gamma"], kernel = params["kernel"], degree = params["degree"])
		elif model_select == "Decision Tree":
				data = fetch_code("reg_decision_tree")
				data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], criterion = params["criterion"], splitter = params["splitter"], min_samples_split = params["min_samples_split"], min_samples_leaf = params["min_samples_leaf"])
		elif model_select == "Random Forest":
				data = fetch_code("reg_random_forest")
				if params["max_features"] is None:
					data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], n_estimators = params["n_estimators"], max_features = params["max_features"], min_samples_split = params["min_samples_split"], min_samples_leaf = params["min_samples_leaf"])
				else:
					max_f = "\""+params["max_features"]+"\""
					data = data.format(filename=f_var["filename"], target = f_var["target"], test_size = f_var["tst_size"], n_estimators = params["n_estimators"], max_features = max_f, min_samples_split = params["min_samples_split"], min_samples_leaf = params["min_samples_leaf"])
	return data


def intr_plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="True", color="Count"),
                    x=[str(i) for i in range(cm.shape[1])], 
                    y=[str(i) for i in range(cm.shape[0])])
    fig.update_layout(title='Confusion Matrix')
    return fig

def intr_plot_class_distribution(y_pred):
    unique_classes, counts = np.unique(y_pred, return_counts=True)
    fig = px.bar(x=unique_classes, y=counts, labels={'x': 'Class', 'y': 'Number of Instances'})
    fig.update_layout(title='Class Distribution')
    return fig

def intr_plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    return fig

def intr_plot_precision_recall_curve(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall curve'))
    fig.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision')
    return fig

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    return fig

def plot_class_distribution(y_pred):
    unique_classes, counts = np.unique(y_pred, return_counts=True)
    fig = plt.figure(figsize=(10, 7))
    plt.bar(unique_classes, counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.title('Class Distribution')
    plt.xticks(unique_classes)
    return fig

def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    fig = plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    return fig

def plot_precision_recall_curve(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
    fig = plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    return fig

def intr_plot_predicted_vs_actual(y_true, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predicted vs Actual', opacity=0.5))
    fig.add_trace(go.Scatter(x=[min(y_true), max(y_true)], y=[min(y_true), max(y_true)], mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
    fig.update_layout(title='Predicted vs. Actual Values', xaxis_title='Actual Values', yaxis_title='Predicted Values')
    return fig

def intr_plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals', opacity=0.5))
    fig.add_trace(go.Scatter(x=[min(y_pred), max(y_pred)], y=[0, 0], mode='lines', name='Zero Residuals', line=dict(color='red', dash='dash')))
    fig.update_layout(title='Residuals Plot', xaxis_title='Predicted Values', yaxis_title='Residuals')
    return fig

def intr_plot_error_distribution(y_true, y_pred):
    errors = y_true - y_pred
    fig = px.histogram(errors, nbins=50, title='Distribution of Prediction Errors')
    fig.update_layout(xaxis_title='Prediction Error', yaxis_title='Count')
    return fig

def plot_predicted_vs_actual(y_true, y_pred):
    fig = plt.figure(figsize=(10, 7))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs. Actual Values')
    return fig

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig = plt.figure(figsize=(10, 7))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    return fig

def plot_error_distribution(y_true, y_pred):
    errors = y_true - y_pred
    fig = plt.figure(figsize=(10, 7))
    sns.histplot(errors, kde=True, color='blue')
    plt.xlabel('Prediction Error')
    plt.title('Distribution of Prediction Errors')
    return fig

def model_download(se, model):
	with st.sidebar:
		with st.spinner("Saving model..."):
			buffer = BytesIO()
			pickle.dump(model, buffer)
			buffer.seek(0)
			time.sleep(1)
	st.toast("Model is saved and ready to download")
	se.download_button(
	    label="Download Model",
	    data=buffer,
	    file_name="model.pkl",
	    mime="application/octet-stream",
	    use_container_width=True,
	    type="primary"
	)


# Plotting functions
def pdf_plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

def pdf_plot_class_distribution(y_pred):
    unique_classes, counts = np.unique(y_pred, return_counts=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(unique_classes, counts, color='skyblue')
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Instances')
    ax.set_title('Class Distribution')
    ax.set_xticks(unique_classes)
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

def pdf_plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

def pdf_plot_precision_recall_curve(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(recall, precision, color='blue', lw=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

def pdf_plot_predicted_vs_actual(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(y_true, y_pred, alpha=0.3)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Predicted vs. Actual Values')
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

def pdf_plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(y_pred, residuals, alpha=0.3)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals Plot')
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

def pdf_plot_error_distribution(y_true, y_pred):
    errors = y_true - y_pred
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.histplot(errors, kde=True, color='blue', ax=ax)
    ax.set_xlabel('Prediction Error')
    ax.set_title('Distribution of Prediction Errors')
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf


# Function to add custom font
def add_custom_fonts(pdf):
    pdf.add_font('AvenirNext', '', 'AvenirNext.ttf', uni=True)
    pdf.add_font('AvenirNext', 'B', 'AvenirNextDemi.ttf', uni=True)

# PDF generation
class PDFReport(FPDF):
    def header(self):
        self.set_font('AvenirNext', 'B', 12)
        self.cell(0, 10, 'Machine Learning Model Report', 1, 1, 'C')
        self.ln(5)

    def footer(self):
    	self.set_y(-15)  # Position 15 mm from the bottom
    	self.set_font('AvenirNext', '', 10)
    	self.set_text_color(0,0,255)  # Blue color
    	self.cell(0, 10, 'Generated by Instant ML', 0, 0, 'C', link='https://instant-ml.streamlit.app/')

    def chapter_title(self, title):
    	self.set_font('AvenirNext', 'B', 12)
    	self.cell(0, 10, title, 0, 1, 'L')
    	self.ln(1)

    def chapter_body(self, body, ln=5):
    	self.set_font('AvenirNext', '', 12)
    	self.multi_cell(0, 9, body)
    	self.ln(ln)

    def courier_text(self, body):
    	self.set_font('Courier', '', 12)
    	self.multi_cell(0, 9, body)
    	self.ln(5)

    def add_image(self, img_buf, width=6, height=4):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            img_buf.seek(0)
            tmpfile.write(img_buf.read())
            tmpfile.flush()
            # Calculate x position to center the image
            page_width = self.w - 2 * self.l_margin
            img_width_mm = width * 25.4  # convert width from inches to mm
            x = (page_width - img_width_mm) / 2 + self.l_margin
            self.image(tmpfile.name, x=x, y=None, w=img_width_mm, h=height*25.4)  # convert inches to mm
        self.ln()

def create_pdf_report(algo_type, model, report_params):
	with st.sidebar:
		with st.spinner("Generating Report..."):
		    pdf = PDFReport()
		    add_custom_fonts(pdf)
		    pdf.add_page()

		    # Model and data summaries
		    model_name = type(model).__name__
		    model_params = model.get_params()
		    num_features = report_params["X_test"].shape[1]
		    num_samples = report_params["X_test"].shape[0]
		    num_train_samples = report_params["X_train"].shape[0]
		    num_target = 1

		    # Introduction
		    pdf.chapter_body(f"Date of the report generation: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", 2)
		    
		    # Data Summary
		    pdf.chapter_title("Data Summary")
		    pdf.chapter_body(f"Number of features: {num_features}\nNumber of target labels: {num_target}\nNumber of samples: {num_samples+num_train_samples}")

		    # Model Summary
		    pdf.chapter_title("Model Summary")
		    n = report_params["y_test"].nunique()
		    if algo_type == "Classification":
		    	if n == 2:
		    		task = "Binary Classification"
		    	else:
		    		task = "Multiclass Classification"
		    else:
		    	task = algo_type
		    pdf.chapter_body(f"Task: {task}\nModel: {model_name}\nHyperparameters: {model_params}\nTrain/Test Split: {num_train_samples}/{num_samples}")

		    # Model Performance
		    pdf.chapter_title("Model Performance")
		    if algo_type == "Classification":
		    	performance = classification_report(report_params["y_test"], report_params["y_pred"], output_dict=True)
		    	performance_text = pd.DataFrame(performance).transpose().to_string()
		    	pdf.chapter_body(f"Train Accuracy: {report_params['train_score']*100:.4f} %\nTest Accuracy: {report_params['test_score']*100:.4f} %\nClassification Report:", 1)
		    	pdf.courier_text(performance_text)
		    else:
		    	pdf.chapter_body(f"Train Score: {report_params['train_score']*100:.4f} %\nTest Score: {report_params['test_score']*100:.4f} %\nMean Absolute Error: {report_params['mae']:.2f}\nMean Squared Error: {report_params['mse']:.2f}\nRoot Mean Squared Error: {report_params['rmse']:.2f}\nR2 Score: {report_params['r2']:.2f}")

		    pdf.add_page()

		    plot_functions = list()
		    if algo_type == "Classification":
		    	plot_functions = [
		    		("Confusion Matrix", pdf_plot_confusion_matrix, (report_params["y_test"], report_params["y_pred"])),
		    		("Class Distribution", pdf_plot_class_distribution, (report_params["y_pred"],))
		    	]
		    	# n = y_test.nunique()
		    	if n == 2 and report_params['y_proba'] is not report_params['y_pred']:
		    		plot_functions.append(("ROC Curve", pdf_plot_roc_curve, (report_params["y_test"], report_params["y_proba"])))
		    		plot_functions.append(("Precision-Recall Curve", pdf_plot_precision_recall_curve, (report_params["y_test"], report_params["y_proba"])))
		    else:
		    	plot_functions = [
			        ("Predicted vs. Actual", pdf_plot_predicted_vs_actual, (report_params["y_test"], report_params["y_pred"])),
			        ("Residuals", pdf_plot_residuals, (report_params["y_test"], report_params["y_pred"])),
			        ("Error Distribution", pdf_plot_error_distribution, (report_params["y_test"], report_params["y_pred"]))
			    ]

		    # Plots
		    
		    for title, plot_func, args in plot_functions:
		        pdf.chapter_title(title)
		        img_buf = plot_func(*args)
		        pdf.add_image(img_buf)

		    # pdf.output(report_path)
		    pdf_buffer = BytesIO()
		    pdf_output = pdf.output(dest='S').encode('latin1')
		    pdf_buffer.write(pdf_output)
		    pdf_buffer.seek(0)
	st.toast("Report generated and ready to download")
	return pdf_buffer


def algorithm(df, demo="no"):
	if not df.empty:
		show_data(df)
		cols = ("select", )
		for i, j in enumerate(df.columns):
			cols = cols + (j,)
		if demo == "no":
			target = st.sidebar.selectbox(
				'Select target value',
				cols,
				)
		else:
			if demo == "clf_demo":
				target = "Class"
			elif demo == "reg_demo":
				target = "Price"
		if target != "select":
			st.sidebar.write("")
			create_btn = st.sidebar.toggle("Create Model")
			st.sidebar.write("")
		if target != "select" and create_btn:
			X, y = get_data(df, target)
			if not X.empty:
				tst_size = st.sidebar.slider("Select the test size of the dataset to split", 0.1, 0.9, 0.2)
				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = tst_size, random_state = 101)
				st.write("")
				st.subheader("Shape of")
				st.write(f"- X_train: **{X_train.shape}**")
				st.write(f"- X_test: **{X_test.shape}**")
				st.write(f"- y_train: **{y_train.shape}**")
				st.write(f"- y_test: **{y_test.shape}**")

				if demo == "no":
					algo_type = st.sidebar.selectbox(
						'Select an algorithm type',
						('Classification', 'Regression')
						)
				else:
					if demo == "clf_demo":
						algo_type = "Classification"
						st.sidebar.subheader("Classification")
					elif demo == "reg_demo":
						algo_type = "Regression"
						st.sidebar.subheader("Regression")
				if algo_type == "Classification":
					start_time = time.time()
					model = classification()
					model.fit(X_train, y_train)
					end_time = time.time()
					time_taken = end_time - start_time
					y_pred = model.predict(X_test)

					if auto == "auto":
						params = model.best_params_
						st.sidebar.caption("Better Parameters")
						st.sidebar.write(model.best_params_)
						st.sidebar.caption("Average Score")
						st.sidebar.write(model.best_score_*100)
					else:
						params = model.get_params()

					st.markdown(
					"""
					---
					"""
					)
					st.sidebar.write("")
					se = st.sidebar.empty()
					model_download_btn = se.button("Save Model", use_container_width=True, type="primary")
					if model_download_btn:
						model_download(se, model)

					try:
						y_proba = model.predict_proba(X_test)
					except:
						y_proba = y_pred

					# accuracy = accuracy_score(y_test, y_pred)
					train_score = model.score(X_train, y_train)
					test_score = model.score(X_test, y_test)
					# st.subheader(f"accuracy: {accuracy}")

					report_params = {"X_train": X_train,
					"X_test": X_test,
					"y_test": y_test,
					"y_pred": y_pred,
					"y_proba": y_proba,
					"train_score": train_score,
					"test_score": test_score
					}

					ety = st.sidebar.empty()
					if ety.button("Generate Report", use_container_width=True):
						pdf_buffer = create_pdf_report(algo_type, model, report_params)
						ety.download_button('Download Report', pdf_buffer, file_name='report.pdf', mime='application/pdf', use_container_width=True)

					st.sidebar.write("")
					st.sidebar.caption("Model Creation Time (in seconds)")
					st.sidebar.write(time_taken)

					st.subheader("Model Performance")
					st.write("")

					train_color = "green"
					test_color = "green"
					if train_score < 0.5:
						train_color = "red"
					if test_score < 0.5:
						test_color = "red"
					st.progress(train_score if train_score > 0 else 0, f"# Train Accuracy : :{train_color}[{train_score*100:.4f} %]")
					st.write("")
					st.progress(test_score if test_score > 0 else 0, f"# Test Accuracy : :{test_color}[{test_score*100:.4f} %]")

					# st.subheader(f"train accuracy: {train_score*100:.4f} %")
					# st.subheader(f"test accuracy: {test_score*100:.4f} %")
					st.header("\n")
					# st.sidebar.write(list(model.cv_results_.keys()))
					cr = classification_report(y_test, y_pred)
					st.code(f"Classification Report: \n\n {cr}")
					cm = confusion_matrix(y_test, y_pred)
					st.code(f"Confusion Matrix: \n\n {cm}")

					st.subheader("")
					show = st.toggle("**Show Comparisons**", value=True)
					if show:
						count = st.slider("How many rows do you want to see", 1, 30, 5)
						col1, col2 = st.columns(2)
						with col1:
							# st.caption("Actual target values")
							st.dataframe(y_test.head(count), hide_index = True, use_container_width = True, column_config = {target : "Actual Target Values"})
						with col2:
							# st.caption("Predicted target values")
							st.dataframe(y_pred[:count], hide_index = True, use_container_width = True, column_config = {"value" : "Predicted Target Values"})

					st.subheader("")

					tab1, tab2 = st.tabs(["Interactive", "Normal"])
					n = y_test.nunique()
					try:
						y_proba = model.predict_proba(X_test)
					except:
						y_proba = y_pred
					with tab1:
						st.write("")
						st.subheader("Confusion Matrix")
						st.write("")
						ifig = intr_plot_confusion_matrix(y_test, y_pred)
						st.plotly_chart(ifig)
						st.subheader("")

						st.subheader("Class Distribution")
						st.write("")
						ifig2 = intr_plot_class_distribution(y_pred)
						st.plotly_chart(ifig2)

						if n == 2 and y_proba is not y_pred:
							st.subheader("")
							st.subheader("ROC Curve")
							ifig3 = intr_plot_roc_curve(y_test, y_proba)
							st.plotly_chart(ifig3)
							st.subheader("")

							st.subheader("Precision-Recall Curve")
							ifig4 = intr_plot_precision_recall_curve(y_test, y_proba)
							st.plotly_chart(ifig4)


					with tab2:
						st.write("")
						st.subheader("Confusion Matrix")
						st.write("")
						fig = plot_confusion_matrix(y_test, y_pred)
						st.pyplot(fig)
						st.subheader("")

						st.subheader("Class Distribution")
						st.write("")
						fig2 = plot_class_distribution(y_pred)
						st.pyplot(fig2)

						if n == 2 and y_proba is not y_pred:
							st.subheader("")
							st.subheader("ROC Curve")
							st.write("")
							fig3 = plot_roc_curve(y_test, y_proba)
							st.pyplot(fig3)
							st.subheader("")

							st.subheader("Precision-Recall Curve")
							st.write("")
							fig4 = plot_precision_recall_curve(y_test, y_proba)
							st.pyplot(fig4)


					st.header("")
					gen = st.toggle("**Generate Code**")
					if gen:
						format_variable = {"filename":filename, "target":target, "tst_size":tst_size}
						data = get_code(algo_type, format_variable, params)
						st.code(data)
						st.download_button(
						    label="Download Code",
						    data=data,
						    file_name=filename.replace('.csv', "") + "_" + model_select.replace(" ", "_") + ".py",
						    mime='text/python',
						    help="Download"
						)
				else:
					start_time = time.time()
					model = regression()
					model.fit(X_train, y_train)
					end_time = time.time()
					time_taken = end_time - start_time
					y_pred = model.predict(X_test)

					if auto == "auto":
						params = model.best_params_
						st.sidebar.caption("Better Parameters")
						st.sidebar.write(model.best_params_)
						st.sidebar.caption("Average Score")
						st.sidebar.write(model.best_score_*100)
					else:
						params = model.get_params()


					st.markdown(
					"""
					---
					"""
					)
					st.sidebar.write("")
					se = st.sidebar.empty()
					model_download_btn = se.button("Save Model", use_container_width=True, type="primary")
					if model_download_btn:
						model_download(se, model)


					train_score = model.score(X_train, y_train)
					test_score = model.score(X_test, y_test)
					mae = mean_absolute_error(y_test, y_pred)
					mse = mean_squared_error(y_test, y_pred)
					rmse = root_mean_squared_error(y_test, y_pred)
					r2 = r2_score(y_test, y_pred)

					report_params = {"X_train": X_train,
					"X_test": X_test,
					"y_test": y_test,
					"y_pred": y_pred,
					"train_score": train_score,
					"test_score": test_score,
					"mae": mae,
					"mse": mse,
					"rmse": rmse,
					"r2": r2
					}

					ety = st.sidebar.empty()
					if ety.button("Generate Report", use_container_width=True):
						pdf_buffer = create_pdf_report(algo_type, model, report_params)
						ety.download_button('Download Report', pdf_buffer, file_name='report.pdf', mime='application/pdf', use_container_width=True)

					st.sidebar.write("")
					st.sidebar.caption("Model Creation Time (in seconds)")
					st.sidebar.write(time_taken)
					
					st.subheader("Model Performance")
					st.write("")

					train_color = "green"
					test_color = "green"
					if train_score < 0.5:
						train_color = "red"
					if test_score < 0.5:
						test_color = "red"
					st.progress(train_score if train_score > 0 else 0, f"# Train Score : :{train_color}[{train_score:.4f}]")
					st.write("")
					st.progress(test_score if test_score > 0 else 0, f"# Test Score : :{test_color}[{test_score:.4f}]")
					st.subheader("")

					col1, col2 = st.columns(2, gap="large")
					col1.metric("# **:blue[Mean Absolute Error]**", f"{mae:.2f}")
					col2.metric("# **:blue[Mean Squared Error]**", f"{mse:.2f}")
					col3, col4 = st.columns(2, gap="large")
					col3.metric("# **:blue[Root Mean Squared Error]**", f"{rmse:.2f}")
					col4.metric("# **:blue[R2 Score]**", f"{r2:.2f}")
					st.write("")

					# st.subheader(f"train score: {train_score:.4f}")
					# st.subheader(f"test score: {test_score:.4f}")
					# st.subheader(f"Mean Absolute Error: {mae:.4f}")
					# st.subheader(f"Mean Squared Error: {mse:.4f}")
					# st.subheader(f"Root Mean Squared Error: {rmse:.4f}")
					# st.subheader(f"R2 Score: {r2:.4f}")

					st.subheader("")
					show = st.toggle("**Show Comparisons**", value=True)
					if show:
						count = st.slider("How many rows do you want to see", 1, 30, 5)
						col1, col2 = st.columns(2)
						with col1:
							st.dataframe(y_test.head(count), hide_index = True, use_container_width = True, column_config = {target : "Actual Target Values"})
						with col2:
							st.dataframe(y_pred[:count], hide_index = True, use_container_width = True, column_config = {"value" : "Predicted Target Values"})

					st.subheader("")

					col = len(X_test.columns)

					t1, t2 = st.tabs(["Interactive", "Normal"])

					with t1:
						st.write("")
						col_select = st.slider("Select column for graph", 1, col, 1, key=1)

						ifig = go.Figure()
						ifig.add_trace(go.Scatter(x=X_test.iloc[:, col_select-1], y=y_test, mode='markers', name='Actual', marker=dict(color='blue')))
						ifig.add_trace(go.Scatter(x=X_test.iloc[:, col_select-1], y=y_pred, mode='lines', name='Predicted', line=dict(color='green')))
						ifig.update_layout(
						    title=f"Actual vs. Predicted for column {col_select}",
						    xaxis_title=f"X_test column {col_select}",
						    yaxis_title="Values",
						    legend=dict(x=0, y=1)
						)
						st.plotly_chart(ifig)

						# st.write("")
						st.divider()
						st.write("")

						st.subheader("Predicted vs Actual")
						ifig2 = intr_plot_predicted_vs_actual(y_test, y_pred)
						st.plotly_chart(ifig2)
						st.subheader("")

						st.subheader("Residuals")
						ifig3 = intr_plot_residuals(y_test, y_pred)
						st.plotly_chart(ifig3)
						st.subheader("")

						st.subheader("Error Distribution")
						ifig4 = intr_plot_error_distribution(y_test, y_pred)
						st.plotly_chart(ifig4)

					with t2:
						st.write("")
						col_select = st.slider("Select column for graph", 1, col, 1, key=2)

						fig = plt.figure(figsize=(10, 7))
						sns.scatterplot(x=X_test.iloc[:, col_select-1], y=y_test, color='b', label='Actual')
						sns.lineplot(x=X_test.iloc[:, col_select-1], y=y_pred, color='g', label='Predicted')
						plt.xlabel(f"X_test column {col_select}")
						plt.ylabel("Values")
						plt.title("Actual vs. Predicted for a perticular column")
						plt.legend()
						st.pyplot(fig)

						st.write("")
						st.divider()
						st.write("")


						st.subheader("Predicted vs Actual")
						st.write("")
						fig2 = plot_predicted_vs_actual(y_test, y_pred)
						st.pyplot(fig2)
						st.subheader("")

						st.subheader("Residuals")
						st.write("")
						fig3 = plot_residuals(y_test, y_pred)
						st.pyplot(fig3)
						st.subheader("")

						st.subheader("Error Distribution")
						st.write("")
						fig4 = plot_error_distribution(y_test, y_pred)
						st.pyplot(fig4)

					st.header("")
					gen = st.toggle("**Generate Code**")
					if gen:
						format_variable = {"filename":filename, "target":target, "tst_size":tst_size}
						data = get_code(algo_type, format_variable, params)
						st.code(data)
						st.download_button(
						    label="Download Code",
						    data=data,
						    file_name=filename.replace('.csv', "") + "_" + model_select.replace(" ", "_") + ".py",
						    mime='text/python',
						    help="Download"
						)


def upload_file():
	uploaded_file = st.sidebar.file_uploader("Upload the CSV file (separator must be coma)", type=['csv'])
	if uploaded_file is not None:
		try:
			df = pd.read_csv(uploaded_file)
			global filename
			filename = uploaded_file.name
			lt.empty()
		except:
			st.sidebar.error("The File is empty or unable to read!")
			df = pd.DataFrame()
		finally:
			algorithm(df)

choice = st.sidebar.selectbox("Choose data upload option", ("-- select --", "Try with demo data", "Upload data file"))
if choice == "Try with demo data":
	global filename
	f_choice = st.sidebar.selectbox("Choose data file", ("-- select --", "Wine-data", "Housing-data"))
	if f_choice == "Wine-data":
		try:
			df = pd.read_csv("data_files/wine-data.csv")
			filename = "wine-data.csv"
			lt.empty()
		except:
			st.sidebar.error("Error while loading the file!")
			df = pd.DataFrame()
		finally:
			algorithm(df, "clf_demo")
	elif f_choice == "Housing-data":
		try:
			df = pd.read_csv("data_files/Housing-data.csv")
			filename = "Housing-data.csv"
			lt.empty()
		except:
			st.sidebar.error("Error while loading the file!")
			df = pd.DataFrame()
		finally:
			algorithm(df, "reg_demo")
elif choice == "Upload data file":
	upload_file()