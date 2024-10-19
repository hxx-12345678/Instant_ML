train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"train score: {{train_score:.2f}}")
print(f"test score: {{test_score:.2f}}")
print(f"Mean Absolute Error: {{mae:.2f}}")
print(f"Mean Squared Error: {{mse:.2f}}")
print(f"Root Mean Squared Error: {{rmse:.2f}}")
print(f"R2 Score: {{r2:.2f}}")

# plot graphs
def intr_plot_column_vs_actual_predicted(X_test, y_test, y_pred, column):
	ifig = go.Figure()
	ifig.add_trace(go.Scatter(x=X_test.iloc[:, column-1], y=y_test, mode='markers', name='Actual', marker=dict(color='blue')))
	ifig.add_trace(go.Scatter(x=X_test.iloc[:, column-1], y=y_pred, mode='lines', name='Predicted', line=dict(color='green')))
	ifig.update_layout(
	    title=f"Actual vs. Predicted for column {{column}}",
	    xaxis_title=f"X_test column {{column}}",
	    yaxis_title="Values",
	    legend=dict(x=0, y=1)
	)
	ifig.show()

def intr_plot_predicted_vs_actual(y_true, y_pred):
    ifig = go.Figure()
    ifig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predicted vs Actual', opacity=0.5))
    ifig.add_trace(go.Scatter(x=[min(y_true), max(y_true)], y=[min(y_true), max(y_true)], mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
    ifig.update_layout(title='Predicted vs. Actual Values', xaxis_title='Actual Values', yaxis_title='Predicted Values')
    ifig.show()

def intr_plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    ifig = go.Figure()
    ifig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals', opacity=0.5))
    ifig.add_trace(go.Scatter(x=[min(y_pred), max(y_pred)], y=[0, 0], mode='lines', name='Zero Residuals', line=dict(color='red', dash='dash')))
    ifig.update_layout(title='Residuals Plot', xaxis_title='Predicted Values', yaxis_title='Residuals')
    ifig.show()

def intr_plot_error_distribution(y_true, y_pred):
    errors = y_true - y_pred
    ifig = px.histogram(errors, nbins=50, title='Distribution of Prediction Errors')
    ifig.update_layout(xaxis_title='Prediction Error', yaxis_title='Count')
    ifig.show()

def plot_column_vs_actual_predicted(X_test, y_test, y_pred, column):
	fig = plt.figure(figsize=(10, 7))
	sns.scatterplot(x=X_test.iloc[:, column-1], y=y_test, color='b', label='Actual')
	sns.lineplot(x=X_test.iloc[:, column-1], y=y_pred, color='g', label='Predicted')
	plt.xlabel(f"X_test column {{column}}")
	plt.ylabel("Values")
	plt.title("Actual vs. Predicted for a perticular column")
	plt.legend()
	plt.show()

def plot_predicted_vs_actual(y_true, y_pred):
    fig = plt.figure(figsize=(10, 7))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs. Actual Values')
    plt.show()

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig = plt.figure(figsize=(10, 7))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.show()

def plot_error_distribution(y_true, y_pred):
    errors = y_true - y_pred
    fig = plt.figure(figsize=(10, 7))
    sns.histplot(errors, kde=True, color='blue')
    plt.xlabel('Prediction Error')
    plt.title('Distribution of Prediction Errors')
    plt.show()

interactive = True
if interactive:
	column = 1
	intr_plot_column_vs_actual_predicted(X_test, y_test, y_pred, column)
	intr_plot_predicted_vs_actual(y_test, y_pred)
	intr_plot_residuals(y_test, y_pred)
	intr_plot_error_distribution(y_test, y_pred)

else:
	column = 1
	plot_column_vs_actual_predicted(X_test, y_test, y_pred, column)
	plot_predicted_vs_actual(y_test, y_pred)
	plot_residuals(y_test, y_pred)
	plot_error_distribution(y_test, y_pred)