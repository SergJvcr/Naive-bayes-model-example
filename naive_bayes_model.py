import pandas as pd
from sklearn import naive_bayes, model_selection, metrics
import matplotlib.pyplot as plt

extracted_data = pd.read_csv('google_data_analitics\\extracted_nba_players_data.csv')

print(extracted_data.head(10))

# Model preparation
# Define the y (target) variable
y_variable = extracted_data['target_5yrs']
# Define the X (predictor) variables
X_variables = extracted_data.copy().drop(['target_5yrs'], axis=1)

print(y_variable.head(10))
print(X_variables.head(10))

# Perform the split operation on our data to get two subsets: train and test
# Assign the outputs as follows: X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_variables, y_variable, test_size=0.25,
                                                                   stratify=y_variable, random_state=42)

# Print the shape (rows, columns) of the output from the train-test split
# Print the shape of X_train
print(f'X_train contains {X_train.shape[0]} rows and {X_train.shape[1]} columns.')
# Print the shape of X_test
print(f'X_test contains {X_test.shape[0]} rows and {X_test.shape[1]} columns.')
# Print the shape of y_train
print(f'y_train contains {y_train.shape[0]} rows.') # there are no columns!!!
# Print the shape of y_test
print(f'y_test contains {y_test.shape[0]} rows.') # there are no columns!!!

# Model building
# Assign `nb` to be the appropriate implementation of Naive Bayes
nb = naive_bayes.GaussianNB()
# Fit the model on your training data
nb.fit(X_train, y_train)
# Apply your model to predict on your test data. Call this "y_pred"
y_pred = nb.predict(X_test)

# Results and evaluation
# Print an accuracy score
print(f'Accuracy score: {round(metrics.accuracy_score(y_test, y_pred), 2)}')
# Print a precision score
print(f'Precision score: {round(metrics.precision_score(y_test, y_pred), 2)}')
# Print a recall score
print(f'Recall score: {round(metrics.recall_score(y_test, y_pred), 2)}')
# Print a f1 score
print(f'F1 score: {round(metrics.f1_score(y_test, y_pred), 2)}')

# Gain clarity with the confusion matrix
# Construct and display a confusion matrix
# Construct the confusion matrix for predicted and test values
cm = metrics.confusion_matrix(y_test, y_pred)
# Create the display for your confusion matrix
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb.classes_)
# Plot the visual in-line
disp.plot(values_format='') # `values_format=''` suppresses scientific notation
plt.title('The Confusion Matrix for Predicted and Test Values')
plt.show()
