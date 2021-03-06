# Project: Life Expectancy Prediction with Linear Regression
# Dataset: https://www.gapminder.org/data/
# Goal: to predict the life expectancy in a given country using Gapminder data.

# Import numpy and pandas
import numpy as np
import pandas as pd

# Import LinearRegression, a common model evaluation metrics, training neccesity
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error  # Root Mean Squared Error (RMSE)
from sklearn.model_selection import cross_val_score  # Cross Validation

# Read the CSV file into a DataFrame: df
df = pd.read_csv('datasets/gapminder.csv')

# Create arrays for features and target variable
y = df['life']
X = df['fertility']

# EDA
# Print the dimensions of y and X before reshaping
print("Dimensions of y before reshaping: ", y.shape)
print("Dimensions of X before reshaping: ", X.shape)

# Reshape X and y
y_reshaped = y.reshape(-1,1)
X_reshaped = X.reshape(-1,1)

# Print the dimensions of y_reshaped and X_reshaped
print("Dimensions of y after reshaping: ", y_reshaped.shape)
print("Dimensions of X after reshaping: ", X_reshaped.shape)

# Heatmap
sns.heatmap(df.corr(), square=True, cmap='RdYlGn')

# Linear Regression plotting 

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# Fit the model to the data
reg.fit(X_fertility, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2 
print(reg.score(X_fertility, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()

# Train/test/split 
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

# Model Evaluation - cross validation
# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

