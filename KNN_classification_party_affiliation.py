# Project: Predicting Party Affiliation using KNN (binary classification)
# Dataset: https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
# to predict US House of Representatives Congressmen's party affiliation ('Democrat' or 'Republican') 
# based on how they voted on certain key issues

# Import necessary modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# depreciated: from sklearn.preprocessing import Imputer

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 

# Load data
df = pd.read_csv('datasets/house_votes_84.csv')  # header default is 'infer'

# data inspection: no need for scaling/centering as data is binary 

# # Convert '?' to NaN
# df[df == '?'] = np.nan

# # Print the number of NaNs
# print(df.isnull().sum())

# Fix the missing values ? with a imputer
imp = SimpleImputer(missing_values='?', strategy='most_frequent')
df[:] = imp.fit_transform(df)  # fit_transform() return a 2D numpy array, so assign the values back 

# Visual Exploratory Data Analysis (EDA)
plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Encode the labels (y/n) to True/False - can also use LabelEncoder()
X[:] = X[:] == 'y'

# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the train set
knn.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = knn.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))
