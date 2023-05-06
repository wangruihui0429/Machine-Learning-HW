import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv('HW1-Linear regression.csv')
#df = pd.read_csv('HW1-Linear regression.csv', error_bad_lines=False)
# Step 2: Split the dataset into training and testing sets
X_train = df['review'][:2]
y_train = df['rating'][:2]
X_test = df['review'][2:]
y_test = df['rating'][2:]

# Step 3: Create a CountVectorizer object and fit it to the training data
cv = CountVectorizer()
cv.fit(X_train)

# Step 4: Transform the training and testing data using the CountVectorizer
X_train_cv = cv.transform(X_train)
X_test_cv = cv.transform(X_test)

# Step 5: Train a linear regression model using the transformed training data
lr = LinearRegression()
lr.fit(X_train_cv, y_train)

# Step 6: Make predictions on the transformed testing data
y_pred = lr.predict(X_test_cv)

# Step 7: Evaluate the performance of the model
print(y_pred)
print('Mean squared error: ', mean_squared_error(y_test, y_pred))
print('R2 score: ', r2_score(y_test, y_pred))
