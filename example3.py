
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

df = pd.read_csv("https://raw.githubusercontent.com/acherm/teaching-MDE1920/master/boston/boston.csv", index_col=0)
df.head()
X = df.drop(["medv"])
y = df["medv"]

train_size = 0.8
test_size = 1 - train_size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

# !begin OrdinaryLinearRegression
print(" ----- OrdinaryLinearRegression -----")
clf = linear_model.LinearRegression()
clf.fit(X_train, y_train)

value1 = mean_squared_error(y_test, clf.predict(X_test), multioutput="raw_values")

print("Parametized MSE for Ordinary linear regression = ", value1)
# !end OrdinaryLinearRegression