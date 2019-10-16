
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree, linear_model
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("https://raw.githubusercontent.com/acherm/teaching-MDE1920/master/boston/boston.csv", index_col=0)
df.head()
X = df.drop(["medv"])
y = df["medv"]

test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

# !begin DecisionTreeRegression
print(" ----- DecisionTreeRegression -----")
clf = tree.DecisionTreeRegressor()
clf.fit(X_train, y_train)

value1 = mean_squared_error(y_test, clf.predict(X_test))

print("Boston MSE from DTR = ", value1)
# !end DecisionTreeRegression


# !begin BayesianLinearRegression
print(" ----- BayesianLinearRegression -----")
clf = linear_model.BayesianRidge()
clf.fit(X_train, y_train)

value1 = r2_score(y_test, clf.predict(X_test))

print("Boston R2 score from Bayesian = ", value1)
# !end DecisionTreeRegressor