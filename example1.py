
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import mean_squared_error

df = pd.read_csv("https://raw.githubusercontent.com/acherm/teaching-MDE1920/master/boston/boston.csv", index_col=0)
df.head()
X = df[["crim", "indus", "chas", "rm", "age", "tax", "ptratio", "lstat"]]
y = df["medv"]

test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

# !begin DecisionTreeRegressor
print(" ----- DecisionTreeRegression -----")
clf = tree.DecisionTreeRegressor()
clf.fit(X_train, y_train)

value1 = mean_squared_error(y_test, clf.predict(X_test))
value2 = r2_score(y_test, clf.predict(X_test))

print("Boston MSE = ", value1)
print("r2_score(y_true, y_pred) = ", value2)
# !end DecisionTreeRegression

