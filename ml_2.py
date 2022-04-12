import pandas as pd
from sklearn.model_selection import train_test_split

nyc = pd.read_csv("ave_hi_nyc_jan_1895-2018.csv")

print(nyc.head(3))

# below not yet in a format necessary for data model.
# need it in two dimensions even though we only have one
# relevant set of values
print(nyc.Date.values)

# below (-1 means to infer the number of rows, 1 gives it the number of columns)
print(nyc.Date.values.reshape(-1, 1))


X_train, X_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1, 1), nyc.Temperature.values, random_state=11
)

print(X_train.shape)
print(X_test.shape)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X=X_train, y=y_train)

print(lr.coef_)
print(lr.intercept_)


predicted = lr.predict(X_test)

expected = y_test

for p, e in zip(predicted[::5], expected[::5]):
    print(f"predicted: {p:.2f}, expected: {e:.2f}")

print(format(lr.score(X_test, y_test), ".2%"))

predict = lambda x: lr.coef_ * x + lr.intercept_

print(predict(2025), predict(1890))

import seaborn as sns

axes = sns.scatterplot(
    data=nyc,
    x="Date",
    y="Temperature",
    hue="Temperature",
    palette="winter",
    legend=False,
)

axes.set_ylim(10, 70)

import numpy as np

x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
print(x)
y = predict(x)
print(y)

import matplotlib.pyplot as plt

line = plt.plot(x, y)
plt.show()
