""" Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line """

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets


# how many sameples and How many features?
db = datasets.load_diabetes()
print(db)
print(db.target)
print(db.data.shape)


# What does feature s6 represent?
print(db.DESCR)

# print out the coefficient
data_train, data_test, target_train, target_test = train_test_split(
    db.data, db.target, random_state=11
)
"""
print(data_train.shape)
print(data_test.shape)
print(target_train.shape)
print(target_test.shape)
"""
lr = LinearRegression()

lr.fit(X=data_train, y=target_train)

print(lr.coef_)

# print out the intercept
print(lr.intercept_)

# create a scatterplot with regression line

predicted = lr.predict(data_test)

expected = target_test

for p, e in zip(predicted[::5], expected[::5]):
    print(f"predicted: {p:.2f}, expected: {e:.2f}")

# used for score
print(format(lr.score(data_test, target_test), ".2%"))


plt.plot(expected, predicted, ".")

x = np.linspace(0, 330, 100)
y = x
line = plt.plot(x, y)
plt.show()
