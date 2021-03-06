from sklearn.datasets import load_digits

digits = load_digits()

print(digits.data[:2])

print(digits.data.shape)

print(digits.target[:2])
print(digits.target.shape)

print(digits.images[:2])

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(6, 4))

for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)

plt.tight_layout()
# plt.show()
# function for zip is to iterate items (lists for example) together instead of with their
# own individual for loops

from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    digits.data, digits.target, random_state=11
)

print(data_train.shape)
print(data_test.shape)
print(target_train.shape)
print(target_test.shape)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X=data_train, y=target_train)
# x is all the features or input, Y is the target value or the output that the features represent

predicted = knn.predict(X=data_test)
# whole job of the predict is to give a target answer, so that is why we do not need a y

expected = target_test

print(predicted[:20])
print(expected[:20])

print(format(knn.score(data_test, target_test), ".2%"))

wrong = [(p, e) for (p, e) in zip(predicted, expected) if p != e]

print(wrong)

from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_true=expected, y_pred=predicted)

print(confusion)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt2

confusion_df = pd.DataFrame(confusion, index=range(10), columns=range(10))

figure = plt2.figure(figsize=(7, 6))
axes = sns.heatmap(confusion_df, annot=True, cmap=plt2.cm.nipy_spectral_r)

plt2.show()
