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
plt.show()
# function for zip is to iterate items (lists for example) together instead of with their
# own individual for loops
