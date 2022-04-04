from sklearn.datasets import load_digits

digits = load_digits()

print(digits.data[:2])
print(digits.data.shape)
