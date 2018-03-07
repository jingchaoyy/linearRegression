import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

H = np.loadtxt(r"/Users/YJccccc/PycharmProjects/DataMiningClass/data/housing.data", dtype=float)

X = H[:,0]
y = H[:,13]

# print(X)
# print('#############')
# print(X.reshape(-1,1))

plt.scatter(X, y)
plt.xlabel("H0")
plt.ylabel("H13")
# plt.show()

lr = LinearRegression()
lr.fit(X.reshape(-1,1), y)

intercept = lr.intercept_
slop = lr.coef_

h0 = 0.5
h13 = intercept + slop * h0

print("intercept = ", intercept, "\nslop = ", slop)
print("When a new suburb having a crime rate H0 of 0.5, the expected median value \
of owner-occupied homes in this suburb will be ", h13)

y_pred = lr.predict(X.reshape(-1,1))
plt.plot(X, y_pred, color = 'red')

plt.show()

