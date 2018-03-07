import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy
from sklearn.feature_selection import f_regression


H = np.loadtxt(r"/Users/YJccccc/PycharmProjects/DataMiningClass/data/housing.data", dtype=float)
X = H[:,0:13]
# X[:,12] = np.random.normal(0, 1, len(X))
y = H[:,13]

mlr = LinearRegression()
mlr.fit(X, y)

mintercept = mlr.intercept_
mslop = mlr.coef_
print("intercept: ", mintercept, "\nslops: ", mslop)

F, pval = f_regression(X, y)
print("F scores: ", F, "\np values: ", pval)

highSig = []
count = 0
for p in range (len(pval)):
    if pval[p] < 0.1:
        highSig.append(count)
    count += 1
print("Below are variables having at least a high significance:\n", highSig)

y_pred = mlr.predict(X)
plt.plot(X, y_pred, color = 'red')
plt.show()