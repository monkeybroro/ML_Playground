import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

price=np.genfromtxt("chepai.csv", delimiter=",")
x = price[:,0:2]
y = price[:,2]

#Simple linear regression
# print("Using linear regression..")
# reg = linear_model.LinearRegression()
# reg.fit(x, y)

# prediction = np.dot(x, reg.coef_)
# print(reg.coef_)
# print(y-prediction)

#Polynomial regression degree 2
print("Using 3 degree Polynomial regression..")
poly = PolynomialFeatures(degree=3)
X_ = poly.fit_transform(x)

clf = linear_model.LinearRegression()
clf.fit(X_, y)
#print(y-clf.predict(X_))

print('predicting..')
x_test = np.matrix([[12183, 226316]])
print(x_test)
x_test_ = poly.fit_transform(x_test)
print(clf.predict(x_test_))