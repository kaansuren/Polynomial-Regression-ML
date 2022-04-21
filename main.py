# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading Csv
data = pd.read_csv("rank_salary.csv")
print(data)

x = data.iloc[:,1:2]
y = data.iloc[:,2:]

# Converting to Numpy arrays
X = x.values
Y = y.values

# Creating Linear Regression Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

#Creating Polynomial Regression (Non-Linear) Model
from sklearn.preprocessing import PolynomialFeatures
p_reg = PolynomialFeatures(degree = 2)
x_poly = p_reg.fit_transform(X)
print(x_poly)

lr.fit(x_poly, y)

# Visualization
plt.scatter(X, Y)
plt.plot(x, lr.predict(x_poly))
plt.show()

# Increasing Degree

lr2 = LinearRegression()
p_reg2 = PolynomialFeatures(degree = 4)
x_poly2 = p_reg2.fit_transform(X)
print(x_poly2)

lr2.fit(x_poly2, y)

# Visualization - 2
plt.scatter(X, Y)
plt.plot(x, lr2.predict(x_poly2))
plt.show()

print("Degree 2 - level 5.5:",lr.predict(p_reg.fit_transform([[5.5]])))
print("Degree 4 - level 5.5:",lr2.predict(p_reg2.fit_transform([[5.5]])))




















