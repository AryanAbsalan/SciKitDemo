#!/usr/bin/env python
# coding: utf-8

# # Linear Regression 
# The objective of the least squares method is to find values of α and β
# that minimise the sum of the squared difference between Y and Yₑ. 
# We will not go through the derivation here, but using calculus we can show that 
# the values of the unknown parameters are as follows:
# Beta = sum of (x item - mean of x)(y item - mean of y) / (x item - mean of x)
# Alpha = mean of y -( beta * mean of x) 
# 
# where X̄ is the mean of X values and Ȳ is the mean of Y values.

# If you are familiar with statistics, you may recognise β as simply
# Cov(X, Y) / Var(X).

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Generate 'random' data
np.random.seed(0)
X = 2.5 * np.random.randn(100) + 1.5   # Array of 100 values with mean = 1.5, stddev = 2.5
res = 0.5 * np.random.randn(100)       # Generate 100 residual terms
y = 2 + 0.3 * X + res                  # Actual values of Y

# Create pandas dataframe to store our X and y values
df = pd.DataFrame({'X': X, 'y': y})

# Show the first five rows of our dataframe
print(df.head())

# Calculate the mean of X and y
xmean = np.mean(X)
ymean = np.mean(y)

# Calculate the terms needed for the numator and denominator of beta
df['xycov'] = (df['X'] - xmean) * (df['y'] - ymean)

df['xvar'] = (df['X'] - xmean)**2
df['yvar'] = (df['y'] - ymean)**2

#print('Var of X', df['xvar'].sum())
#print('Var of y', df['yvar'].sum())

# Calculate beta and alpha
beta = df['xycov'].sum() / df['xvar'].sum()
alpha = ymean - (beta * xmean)
#print(f'alpha = {alpha}')
#print(f'beta = {beta}')

# Yₑ = 2.003 + 0.323 X
# For example, if we had a value X = 10, we can predict that:
# Yₑ = 2.003 + 0.323 (10) = 5.233.

ypred = alpha + beta * X

# Let’s plot our prediction ypred against the actual values of y,to get a better visual understanding of our model.

# Plot regression against actual data
plt.figure(figsize=(12, 6))
plt.plot(X, ypred)     # regression line
plt.plot(X, y, 'ro')   # scatter plot showing actual data
plt.title('Actual vs Predicted')
plt.xlabel('X')
plt.ylabel('y')

#plt.show()


# #  multiple linear regression model
# Yₑ = α + β₁X₁ + β₂X₂ + … + βₚXₚ, where p is the number of predictors

# Import and display first five rows of advertising dataset
advert = pd.read_csv(r'C:\Users\Aryan.ABSALAN\MLGO\Advertising.csv', encoding="utf-8")
advert.head()

from sklearn.linear_model import LinearRegression

# Build linear regression model using TV and Radio as predictors
# Split data into predictors X and output Y
# Also : 
# predictors = ['TV', 'Radio']
# X = advert[predictors]

X= advert[['TV', 'Radio']]
y = advert['Sales']

# Initialise and fit model
lm = LinearRegression()
model = lm.fit(X, y)

# After building a model there is no need to calculate the values for alpha and betas ourselves.
# we just have to call 

# .intercept_ for alpha, and 
# .coef_ for an array with our coefficients beta1 and beta2

# print(f'alpha of Advertising dataset = {model.intercept_}')
# print(f'betas of Advertising dataset = {model.coef_}')
# print(f'betas of Advertising dataset = {model.coef_[0]}')

# Sales = α + β₁*TV + β₂*Radio
# Sales = 2.921 + 0.046*TV + 0.1880*Radio.

ypred = model.predict(X)
#print(ypred)

new_X = [[300, 200]]
#print(model.predict(new_X))


