#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 19:51:06 2024

@author: gracehenke
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Add datasets together
df1 = pd.read_csv("reg_stats.txt")
df2 = pd.read_csv("advanced.txt")
merged = pd.merge(df1, df2, on="Player")
FULL = merged.drop_duplicates(subset=('Player'))



# Define Feature [CF, CA, G, A, TK,BLK] and Target [PS]
features = FULL.drop(['Player', 'Age','S','Pos', 'GP','PS','Thru%', '-9999_y', '-9999_x'], axis = 1)
print(features)
target = FULL['PS']

x = features
y = target

x_train, x_test, y_train, y_test = train_test_split(x,y)


model = LinearRegression()

# Fit the model
model.fit(x_train, y_train)


# Display coefficients
print('Coefficients:', model.coef_)


# Make predictions
y_pred = model.predict(x_test)

plt.scatter(y_test, y_pred, color='#4E6CB4')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='#B7B09C', linewidth=2)
plt.xlabel('Actual PS Value')
plt.ylabel('Predicted PS Value')
plt.title('Actual vs Predicted PS')
plt.show()


# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('R-squared:', r2)






#PS = Point Share, points they contribute to the team
# Regression with PS = (CF * Wcf) + (CA * Wca) + (G * Wg) + (A * Wa) + (S * Ws) + (TK * Wtk) 

