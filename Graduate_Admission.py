##Dataset downloaded from https://www.kaggle.com/mohansacharya/graduate-admissions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.stats import pearsonr
from os import chdir
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
chdir('c:\\users\\jeryl\\desktop\\Python\\kaggle\\Graduate admissions')

#Reading data
train = pd.read_csv('Admission.csv', header=0)
#Dropping irrelevant columns
train =  train.drop(['Serial No.'], axis=1)
#Split data into features and labels
X,y=train.iloc[:,0:7], train.iloc[:,7]

#EDA
corr = train.corr()
sns.heatmap(corr, annot=True, vmin=0, vmax=1)
plt.show()

#Scaler is not necessary as it does not affect prediction results
scaler = MinMaxScaler(feature_range=(0,1)) 
rescaledX = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(rescaledX, y, test_size=0.2, random_state=42)

#Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(mean_squared_error(y_test, y_pred))
print(lr.score(X_test,y_test))
print(lr.coef_)
print(lr.intercept_)

#Cross-Validation
cv_results = cross_val_score(lr, rescaledX, y, cv=5, scoring='neg_mean_squared_error')
print(np.mean(cv_results))

#Graph showing closeness of prediction to actual results
x = np.arange(1,101)
plt.plot(x, y_test, label='Actual', color='r')
plt.plot(x, y_pred, ':b',label='Predicted', color='b')
plt.legend()
plt.show()  

#Finding standard errors of coefficients
features = sm.add_constant(X_train)
model = sm.OLS(y_train, features)
results = model.fit()
print(results.summary())
