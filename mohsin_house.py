# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 17:19:47 2020

@author: FnH
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")
X = train_data.iloc[:, :-1]
X_test =test_data
y = train_data.iloc[:, -1].values

l= train_data["SalePrice"]

X = pd.get_dummies(X)


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X=imputer.fit_transform(X)
X_test = imputer.fit_transform(X_test[:,3])
X_test[:,18] = imputer.fit_transform(X_test[:,18])


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)


