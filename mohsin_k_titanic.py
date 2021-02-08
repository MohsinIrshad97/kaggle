# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:59:39 2020

@author: FnH
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")

X = train_data.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values




y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch","Embarked","Age"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
X=imputer.fit_transform(X)
X_test = imputer.fit_transform(X_test)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('Mohsin_submission.csv', index=False)
print("Your submission was successfully saved!")