##lab00: report format

## Intro

## Methods
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from joblib import dump

## Data
penguins_train = pd.read_csv("https://cs307.org/lab-00/data/penguins-train.csv")
penguins_test = pd.read_csv("https://cs307.org/lab-00/data/penguins-test.csv")

## What data is?
##

## Summary Statistics

## Visualization (check lab policy!!!)
## 


## Models (the ML!)
## process data for ML
X_train = penguins_train[["bill_length_mm","bill_depth_mm"]]
Y_train = penguins_train["species"]
X_test = penguins_test[["bill_length_mm","bill_depth_mm"]]
Y_test = penguins_test["species"]


## train models
dummy_clf = DummyClassifier()
dt_clf = DecisionTreeClassifier()

dummy_clf.fit(X_train, Y_train)
dt_clf.fit(X_train, Y_train)

## Results
## report model metrics
print(np.mean(Y_test.to_numpy() == dummy_clf.predict(X_test))) ## = 0.48
print(np.mean(Y_test.to_numpy() == dt_clf.predict(X_test))) # = 0.94

# say result: 



# save models
dump(dummy_clf, "penguins-dummy.joblib")
dump(dt_clf, "penguins-dt.joblib")
# Discussion: 
# Shall I use this model? Shall people use this method? "usually say no: bc 94% here is terrible"
## Summary 



