# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 22:08:44 2019

@author: Nisar
"""
#importing the datasets
import pandas as pd
df = pd.read_csv("titanic.csv")
df.head()

#dropping the unnecessary attributes
inputs = df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked','Survived'],axis='columns')

#setting the target attribute
target = df['Survived']

#importing sklearn library
from sklearn.preprocessing import LabelEncoder

#labelling the non integer attributes
le_Sex = LabelEncoder()
inputs['Sex_n'] = le_Sex.fit_transform(inputs['Sex'])

#dropping the non integer atrributes
inputs_n = inputs.drop(['Sex'],axis='columns')

#filling missing values to '0'
inputs_n = inputs_n.fillna(0) 

#importing tree to implement decision tree
from sklearn import tree

#generating the model
model = tree.DecisionTreeClassifier()

#fitting the model to input and target
model.fit(inputs_n, target)

#checking the accuracy of the above model
model.score(inputs_n,target)

#testing the above model

import numpy as np

Pclass_user = input("Enter the passenger's class : ")
Age_user = input("Enter the passenger's age : ")
Fare_user = input("Enter the passenger's fare : ")
Sex_user = input("Enter the passenger's sex{'0 for male' and '1 for female'} : ")

output = model.predict([[Pclass_user,Age_user,Fare_user,Sex_user]])

if np.equal(output,[0]):
     print("The passenger died!")
else:
    print("The passenger survived")

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
