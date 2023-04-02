# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for Gradient Design.
2.Set variables for assigning dataset values.
3.Import LinearRegression from the sklearn.
4.Assign the points for representing the graph.
5.Predict the regression for marks by using the representation of graph.
6.Compare the graphs and hence we obtain the LinearRegression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Divakar R
RegisterNumber: 212222240026 
*/
/*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
#displaying the content in datafile 
df.head()
df.tail()
#segregating data to variable 
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
#splitting train anbd train data 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test
#graph plot for training data 
plt.scatter (x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("hours vs Scores (training set)")
plt.xlabel("hours")
plt.ylabel("scores")
plt.show()
#graph plot for test data 
plt.scatter (x_test,y_test,color="orange")
plt.plot(x_test,regressor.predict(x_test),color="red")
plt.title("hours vs Scores (test set)")
plt.xlabel("hours")
plt.ylabel("scores")
plt.show()
#graph plot for test data 
plt.scatter (x_test,y_test,color="orange")
plt.plot(x_test,regressor.predict(x_test),color="red")
plt.title("hours vs Scores (test set)")
plt.xlabel("hours")
plt.ylabel("scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
*/
## Output:
df.head


![Screenshot 2023-04-02 145220](https://user-images.githubusercontent.com/121932143/229348958-4ea1a3a7-a1a3-491e-b52b-e6e0c82008b5.png)


df.ta!
![image](https://user-images.githubusercontent.com/121932143/229344637-92ae6ae0-d302-4dee-a036-8d52484478e9.png)
value of X:

![image](https://user-images.githubusercontent.com/121932143/229344660-de5c7901-4dbd-428c-886d-75d9142c1fcb.png)
value of y:
![image](https://user-images.githubusercontent.com/121932143/229344700-15942d83-e2d2-46b4-b7be-22ff69629257.png)
predicted value of Y:
![image](https://user-images.githubusercontent.com/121932143/229344732-fededdd8-1a01-47e9-a3ff-45d307463a30.png)

Tested value of Y:
![image](https://user-images.githubusercontent.com/121932143/229344774-9135e2c0-80aa-4a3d-8316-e7d9ebd85acb.png)
Graph for Training Set:
![image](https://user-images.githubusercontent.com/121932143/229344809-eb10199e-58f5-48f6-9d1e-2b70daa95128.png)
Graph for Test Set:
![image](https://user-images.githubusercontent.com/121932143/229344853-0e78a933-b4e8-44c1-b8e9-c78d923e0ddd.png)
Value for MSE, MAE, RMSE:
![image](https://user-images.githubusercontent.com/121932143/229344894-8a805956-d05e-471b-a884-56cbd466b773.png)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
