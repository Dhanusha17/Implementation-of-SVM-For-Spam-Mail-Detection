# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:

/*
Program to implement the SVM For Spam Mail Detection..

Developed by: DHANUSHA K

RegisterNumber: 212223040034  
*/
```
import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
## data.head():
![326377770-e40d1f85-d42b-44b2-8d83-b0be2d271f79](https://github.com/Dhanusha17/Implementation-of-SVM-For-Spam-Mail-Detection/assets/151549957/5729c738-d4a2-4d23-b48e-ac94abe452b8)

## data.info():
![326377904-8589603c-b315-497a-bf65-595a390810b0](https://github.com/Dhanusha17/Implementation-of-SVM-For-Spam-Mail-Detection/assets/151549957/acd967a9-2730-4474-b6b1-45a0c2d7b7d6)

## y_predict:
![326378047-196d717e-b11b-4357-b4bc-917e2f3c89d6](https://github.com/Dhanusha17/Implementation-of-SVM-For-Spam-Mail-Detection/assets/151549957/ea3570cc-0ec6-4f31-9023-e46e4dd3c721)

## Accuracy:
![326378132-6b00736d-9efd-4e5b-9322-b8f1650d68f7](https://github.com/Dhanusha17/Implementation-of-SVM-For-Spam-Mail-Detection/assets/151549957/c24fb6ff-1d1e-4f8e-821e-e3abc70d7883)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
