# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Read the data frame using pandas. 
3. Get the information regarding the null values present in the dataframe.
4. Split the data into training and testing sets.
5. Convert the text data into a numerical representation using CountVectorizer.
6. Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.
7. Finally, evaluate the accuracy of the model.
 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: DHANUSHA K

RegisterNumber:  212223040034
*/
```
```
import chardet 
file='spam.csv'
with open(file, 'rb') as rawdata: 
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()

X = data["v1"].values
Y = data["v2"].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print("Y_prediction Value: ",Y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy
```

## Output:
## Result Output
![329154757-d9448238-e689-45df-9f15-8166741072ee](https://github.com/AkilaMohan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/151549957/3979a9bb-719a-44cb-b055-5804cde2a27d)


## data.head()
![329154836-e584fcd5-2ef9-4fd4-a57e-5cc1369d522e](https://github.com/AkilaMohan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/151549957/7560c6aa-c1a7-4873-ba91-60638c8d5190)

## data.isnull().sum()
![329155446-2fe8b559-6efd-40ff-801c-d75a9197cce2](https://github.com/AkilaMohan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/151549957/ccc5374b-d52a-4979-a88e-9040a3a8e0f3)

## Y_prediction Value
![329155395-c5b8927f-6f8e-4528-aed8-7bb311aca973](https://github.com/AkilaMohan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/151549957/1691c7e7-dc46-42d2-b00b-fdd61606f47b)

## Accuracy Value
![329155362-4b26c0c7-75bb-41b5-9b46-ac38be5fac37](https://github.com/AkilaMohan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/151549957/09d49bab-fb89-4843-97a1-845c8d061a36)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
