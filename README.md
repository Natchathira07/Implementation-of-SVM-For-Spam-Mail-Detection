# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the program.

2.Import the required libraries such as pandas, sklearn, and chardet.

3.Detect the encoding format of the spam.csv file.

4.Load the dataset using pandas.

5.Separate the dataset into input (message text) and output (spam or ham label).

6.Split the data into training data and testing data using train_test_split.

7.Convert the text messages into numerical form using CountVectorizer.

8.Train the SVM (Support Vector Machine) model using the training data.

9.Use the trained model to predict whether messages are spam or not.

10.Calculate and display the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VD Natchathira
RegisterNumber: 212224230178
*/
import chardet

file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)

import pandas as pd
data = pd.read_csv('spam.csv', encoding='Windows-1252')

data.info()
print(data.isnull().sum())

x = data["v1"].values
y = data["v2"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

print("Predicted values:")
print(y_pred)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
```

## Output:
## ENCODING
<img width="855" height="47" alt="image" src="https://github.com/user-attachments/assets/556e91b9-cc1b-4be4-8731-fad555435879" />

## HEAD():
<img width="907" height="242" alt="image" src="https://github.com/user-attachments/assets/a2333e01-667d-4ed2-b834-12d1acb70d75" />

## isnul().sum:
<img width="321" height="192" alt="image" src="https://github.com/user-attachments/assets/ecc6b550-634b-4428-be3d-3f8b0d056e40" />

## Prediction of Y:
<img width="778" height="89" alt="image" src="https://github.com/user-attachments/assets/f841c03a-1a15-4707-87e9-795c19f49843" />

## Accuracy:
<img width="307" height="48" alt="image" src="https://github.com/user-attachments/assets/f08bbb27-d2b8-494b-9b44-4ac36c80a071" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
