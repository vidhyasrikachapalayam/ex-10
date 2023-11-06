EXP-10 Data Science Process on Complex Dataset
# DATE:26/10/23

# AIM

To Perform Data Science Process on a complex dataset and save the data to a file.

# ALGORITHM
Step 1 Read the given Data

Step 2 Clean the Data Set using Data Cleaning Process

Step 3 Apply Feature Generation/Feature Selection Techniques on the data set

Step 4 Apply EDA /Data visualization techniques to all the features of the data set

# CODE
import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv("/content/Disease_symptom_and_patient_profile_dataset.csv")

df

df.head()

df.info()

df.tail()

df.isnull().sum()

df.shape

df.nunique()

sns.catplot(x = 'Outcome Variable' , y = 'Age' , data = df , kind = "swarm")

plt.figure(figsize=(20,10))

plt.figure(figsize=(20,10))

sns.barplot(x='Fever',y='Age',data =df);

sns.displot(df['Age'] , kde=True)

sns.catplot(x='Cough' , kind='count',data=df , hue = "Cholesterol Level")

df.groupby('Blood Pressure').size().plot(kind='pie', autopct='%.2f')

sns.catplot(x='Gender' , kind='count',data=df , hue = "Cholesterol Level")

df = df.iloc[:,1:]

df.groupby('Fatigue').size().plot(kind='pie', autopct='%.2f')

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:, [0, 2]]

Y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Example dataset
data = [[10, 20, 30],

[5, 15, 25],

[3, 6, 9],


[8, 12, 16]]
# Min-max scaling
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)

print("Min-max scaled data:")

print(scaled_data)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)

print("Standard scaled data:")

print(scaled_data)

np.random.seed(42

data = np.random.normal(0, 1, 1000)

plt.figure(figsize=(8, 6))

plt.hist(data, bins=30, edgecolor='black')

plt.xlabel('Value')

plt.ylabel('Frequency')

plt.title('Histogram')

plt.show()
# OUTPUT

![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/ff45c6ff-6360-4f72-ac42-db3c281b36a6)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/7b6fe9b0-c591-432d-be8b-258e367c67fd)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/72c02d56-48b2-4cf4-8f93-fd0527d8f485)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/1db34ad2-1a29-411e-ae42-791bf3bb0fb5)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/53533d5d-aedc-4b05-a802-c62cc55bf3d4)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/841666b8-5498-4e36-a859-fad58c218267)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/1e8aedcb-f9d9-4a94-88d9-5547e9199454)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/547472c3-2197-4663-a4dd-2bf4a39f1085)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/98455ac7-c067-44b9-96fe-4b64eba663a6)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/896ee2f6-32cd-45ce-8f9f-1ef88c52ce09)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/9c2114cf-43dc-4b50-aaf0-74801f2ae7d1)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/403cbe32-d76a-4fcd-bbb2-3b805893272a)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/4166cf45-a35c-4237-9e1c-297f23ca22bd)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/da35586e-d088-40ac-9533-cf5a1d836a77)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/6432f443-67fe-4064-9e7f-bb505b09a407)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/81e8ed28-a7b9-453e-97ca-0b47c5d00356)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/8f50af46-25ce-4a53-bb41-0749484267a8)

# RESULT
Thus Data Science Process on a complex dataset was performed successfully.
