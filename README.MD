EXP-10 Data Science Process on Complex Dataset
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
np.random.seed(42)
data = np.random.normal(0, 1, 1000)
plt.figure(figsize=(8, 6))
plt.hist(data, bins=30, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()
