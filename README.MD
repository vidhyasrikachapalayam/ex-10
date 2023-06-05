EXP-10 Data Science Process on Complex Dataset
# AIM

To Perform Data Science Process on a complex dataset and save the data to a file.

# ALGORITHM
Step 1 Read the given Data

Step 2 Clean the Data Set using Data Cleaning Process

Step 3 Apply Feature Generation/Feature Selection Techniques on the data set

Step 4 Apply EDA /Data visualization techniques to all the features of the data set

# CODE
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

df.head()

df.isnull().sum()

plt.figure(figsize=(5,5))

plt.title("Data with Outliers")

df.boxplot()

plt.show()

plt.figure(figsize=(5,5))

cols = ['size','tip','total_bill']

Q1 = df[cols].quantile(0.25)

Q3 = df[cols].quantile(0.75)

IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

plt.title("Dataset after removing outliers")

df.boxplot()

plt.show()

df['sex'].unique()

!pip install --upgrade category_encoders

from category_encoders import BinaryEncoder

be = BinaryEncoder()

data = be.fit_transform(df['sex'])

df = pd.concat([df,data],axis=1)

df

df['smoker'].unique()

data = be.fit_transform(df['smoker'])

df = pd.concat([df,data],axis=1)

df

df['day'].unique()

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

clim = ['Thur','Fri','Sat','Sun']

en= OrdinalEncoder(categories = [clim])

df['day']=en.fit_transform(df[["day"]])

df

df['time'].unique()

le = LabelEncoder()

df['time'] = le.fit_transform(df[["time"]])

df

df.drop('sex',axis=1,inplace=True)

df.drop('smoker',axis=1,inplace=True)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df)

print("Min-max scaled data:")

print(scaled_data)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

print("Standard scaled data:")

print(scaled_data)

import seaborn as sns

sns.scatterplot(data=df)

sns.displot(df['size'],kde=True)

sns.scatterplot(x="total_bill", y="tip", data=df)

plt.title("Correlation between Tip Amount and Total Bill Amount")

plt.show()

df["tip_percent"] = df["tip"] / df["total_bill"]

sns.barplot(x=df['size'],y=df['tip_percent'],data=df)

plt.title("Tip Percentage by Dining Party Size"
sns.barplot(x=df['time'], y=df['total_bill'])

plt.title("Highest Total Bill Amount by Time")

plt.show()

df.corr()

sns.heatmap(df.corr(),annot=True)

# OUTPUT
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/b782efd8-b213-45df-9a0a-bb6e0195c664)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/3329bbee-ac83-4d40-b69e-b705df01128f)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/7ba2bfef-89b7-477c-b88a-d5ba9f68bc16)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/98643022-beec-499f-b83f-35eea5c96ac3)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/ea975646-5c15-403e-8d62-d70726606eac)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/3e74e13a-3ae8-4242-a663-b3b9e06263ba)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/e3c506e7-5c32-4f9a-bd6b-ae5527367116)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/1684468e-5503-4730-b69b-48fc97061ea1)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/9c67607a-4658-446e-9912-305bac94e6ab)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/04926346-3279-466f-a431-8bf484b7578c)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/7394e244-605e-46b7-b204-8ee14a8883e7)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/79c5c862-1afa-4bdf-986c-14a088004d2c)
![image](https://github.com/vidhyasrikachapalayam/ex-10/assets/119477817/52e99938-4519-4bbc-b613-0e3958a7a2e7)

# RESULT
Thus data science process on complex datascience is completed successfully

