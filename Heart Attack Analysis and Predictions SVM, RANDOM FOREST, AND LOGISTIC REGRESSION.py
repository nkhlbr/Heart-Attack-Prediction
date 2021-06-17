# -*- coding: utf-8 -*-
"""
Spyder Editor

Heart Attack Prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('heart.csv')

df

df.head()
df.info()

df.shape

df[df.duplicated()]  #Helps observe the duplicate row

df.drop_duplicates(inplace=True)

df.reset_index(drop=True, inplace=True)

df.shape


numerical_cols = ['age', 'trtbps', 'chol', 'thalach', 'oldpeak']
categorical_cols = ['sex', 'cp', 'caa', 'fbs', 'restecg', 'exng', 'slp', 'thall']
target = ['output']

plt.figure(figsize=(12,6))
sns.set(font_scale=1.4)
sns.set_style('whitegrid')
sns.countplot(x='output', data=df, palette='magma').set(xlabel="Output")
plt.title("Countplot of People with less or more chance of heart attack")

plt.figure(figsize=(12,6))
sns.countplot(x='sex', data=df, palette='magma')
plt.title('Countplot of Sex')

#Categorical Features

fig, ax = plt.subplots(2,3, figsize=(20,18))
sns.countplot(x='fbs', data=df, palette='magma', ax=ax[0][0]).set(title='Fasting Blood Sugar')
sns.countplot(x='exng', data=df, palette='magma', ax=ax[0][1]).set(title='Exercise Induced Angina')
sns.countplot(x='restecg', data=df, palette='magma', ax=ax[1][0]).set(title='Rest ECG')
sns.countplot(x='cp', data=df, palette='magma', ax=ax[0][2]).set(title='Chest Pain Type')
sns.countplot(x='caa', data=df, palette='magma', ax=ax[1][1]).set(title='Number of major vessels')
sns.countplot(x='thall', data=df, palette='magma', ax=ax[1][2]).set(title='Thallium Stress Test')

#Numerical Features
fig, ax = plt.subplots(2,2, figsize=(20,18))
sns.histplot(x=df["age"], ax=ax[0][0], color="red", kde=True).set(title='Age')
sns.histplot(x=df["trtbps"], ax=ax[0][1], color="blue", kde=True).set(title='Resting Blood Pressure')
sns.histplot(x=df["chol"], ax=ax[1][0], color="orange", kde=True).set(title='Cholestrol Levels')
sns.histplot(x=df["thalachh"], ax=ax[1][1], color="green", kde=True).set(title='Maximum Heart Rate Achieved')


plt.figure(figsize=(20,15))
matrix = np.triu(df.corr())
sns.heatmap(df.corr(), annot=True, cmap="Purples", mask=matrix)



fig, ax = plt.subplots(2,3, figsize=(22,18))
sns.kdeplot(x="age", data=df, hue="output", ax=ax[0][0], fill="True", palette="magma").set(title="Heart Attack related to Age")
sns.kdeplot(x="cp", data=df, hue="output", ax=ax[0][1], fill="True", palette="viridis").set(title="Heart Attack related to Chest Pain Types")
sns.kdeplot(x="thalachh", data=df, hue="output", ax=ax[1][0], fill="True", palette="viridis").set(title="Heart Attack related to Heart Rate")
sns.kdeplot(x="chol", data=df, hue="output", ax=ax[1][1], fill="True", palette="magma").set(title="Heart Attack related to Cholestrol")
sns.kdeplot(x="thall", data=df, hue="output", ax=ax[0][2], fill="True", palette="magma").set(title="Heart Attack related to Thallium Stress Test")
sns.kdeplot(x="trtbps", data=df, hue="output", ax=ax[1][2], fill="True", palette="viridis").set(title="Heart Attack related to Blood Pressure")




fig, ax = plt.subplots(2,2, figsize=(18, 15))
sns.boxplot(x="thall", y="thalachh", data=df, palette="magma", ax=ax[0][0]).set(title="Thallium Stress Test vs Max Heart Rate")
sns.boxplot(x="thall", y="trtbps", data=df, palette="viridis", ax=ax[0][1]).set(title="Thallium Stress Test vs Resting Blood Pressure")
sns.boxplot(x="thall", y="chol", data=df, palette="viridis", ax=ax[1][0]).set(title="Chest Pain Type vs Age")
sns.boxplot(x="cp", y="age", data=df, palette="magma", ax=ax[1][1]).set(title="Chest Pain Type vs Age")




#Data Preprocessing

# Splitting the data into Train and Test Data.
from sklearn.model_selection import train_test_split
X = df.drop('output', axis=1)
y = df['output']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

print("Shape of training set:", X_train.shape)
print("Shape of test set:", X_test.shape)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)



#Classification Models

#1. Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score


logmodel = LogisticRegression()


logmodel.fit(X_train, y_train)
predictions1 = logmodel.predict(X_test)


print("Confusion Matrix: \n", confusion_matrix(y_test, predictions1))
print('\n')
print(classification_report(y_test, predictions1))


logmodel_acc = accuracy_score(y_test, predictions1)
print("Accuracy of the Logistic Regression Model is: ", logmodel_acc)


#2.K Nearest Neighbours

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
predictions2 = knn.predict(X_test)
print(confusion_matrix(y_test, predictions2))
print("\n")
print(classification_report(y_test, predictions2))


error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate, color='orange', linestyle="--",marker='o', markersize=10, markerfacecolor='blue')
plt.title('Error_Rate vs K value')
plt.xlabel = ('K')
plt.ylabel = ('Error Rate')



knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predictions2 = knn.predict(X_test)

print(confusion_matrix(y_test, predictions2))
print("\n")
print(classification_report(y_test, predictions2))

knn_model_acc = accuracy_score(y_test, predictions2)
print("Accuracy of K Neighbors Classifier Model is: ", knn_model_acc)



#3.Random Forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
predictions3 = rfc.predict(X_test)


print("Confusion Matrix: \n", confusion_matrix(y_test, predictions3))
print("\n")
print(classification_report(y_test, predictions3))



#4 Support Vector Machines (SVM)

from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)
predictions4 = svc_model.predict(X_test)

print("Confusion Matrix: \n", confusion_matrix(y_test, predictions4))
print("\n")
print(classification_report(y_test, predictions4))

from sklearn.model_selection import GridSearchCV

param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}

grid = GridSearchCV(SVC(), param_grid, verbose=3)

grid.fit(X_train, y_train)

grid.best_params_

grid_predictions = grid.predict(X_test)


print("Confusion Matrix: \n", confusion_matrix(y_test, grid_predictions))
print("\n")
print(classification_report(y_test, grid_predictions))


svm_acc = accuracy_score(y_test, grid_predictions)
print("Accuracy of SVM model is: ", svm_acc)


#5 Naive Bayes

from sklearn.naive_bayes import GaussianNB

naive = GaussianNB()
naive.fit(X_train, y_train)
predictions5 = naive.predict(X_test)

print("Confusion Matrix: \n", confusion_matrix(y_test, predictions5))
print("\n")
print(classification_report(y_test, predictions5))

nb_acc = accuracy_score(y_test, predictions5)
print("Accuracy of SVM model is: ", nb_acc)

print("Accuracy of Logistic Regression Model is: ",logmodel_acc*100,"%")
print("Accuracy of K Nearest Neighbour Model is: ",knn_model_acc*100,"%")
print("Accuracy of Random Forests Model is: ",rfc_acc*100,"%")
print("Accuracy of SVM Model is: ",svm_acc*100,"%")
print("Accuracy of Naive Bayes Model is: ",nb_acc*100,"%")


plt.figure(figsize=(12,6))
model_acc = [logmodel_acc, knn_model_acc, rfc_acc, svm_acc, nb_acc]
name_of_model = ['LogisticRegression', 'KNN', 'RandomForests', 'SVM', 'Naive Bayes']
sns.barplot(x= model_acc, y=name_of_model, palette='Spectral')








































