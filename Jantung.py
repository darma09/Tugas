# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Z7tLVUFWIA514YR3TRWWzynP12iGkX4t
"""

drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

filePath = '/content/drive/MyDrive/data/heart.csv'

data = pd.read_csv(filePath)

data.head(5)

print("(Rows, columns): " + str(data.shape))
data.columns

data.nunique(axis=0)# returns the number of unique values for each variable.

#summarizes the count, mean, standard deviation, min, and max for numeric variables.
data.describe()

# Display the Missing Values

print(data.isna().sum())

data['target'].value_counts()

# calculate correlation matrix

corr = data.corr()
plt.subplots(figsize=(15,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
sns.heatmap(corr, xticklabels=corr.columns,
            yticklabels=corr.columns,
            annot=True,
            cmap=sns.diverging_palette(220, 20, as_cmap=True))

subData = data[['age','trestbps','chol','thalach','oldpeak']]
sns.pairplot(subData)

sns.catplot(x="target", y="oldpeak", hue="slope", kind="bar", data=data);

plt.title('ST depression (induced by exercise relative to rest) vs. Heart Disease',size=25)
plt.xlabel('Heart Disease',size=20)
plt.ylabel('ST depression',size=20)

plt.figure(figsize=(12,8))
sns.violinplot(x= 'target', y= 'oldpeak',hue="sex", inner='quartile',data= data )
plt.title("Thalach Level vs. Heart Disease",fontsize=20)
plt.xlabel("Heart Disease Target", fontsize=16)
plt.ylabel("Thalach Level", fontsize=16)

plt.figure(figsize=(12,8))
sns.boxplot(x= 'target', y= 'thalach',hue="sex", data=data )
plt.title("ST depression Level vs. Heart Disease", fontsize=20)
plt.xlabel("Heart Disease Target",fontsize=16)
plt.ylabel("ST depression induced by exercise relative to rest", fontsize=16)

# Filtering data by POSITIVE Heart Disease patient
pos_data = data[data['target']==1]
pos_data.describe()

# Filtering data by NEGATIVE Heart Disease patient
neg_data = data[data['target']==0]
neg_data.describe()

print("(Positive Patients ST depression): " + str(pos_data['oldpeak'].mean()))
print("(Negative Patients ST depression): " + str(neg_data['oldpeak'].mean()))

print("(Positive Patients thalach): " + str(pos_data['thalach'].mean()))
print("(Negative Patients thalach): " + str(neg_data['thalach'].mean()))

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression(random_state=1) # get instance of model
model1.fit(x_train, y_train) # Train/Fit model

y_pred1 = model1.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred1)) # output accuracy

from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

model2 = KNeighborsClassifier() # get instance of model
model2.fit(x_train, y_train) # Train/Fit model

y_pred2 = model2.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred2)) # output accuracy

from sklearn.metrics import classification_report
from sklearn.svm import SVC

model3 = SVC(random_state=1) # get instance of model
model3.fit(x_train, y_train) # Train/Fit model

y_pred3 = model3.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred3)) # output accuracy

from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

model4 = GaussianNB() # get instance of model
model4.fit(x_train, y_train) # Train/Fit model

y_pred4 = model4.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred4)) # output accuracy

from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

model5 = DecisionTreeClassifier(random_state=1) # get instance of model
model5.fit(x_train, y_train) # Train/Fit model

y_pred5 = model5.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred5)) # output accuracy

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

model6 = RandomForestClassifier(random_state=1)# get instance of model
model6.fit(x_train, y_train) # Train/Fit model

y_pred6 = model6.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred6)) # output accuracy

from xgboost import XGBClassifier

model7 = XGBClassifier(random_state=1)
model7.fit(x_train, y_train)
y_pred7 = model7.predict(x_test)
print(classification_report(y_test, y_pred7))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred6)
print(cm)
accuracy_score(y_test, y_pred6)

# get importance
importance = model6.feature_importances_

# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

index= data.columns[:-1]
importance = pd.Series(model6.feature_importances_, index=index)
importance.nlargest(13).plot(kind='barh', colormap='winter')

# Assuming you have a trained model called 'model6' and a scaler 'sc'
# (umur, jenis kelamin, jenis nyeri dada, tekanan darah istirahat, kolesterol, gula darah puasa,
# elektrokardiogram istirahat, detak jantung maksimal, angina yang dipicu olahraga, depresi ST,
# kemiringan segmen ST, jumlah pembuluh darah utama yang berwarna, hasil tes thalassemia)

# Input data (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
new_data = [[20, 1, 2, 110, 230, 1, 1, 140, 1, 2.2, 2, 0, 2]]

# Transform the input data using the same scaler
scaled_data = sc.transform(new_data)

# Predict using the model
prediction = model6.predict(scaled_data)

# Interpret the prediction
if prediction == 1:
    print("The result indicates a risk of heart disease.")
else:
    print("The result suggests no significant risk of heart disease.")

# Prediksi nilai target (y) menggunakan model
y_pred = model6.predict(x_test)

# Menggabungkan hasil prediksi dengan nilai sebenarnya (y_test)
hasil_gabungan = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1)

# Menampilkan hasil
print(hasil_gabungan)

if __name__ == '__main__':
    st.title('Heart Disease Classification')
    st.write('This app predicts the presence of heart disease based on various features.')

    # Add your Streamlit code here
    # ...

    # Example Streamlit code
    st.subheader('Prediction')
    new_data = [[20, 1, 2, 110, 230, 1, 1, 140, 1, 2.2, 2, 0, 2]]
    scaled_data = sc.transform(new_data)
    prediction = model6.predict(scaled_data)

    if prediction == 1:
        st.write("The result indicates a risk of heart disease.")
    else:
        st.write("The result suggests no significant risk of heart disease.")

    st.subheader('Prediction Results')
    st.write(hasil_gabungan)