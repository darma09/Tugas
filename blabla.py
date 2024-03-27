# -*- coding: utf-8 -*-
"""blabla.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dbLePIPA6J3vFkAxR32j69SHYuTGzMot
"""

import pandas as pd
filePath = 'https://raw.githubusercontent.com/darma09/Tugas/main/heart.csv'
data = pd.read_csv(filePath)

data

data.shape

data.describe()

data.isnull().sum()

# Display the Missing Values

print(data.isna().sum())

data['target'].value_counts()

data.info()

#let's transform categorical values into dummies/Convert categorical variable into dummy/indicator variables.

data = pd.get_dummies(data, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

data

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

y = data['target']

X = data.drop('target',axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
LR_classifier = LogisticRegression(random_state=0)
clf = svm.SVC()
sgd=SGDClassifier()
forest=RandomForestClassifier(n_estimators=20, random_state=12,max_depth=6)
treee = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth = 6)
LR_classifier.fit(X_train, y_train)
clf.fit(X_train, y_train)
sgd.fit(X_train, y_train)
treee.fit(X_train, y_train)
forest.fit(X_train, y_train)

print(forest)

#traing accuracy
y_pred=LR_classifier.predict(X_train)
y_predsvm=clf.predict(X_train)
y_predsgd=sgd.predict(X_train)
y_predtree=treee.predict(X_train)
y_predforest=forest.predict(X_train)

print(accuracy_score(y_train, y_pred))
print(accuracy_score(y_train, y_predsvm))
print(accuracy_score(y_train, y_predsgd))
print(accuracy_score(y_train, y_predtree))
print(accuracy_score(y_train, y_predforest))

#test accuracy
y_pred=LR_classifier.predict(X_test)
y_predsvm=clf.predict(X_test)
y_predsgd=sgd.predict(X_test)
y_predtree=treee.predict(X_test)
y_predforest=forest.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(accuracy_score(y_test, y_predsvm))
print(accuracy_score(y_test, y_predsgd))
print(accuracy_score(y_test, y_predtree))
print(accuracy_score(y_test, y_predforest))

accuracy_score(y_test, y_predsvm)

accuracy_score(y_test, y_predsgd)

accuracy_score(y_test, y_predtree)

accuracy_score(y_test, y_predforest)

import pickle
pickle.dump(forest, open('Random_forest_model.pkl', 'wb'))
with open('Random_forest_model.pkl', 'rb') as file:
    load_clf = pickle.load(file)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pickle
import streamlit as st

st.write("""
# Heart disease Prediction App

This app predicts if a patient has a heart disease.

Information:
1. Masukan umur kalian (Enter your age)
2. Jenis kelamin kalian (0: laki-laki, 1: perempuan) (Enter your gender: 0 for male, 1 for female)
3. Jenis nyeri dada (0: angina tipikal, 1: angina atipikal, 2: nyeri non-angina, 3: tanpa gejala) (Enter chest pain type: 0 for typical angina, 1 for atypical angina, 2 for non-anginal pain, 3 for asymptomatic)
    or visit website: https://www.halodoc.com/artikel/ketahui-4-jenis-jenis-nyeri-dada-yang-perlu-diwaspadai
4. Tekanan darah istirahat (Enter resting blood pressure)
5. Kolesterol serum dalam mg/dl (Enter serum cholestoral in mg/dl)
6. Gula darah puasa > 120 mg/dl (1: true, 0: false) (Enter fasting blood sugar: 1 for true, 0 for false)
7. Gula darah sehingga hasil elektrokardiografi istirahat > 90 mg/dl (1: normal, 0: abnormal) (Enter resting electrocardiographic results: 1 for normal, 0 for abnormal)
8. EKG epidimiokardiografi maksimum yang dicapai adalah ≥ 50% QRS normal (1: ya, 0: tidak) (Enter maximum heart rate achieved: 1 for yes, 0 for no)
""")

st.sidebar.header('User Input Features')



# Collects user input features into dataframe

def user_input_features():
    age = st.sidebar.number_input('Masukan umur: ')

    sex  = st.sidebar.selectbox('Gender',(0,1))
    cp = st.sidebar.selectbox('Chest pain type',(0,1,2,3))
    tres = st.sidebar.number_input('Resting blood pressure: ')
    chol = st.sidebar.number_input('Serum cholestoral in mg/dl: ')
    fbs = st.sidebar.selectbox('Fasting blood sugar',(0,1))
    res = st.sidebar.number_input('Resting electrocardiographic results: ', 0, 1)
    tha = st.sidebar.number_input('Maximum heart rate achieved: ')
    exa = st.sidebar.selectbox('Exercise induced angina: ',(0,1))
    old = st.sidebar.number_input('oldpeak ',)
    slope = st.sidebar.number_input('he slope of the peak exercise ST segmen: ', 0, 2)
    ca = st.sidebar.selectbox('number of major vessels',(0,1,2,3))
    thal = st.sidebar.selectbox('thal',(0,1,2,3))

    data = {'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps':tres,
            'chol': chol,
            'fbs': fbs,
            'restecg': res,
            'thalach':tha,
            'exang':exa,
            'oldpeak':old,
            'slope':slope,
            'ca':ca,
            'thal':thal
                }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire dataset
# This will be useful for the encoding phase
heart_dataset = pd.read_csv('heart.csv')
heart_dataset = heart_dataset.drop(columns=['target'])

df = pd.concat([input_df,heart_dataset],axis=0)

# Encoding of ordinal features
df_encoded = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Select only the first row (the user input data)

input_encoded = df_encoded[:1]
input_encoded.columns = input_encoded.columns.str.split('.').str[0]

# Make predictions using the loaded classification model
prediction = load_clf.predict(input_encoded)
prediction_proba = load_clf.predict_proba(input_encoded)


# Display the prediction and prediction probability
st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
