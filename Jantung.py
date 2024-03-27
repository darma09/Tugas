import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

filePath = 'https://raw.githubusercontent.com/darma09/Tugas/main/heart.csv'
data = pd.read_csv(filePath)
st.title("Heart Disease Classification")

data.head(5)

print("(Rows, columns): " + str(data.shape))
data.columns

data.nunique(axis=0)

data.describe()

print(data.isna().sum())

data['target'].value_counts()

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

pos_data = data[data['target']==1]
pos_data.describe()

neg_data = data[data['target']==0]
neg_data.describe()

print("(Positive Patients ST depression): " + str(pos_data['oldpeak'].mean()))
print("(Negative Patients ST depression): " + str(neg_data['oldpeak'].mean()))

print("(Positive Patients thalach): " + str(pos_data['thalach'].mean()))
print("(Negative Patients thalach): " + str(neg_data['thalach'].mean()))

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


model6 = RandomForestClassifier(random_state=1)
model6.fit(x_train, y_train)

y_pred6 = model6.predict(x_test)
print(classification_report(y_test, y_pred6))

importance = model6.feature_importances_

for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

index= data.columns[:-1]
importance = pd.Series(model6.feature_importances_, index=index)
importance.nlargest(13).plot(kind='barh', colormap='winter')

y_pred = model6.predict(x_test)

hasil_gabungan = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1)

print(hasil_gabungan)



if __name__ == '__main__':
    st.title('Heart Disease Classification')
    st.write('This app predicts the presence of heart disease based on various features.')
    # Define the user input form
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, step=1)
        sex = st.selectbox("Sex", ("Male", "Female"))
        chest_pain = st.selectbox("Chest Pain", ("Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"))
        resting_blood_pressure = st.number_input("Resting Blood Pressure", min_value=0, max_value=300, step=1)
        serum_cholesterol = st.number_input("Serum Cholesterol", min_value=0, max_value=600, step=1)
        fasting_blood_sugar = st.selectbox("Fasting Blood Sugar", ("Lower than 120mg/dl", "Greater than 120mg/dl"))
        resting_electrocardiographic = st.selectbox("Resting Electrocardiographic", ("Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"))
        maximum_heart_rate_achieved = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250, step=1)
        exercise_induced_angina = st.selectbox("Exercise Induced Angina", ("No", "Yes"))
        st_depression = st.number_input("ST Depression", min_value=0, max_value=10, step=0.1)
        slope_of_the_peak_exercise_st_segment = st.selectbox("Slope of the Peak Exercise ST Segment", ("Upsloping", "Flat", "Downsloping"))
        number_of_major_vessels = st.number_input("Number of Major Vessels", min_value=0, max_value=4, step=1)
        thalassemia = st.selectbox("Thalassemia", ("Normal", "Fixed Defect", "Reversible Defect"))


    # Define the prediction button
    if st.button("Predict"):
        user_input = np.array([age, int(sex), int(chest_pain), resting_blood_pressure, serum_cholesterol, int(fasting_blood_sugar), int(resting_electrocardiographic), maximum_heart_rate_achieved, int(exercise_induced_angina), st_depression, int(slope_of_the_peak_exercise_st_segment), int(number_of_major_vessels), int(thalassemia)])
        user_input = sc.transform([user_input])
        prediction = model6.predict(user_input)[0]
        if prediction == 1:
            st.write("The result indicates a risk of heart disease.")
        else:
            st.write("The result suggests no significant risk of heart disease.")
