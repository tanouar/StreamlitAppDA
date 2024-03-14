import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


title = "Températures terrestres"
sidebar_name = "Modélisation"


def run():
    
    st.image("Data/ML.jpg", width=400)
    st.header("Modélisation")
  
  # df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# y = df['Survived']
# X_cat = df[['Pclass', 'Sex',  'Embarked']]
# X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

# for col in X_cat.columns:
#     X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
# for col in X_num.columns:
#     X_num[col] = X_num[col].fillna(X_num[col].median())
# X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
# X = pd.concat([X_cat_scaled, X_num], axis = 1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# scaler = StandardScaler()
# X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
# X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

# def prediction(classifier):
#     if classifier == 'Random Forest':
#         clf = RandomForestClassifier()
#     elif classifier == 'SVC':
#         clf = SVC()
#     elif classifier == 'Logistic Regression':
#         clf = LogisticRegression()
#     clf.fit(X_train, y_train)
#     return clf

# def scores(clf, choice):
#     if choice == 'Accuracy':
#         return clf.score(X_test, y_test)
#     elif choice == 'Confusion matrix':
#         return confusion_matrix(y_test, clf.predict(X_test))
        
# choix = ['Random Forest', 'SVC', 'Logistic Regression']
# option = st.selectbox('Choix du modèle', choix)
# st.write('Le modèle choisi est :', option)

# clf = prediction(option)
# display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
# if display == 'Accuracy':
#     st.write(scores(clf, display))
# elif display == 'Confusion matrix':
#     st.dataframe(scores(clf, display))

