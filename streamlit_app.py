import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

df=pd.read_csv("merged_owid_temp.csv", index_col=0)  

st.title("Les températures terrestres")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
    st.write("### Exploration")
    st.dataframe(df.head(10))

    st.write(df.shape)
    st.dataframe(df.describe())

    if st.checkbox("Afficher les NA") :
        st.dataframe(df.isna().sum())

if page == pages[1] : 
    st.write("### DataVizualization")
### Émission mondiale de CO2 ###
world_co2=df.loc[df['year']>=1950] #Pour avoir le minimum de valeurs manquantes
fig = px.choropleth(world_co2, locations='iso_code',color= 'co2',
                    animation_frame= 'year',
                    hover_name= 'country',
                    color_continuous_scale='pubu', # couleurs à tester : amp, Blues, Reds, pubu, bluered, orrd, ylorrd
                    projection ='natural earth',
                    title='Émissions mondiales de CO2',
                    range_color=(0, 12000))
fig. update_layout(geo=dict(showframe=False, showcoastlines=False), title=dict(x=0.5,font=dict(size=20)), height=600, width=800)
fig.update_coloraxes(colorbar=dict(x=1, y=0.5,len=1, tickfont=dict(size=10),title = "CO2 (MT)"))

fig.show()


#     fig = plt.figure()
#     sns.countplot(x = 'Survived', data = df)
#     st.pyplot(fig)
#     fig = plt.figure()
#     sns.countplot(x = 'Sex', data = df)
#     plt.title("Répartition du genre des passagers")
#     st.pyplot(fig)
#     fig = plt.figure()
#     sns.countplot(x = 'Pclass', data = df)
#     plt.title("Répartition des classes des passagers")
#     st.pyplot(fig)
#     fig = sns.displot(x = 'Age', data = df)
#     plt.title("Distribution de l'âge des passagers")
#     st.pyplot(fig)

#     fig = plt.figure()
#     sns.countplot(x = 'Survived', hue='Sex', data = df)
#     st.pyplot(fig)
#     fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
#     st.pyplot(fig)
#     fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
#     st.pyplot(fig)

if page == pages[2] : 
    st.write("### Modélisation")
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



