import streamlit as st
import utils.str_func as common

common.page_config("Données")
common.menu()
common.local_css("css_str.css")

st.html("<h1>Conclusion</h1>")
st.write("#### Difficultés rencontrées : ")

st.write("Lors de la conception de notre projet, nous avons été confrontées à plusieurs défis : 🧗‍♂️")
st.write("""
         - Un **ensemble de données extrêmement volumineux**, entrainant des difficultés de fluidité dans la réalisation des tests et des modèles de machine learning.
         - Une exploration additionnelle des **modèles Ridge et Lasso** que nous n'avions pas étudié lors de notre formation.
         - Une **recherche d'optimisation des hyperparamètres compliquée** lors de l'utilisation de "GridSearch" pour obtenir des performances optimales.
         - Une tentative d'élaboration de **modèle de prédiction sans machine learning** chronophage et peu performante.
         """)
st.write("  ")
st.write("#### Bilan :")

st.write("**Les objectifs du projet étaient d'analyser l'évolution du trafic cycliste à Paris et de projeter, pour une date donnée, les horaires et les zones d'affluence du trafic. Ces objectifs ont été atteints avec succès.** :white_check_mark:")

st.write("Nous avons utilisé des **modèles de régression**, comprenant que la variable cible (nombre de comptages)  prend des valeurs continues pouvant varier indéfiniment. Nous avons testé différents algorythmes et analysé **le coefficient de détermination (le score)** pour sélectionner notre modèle. L'objectif était de maximiser ce score global tout en garantissant que les scores sur les ensembles d'entraînement et de test restent proches pour éviter le surajustement (overfitting).")

st.write("Nous avons également comparé plusieurs métriques de performance pour chacun des modèles : le Mean Absolute Error (MAE), le Mean Squared Error (MSE), le Root Mean Squared Error (RMSE), ainsi que la distance euclidienne et le score de silhouette pour le modèle KNN.")
st.write("  ")
st.write("**Résultats :**")
st.write(""" 
         - Le Decision Tree a montré de bons résultats mais tendait à sous-évaluer les comptages extrêmes. 
         - Le modèle KNN détient un “silhouette score” peu élevé. 
         - La Régression Linéaire obtient aussi des scores de performance très bas.
         - Les modèles Ridge et Lasso ont des scores de performance moyens.
         - Le modèle de prédiction sans machine learning manque également de précision.
        """)
st.write("Bien que le Random Forest ait montré de très bons résultats après l'ajustement des paramètres, **le Gradient Boost a été choisi pour sa performance globalement supérieure après réduction des données et utilisation de GridSearch**.")

st.write("  ")
st.write("#### Suite :")

st.write("Nous pourrions envisager l’utilisation de **l’outil “Prophet”** afin d’effectuer des prévisions dans le futur, anticiper les évolutions en 2024. ⏳ ")

st.write("  ")

st.write("##### Equipe :")
st.markdown("""
            - Eléonore Hermand 
            - Myriam Mahdjoub  [Linkedin](https://www.linkedin.com/in/myriam-mahdjoub-419405a8/) 
            - Melissa Chemmama  [Linkedin](https://www.linkedin.com/in/melissa-aidan-8b7161150/) 
            - Lena Guilloux  [Linkedin](https://www.linkedin.com/in/lena-guilloux-4b44a1173/)
            """)

st.write("##### Mentor :")
st.write("Tarik Anouar, DataScientest")

