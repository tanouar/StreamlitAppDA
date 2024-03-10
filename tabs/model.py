import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    
    # st.image("Data/ML.jpg", width=400)
    # st.header("Modélisation")
    st.header("🧩 Modélisation")

    df_ctpzi=pd.read_csv("Data/ctpzi.csv", encoding='latin-1')  
    df=pd.read_csv("Data/merged_owid_temp_zones.csv", index_col=0)
    # On retire tout de suite certaines mesures qui sont directement liées aux autres (donc pas utiles pour notre Machine Learning)
    df = df.drop(["co2_per_capita", "temperature_change_from_ch4","temperature_change_from_co2","temperature_change_from_n2o"], axis= 1)
    df.reset_index(drop=True, inplace=True)
    # ON INTERPOLE LE PIB, LA POPULATION ET LE CO2 QD C'EST POSSIBLE (au sein d'un même pays)
    grouped = df.groupby('iso_code')
    df.gdp = grouped['gdp'].apply(lambda x: x.interpolate())
    df.population = grouped['population'].apply(lambda x: x.interpolate())
    df.co2 = grouped['co2'].apply(lambda x: x.interpolate())
    #  ON RETIRE TOUTES LES LIGNES SANS INFO CO2
    df = df.dropna(subset=['co2'])
    # ON RETIRE TOUTES LES LIGNES SANS GDP (sinon, il faudrait faire des recherches car on ne peut pas "générer" l'info)
    df = df.dropna(subset=['gdp'])
    # ON ELIMINE LES COLONNES MH4 ET N2O afin de garder un nombre de lignes un peu conséquent
    df = df.drop(["methane","nitrous_oxide","iso_code","continent"], axis=1)
    df = df.rename(columns={'gdp' :'pib', 'year':'année', 'temperature_change_from_ghg':'dT°_dû_aux_ghg'})

    st.markdown("""
        ### Prédire l\'augmentation de température par pays par année
        * **Algorithme d'apprentissage supervisé**     
        * **Modèle de machine learning de REGRESSION**
        """)

    if st.checkbox('Préparation des données'):
          if st.button("Pays / Zones / Continents") :
                st.dataframe(df_ctpzi.iloc[:, [0,1,4]])
          if st.button('Nettoyage'):
            st.markdown("""
            - Periode de temps >1950            
            - Colonnes inutiles
            - Interpollation
            - Suppression des colones avec trop de NaN
            """)

          if st.button('Jeu préparé'):
              tab = pd.DataFrame(df.head(15))
              tab['population'] = tab['population'].apply(lambda x: '{:.0f}'.format(x))
              tab['pib'] = tab['pib'].apply(lambda x: '{:.0f}'.format(x))
              tab['dT°_dû_aux_ghg'] = tab['dT°_dû_aux_ghg'].apply(lambda x: '{:.3f}'.format(x))
              tab['co2'] = tab['co2'].apply(lambda x: '{:.3f}'.format(x))
              tab['temperature'] = tab['temperature'].apply(lambda x: '{:.2f}'.format(x))
              
              def style_temp(col):
                  # Condition pour appliquer le style uniquement à la colonne "température"
                  if col.name == 'temperature':
                      return ['background: #b6d7a8'] * len(col)  # Changer la couleur de fond de la colonne température
                  else:
                      return ['']  * len(col) # Aucun style pour les autres colonnes

              # Appliquer le style à la colonne température
              styled_tab = tab.style.apply(style_temp, axis=0)
              st.table(styled_tab)

              st.markdown("""
                - OneHotEncoding pour *zone_géo*          
                - Normalisation pour *année*
                - RobustScaler pour les autres variables
              """)

    if st.checkbox('Machine Learning'):
        st.markdown("""
                    Plusieurs modèles de machine learning ont été testés :
                    - Régression linéaire
                    - Decision Tree Regressor
                    - Random Forest Regressor
                    - Gradient Boosting Regressor
        """)

        if st.button('Choix du modèle'):
          st.markdown("""
                    Plusieurs modèles de machine learning ont été testés :
                    """)
          data = {
          'Modèles': ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor', 'Random Forest Regressor optimisé', 'XGBoost Regressor', 'XGBoost Regressor optimisé'],
          'MSE': [0.21, 0.26, 0.13, 0.14, 0.14, 0.13],
          'RMSE': [0.46, 0.51, 0.37, 0.38, 0.37, 0.36],
          'MAE': [0.35, 0.36, 0.26, 0.28, 0.27, 0.26],
          'R² jeu de test': [0.4, 0.25, 0.62, 0.59, 0.61, 0.64],
          'R² jeu d\'apprentissage': [0.4, 1, 0.95, 0.79, 0.88, 0.94]
          }

          tab2 = pd.DataFrame(data)
          tab2['MSE'] = tab2['MSE'].apply(lambda x: '{:.2f}'.format(x))
          tab2['RMSE'] = tab2['RMSE'].apply(lambda x: '{:.2f}'.format(x))
          tab2['MAE'] = tab2['MAE'].apply(lambda x: '{:.2f}'.format(x))
          tab2['R² jeu de test'] = tab2['R² jeu de test'].apply(lambda x: '{:.2f}'.format(x))
          tab2['R² jeu d\'apprentissage'] = tab2['R² jeu d\'apprentissage'].apply(lambda x: '{:.2f}'.format(x))
  
          def style_temp2(col):
              # Condition pour appliquer le style uniquement à la colonne "R² test"
              if col.name == 'R² jeu de test':
                  return ['background: #b6d7a8'] * len(col)  
              else:
                  return ['']  * len(col) 
          styled_tab2 = tab2.style.apply(style_temp2)
        
          st.table(styled_tab2)
 
        if st.button('Résultats'):
          st.markdown("""
                    Plusieurs modèles de machine learning ont été testés :
                    """)
        
        if st.button('Importance'):
          st.markdown("""
                    Le graphe d'importance :
                    """)




    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

