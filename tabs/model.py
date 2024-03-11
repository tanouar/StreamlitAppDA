import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats as stats
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
# from sklearn.svm import SVC
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import learning_curve
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


title = "Températures terrestres"
sidebar_name = "Modélisation"


def run():
  # st.image("Data/ML.jpg", width=400)
  # st.header("Modélisation")

# LOAD JEU DE DONNEES et TRAITEMENTS (split etc.)
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
  df = df.rename(columns={'gdp' :'pib', 'year':'année', 'temperature_change_from_ghg':'delta_T°_dû_aux_ghg'})

  # séparation des features et de la target
  target = df.temperature
  feats = df.drop("temperature", axis=1)
  # SPLIT du jeu de test et du jeu d'entrainement
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state=42)  # , random_state=42
  # On sépare les données catégorielles et numériques.
  num_cols = ['année','population','pib', 'co2', 'delta_T°_dû_aux_ghg']
  cat_cols = ['zone_geo']
  num_train = X_train[num_cols]
  cat_train = X_train[cat_cols]
  num_test = X_test[num_cols]
  cat_test = X_test[cat_cols]

  # Normalisation des données "années"
  # RobustScaler pour les autres données quantitatives vu qu'on n'a pas de loi normale, et qu'on a bcp d'outliers
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import MinMaxScaler
  from sklearn.preprocessing import RobustScaler

  column_transformer = ColumnTransformer([('min_max_scaler', MinMaxScaler(), ['année']), ('robust_scaler', RobustScaler(), ['population','pib', 'co2', 'delta_T°_dû_aux_ghg'])])

  num_train_scaled = column_transformer.fit_transform(X_train)
  num_test_scaled = column_transformer.transform(X_test)

  # OneHotEncoder pour les zones geo
  # vu qu'on a un nb limité de zones géographiques, on peut se permettre le OneHotEncoder
  from sklearn.preprocessing import OneHotEncoder

  ohe = OneHotEncoder(sparse_output=False)
  cat_train_encoded = ohe.fit_transform(cat_train)
  cat_test_encoded = ohe.transform(cat_test)
  # On regroupe nos jeux de données
  X_train_processed = pd.concat([pd.DataFrame(num_train_scaled, columns=num_cols),
                                pd.DataFrame(cat_train_encoded, columns=ohe.get_feature_names_out(cat_cols))],
                                axis=1)
  X_test_processed = pd.concat([pd.DataFrame(num_test_scaled, columns=num_cols),
                                pd.DataFrame(cat_test_encoded, columns=ohe.get_feature_names_out(cat_cols))],
                              axis=1)
# LOAD MODELE 
    
  def charger_modele(modelPath):
    # Charger le modèle à partir du fichier Pickle
    with open(modelPath, 'rb') as fichier_modele:
        modele = pickle.load(fichier_modele)
    return modele

  # Charger les modèles
  modeleXGB = charger_modele('Data/XGB.pkl')
  modeleRF = charger_modele('Data/RF.pkl')


# FONCTIONS DE VISU POUR LES MODELES
  def residus(y_test, y_pred, titre):
    residuals = y_test - y_pred

    # Créer un subplot 2x2 pour les graphiques de résidus pour ce modèle
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 7))
    fig.subplots_adjust(hspace=0.4, wspace=0.6)
    fig.suptitle(titre, fontsize=16)

    # Graphique de dispersion des résidus avec ligne horizontale à y=0
    sns.scatterplot(x=y_pred, y=residuals, ax=axes[0, 0])
    axes[0, 0].set_title("Graphique de Dispersion des Résidus")
    axes[0, 0].set_xlabel("Prédictions")
    axes[0, 0].set_ylabel("Résidus")
    axes[0, 0].axhline(y=0, color='r', linestyle='--')

    # Histogramme des résidus
    sns.histplot(residuals, ax=axes[0, 1], kde=True)
    axes[0, 1].set_title("Histogramme des Résidus")
    axes[0, 1].set_xlabel("Résidus")

    # Comparaison entre les Valeurs Réelles et Prédites avec ligne diagonale en rouge
    sns.scatterplot(x=y_test, y=y_pred, ax=axes[1, 0])
    axes[1, 0].set_title("Comparaison Valeurs Réelles vs. Prédites")
    axes[1, 0].set_xlabel("Valeurs Réelles")
    axes[1, 0].set_ylabel("Prédictions")
    axes[1, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')

    # QQ plot des résidus
    stats.probplot(residuals, plot=axes[1, 1])
    axes[1, 1].set_title("QQ Plot des Résidus")
    axes[1, 1].set_xlabel("Quantiles théoriques")
    axes[1, 1].set_ylabel("Quantiles empiriques")
    st.pyplot(fig)

  def importances(model,X_train_processed, titre, nb):
    feat_importances = pd.DataFrame(model.feature_importances_, index=X_train_processed.columns, columns=["Importance"])
    feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
    feat_importances = feat_importances.head(nb)
    fig = px.bar(feat_importances, x=feat_importances.index, y='Importance', 
                 title=titre, color=feat_importances.index,
                 color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig)

  # y_predRF = modeleRF.predict(X_test_processed)
  # residus(y_test, y_predRF, 'Résidus pour le RandomForest')



# Interface :
  st.header("🧩 Modélisation")
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
      tab['delta_T°_dû_aux_ghg'] = tab['delta_T°_dû_aux_ghg'].apply(lambda x: '{:.3f}'.format(x))
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
            RandomForest et XGBoost sont les modèles choisis :
            """)
        
      # all_ml_models = ["XGBoost","Random Forest"]
      # model_choice = st.selectbox("Selectionner le modèle à étudier :",all_ml_models)
      # if model_choice == "Random Forest":
      #   st.markdown(""" test """)
      # elif model_choice == "XGBoost":
      y_predXGB = modeleXGB.predict(X_test_processed)
      residus(y_test, y_predXGB, 'Résidus pour le XGBoost')
         
      # y_predRF = modeleRF.predict(X_test_processed)
      # residus(y_test, y_predRF, 'Résidus pour le RandomForest')

    if st.button('Importance'):
      st.markdown("""
            Les graphes d'importances de RandomForest et XGBoost :
            """)
      # nbImp = st.slider('Sélectionnez le nombre de features d\'importance:', 10,len(X_train_processed.columns))
      nbImp=13
      importances(modeleRF,X_train_processed, "Variables les plus importantes pour le modèle RandomForest",nbImp)
      importances(modeleXGB,X_train_processed, "Variables les plus importantes pour le modèle XGBoostRegressor",nbImp)



  st.write("")
  st.write("")
  st.write("")
  st.write("")
  st.write("")
  st.write("")
