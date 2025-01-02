import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats as stats
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
import plotly.colors

title = "Températures terrestres"
sidebar_name = "Modélisation"
#  Chapitre MODELISATION où on montre les modèles qu'on a testés. 
# Premiere partie sur le nettoyage des données pour pouvoir faire apprendre nos modèles ensuite.
# Deuxieme partie pour montrer les modeles
# Troisieme partie pour simuiler une prédiction

# Le début du run continet tous les calculs et les imports de data et de modèles. 
# La suite est la gestion de la visualisation en Streamllit

def run():
# LOAD JEU DE DONNEES et TRAITEMENTS (split etc.)
  #  Jeu avec les zones geographiques
  df_ctpzi=pd.read_csv("Data/ctpzi.csv", encoding='latin-1')
  df_ctpzi.drop(columns=['Num-3', 'Alpha-2','Notes de bas de page'], inplace=True)
  df_ctpzi.Code = df_ctpzi.Code // 1000
  df_ctpzi["continent"] = df_ctpzi.Code // 10
  df_ctpzi = df_ctpzi.dropna()
  df_ctpzi.rename(columns={'Alpha-3' : 'iso_code', 'Code':'zone_geo', 'Pays et zones d\'intérêt' : 'pays'}, inplace=True)

  # Jeu avec toutes les donnees pays de Co2, températures etc.
  df_init=pd.read_csv("Data/merged_owid_temp_zones.csv", index_col=0)
  # On retire tout de suite certaines mesures qui sont directement liées aux autres (donc pas utiles pour notre Machine Learning)
  df = df_init.drop(["co2_per_capita", "temperature_change_from_ch4","temperature_change_from_co2","temperature_change_from_n2o"], axis= 1)
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
  # qq valeurs qui serviront pour la prédiction
  moy_dt=np.mean(df.temperature_change_from_ghg)
  min_co2= min(df.co2)
  max_co2= max(df.co2)
  moy_co2= np.mean(df.co2)
  min_pib= min(df.gdp)
  max_pib= max(df.gdp)
  moy_pib= np.mean(df.gdp)
  min_pop= min(df.population)
  max_pop= max(df.population)
  moy_pop= np.mean(df.population)
  
  df = df.rename(columns={'gdp' :'pib', 'year':'année', 'temperature_change_from_ghg':'delta T°_dû_aux_ghg'})
  
  # séparation des features et de la target
  target = df.temperature
  feats = df.drop("temperature", axis=1)
  # SPLIT du jeu de test et du jeu d'entrainement
  X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state=42)  # , random_state=42
  # On sépare les données catégorielles et numériques.
  num_cols = ['année','population','pib', 'co2', 'delta T°_dû_aux_ghg']
  cat_cols = ['zone_geo']
  num_train = X_train[num_cols]
  cat_train = X_train[cat_cols]
  num_test = X_test[num_cols]
  cat_test = X_test[cat_cols]

  # Normalisation des données "années"
  # RobustScaler pour les autres données quantitatives vu qu'on n'a pas de loi normale, et qu'on a bcp d'outliers
  column_transformer = ColumnTransformer([('min_max_scaler', MinMaxScaler(), ['année']), ('robust_scaler', RobustScaler(), ['population','pib', 'co2', 'delta T°_dû_aux_ghg'])])
  num_train_scaled = column_transformer.fit_transform(X_train)
  num_test_scaled = column_transformer.transform(X_test)
  # OneHotEncoder pour les zones geo
  # vu qu'on a un nb limité de zones géographiques, on peut se permettre le OneHotEncoder

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
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.6)
    fig.suptitle(titre, fontsize=12)

    # Graphique de dispersion des résidus avec ligne horizontale à y=0
    sns.scatterplot(x=y_pred, y=residuals, ax=axes[0, 0])
    axes[0, 0].set_title("Graphique de Dispersion des Résidus", fontsize=11)
    axes[0, 0].set_xlabel("Prédictions", fontsize=10)
    axes[0, 0].set_ylabel("Résidus", fontsize=10)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')

    # Histogramme des résidus
    sns.histplot(residuals, ax=axes[0, 1], kde=True)
    axes[0, 1].set_title("Histogramme des Résidus", fontsize=11)
    axes[0, 1].set_xlabel("Résidus", fontsize=10)

    # Comparaison entre les Valeurs Réelles et Prédites avec ligne diagonale en rouge
    sns.scatterplot(x=y_test, y=y_pred, ax=axes[1, 0])
    axes[1, 0].set_title("Comparaison Valeurs Réelles vs. Prédites", fontsize=11)
    axes[1, 0].set_xlabel("Valeurs Réelles", fontsize=10)
    axes[1, 0].set_ylabel("Prédictions", fontsize=10)
    axes[1, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')

    # QQ plot des résidus
    stats.probplot(residuals, plot=axes[1, 1])
    axes[1, 1].set_title("QQ Plot des Résidus", fontsize=11)
    axes[1, 1].set_xlabel("Quantiles théoriques", fontsize=10)
    axes[1, 1].set_ylabel("Quantiles empiriques", fontsize=10)
    st.pyplot(fig)

  def importances(model,X_train_processed, titre, nb):
    feat_importances = pd.DataFrame(model.feature_importances_, index=X_train_processed.columns, columns=["Importance"])
    feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
    feat_importances = feat_importances.head(nb)
    fig = px.bar(feat_importances, x=feat_importances.index, y='Importance', 
                 title=titre, color=feat_importances.index,
                 color_discrete_sequence=px.colors.qualitative.Plotly,
                 width=650, height=350)
    st.plotly_chart(fig)

# Courbes d'apprentissage
  def plot_learning_curve(model, X, y, titre):
      train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 5))
      # Calculate mean and standard deviation for training set scores
      train_mean = np.mean(train_scores, axis=1)
      train_std = np.std(train_scores, axis=1)
      # Calculate mean and standard deviation for test set scores
      test_mean = np.mean(test_scores, axis=1)
      test_std = np.std(test_scores, axis=1)
      # Plot the learning curve
      plt.figure(figsize=(6, 4))
      plt.plot(train_sizes, train_mean, label="Score d'apprentissage", color='blue')
      plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.1)
      plt.plot(train_sizes, test_mean, label='Score de Cross-Validation', color='orange')
      plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='orange', alpha=0.1)
      # Add labels and title
      plt.xlabel("Taille du jeu d'apprentissage")
      plt.ylabel("Erreur quadratique")
      plt.title(titre)
      plt.legend()
      st.pyplot(plt.gcf())

############################################################################################################ 
###################################### Interface : #########################################################
############################################################################################################ 
  st.write("  ")
  st.header("Modélisation")
  st.markdown("---")
  st.markdown("""
        ### Prédire l\'augmentation de température par pays par année
        ➽ **Utiliser des algorithmes d'apprentissage supervisé**     
        ➽ **Tester puis choisir un modèle de *Machine Learning* de REGRESSION**
        """)

###################### Préparation des données
  if st.checkbox('Préparation des données'):
    with st.expander("**Pays / Zones / Continents**"):
      col1, col2 = st.columns(2)
      with col1:
         st.markdown("Comment gérer la **donnée catégorielle** *Pays* ?\n\n ➽ Nécessité de regrouper les pays en utilisant les données présentes de codification ISO-3166-1.\n\n➽ Utilisation de la \"Classification type des pays et des zones d'intérêt\" du Canada.")
      with col2:
         st.dataframe(df_ctpzi)

      df_ctpzi['zone_geo'] = df_ctpzi['zone_geo'].astype(str)
      zones = {"11.0": "Amérique du Nord" ,
        "12.0":"Amérique Centrale" ,
        "13.0": "Caraïbes" ,
        "14.0": "Amérique du Sud" ,
        "21.0": "Europe de l'Ouest" ,
        "22.0" : "Europe de l'Est" ,
        "23.0" : "Europe du Nord" ,
        "24.0" : "Europe du Sud" ,
        "31.0" :"Afrique de l'Ouest" ,
        "32.0": "Afrique de l'Est" ,
        "33.0": "Afrique du Nord" ,
        "34.0" : "Afrique Centrale  " ,
        "35.0": "Afrique Australe" ,
        "41.0" : "Asie Centrale" ,
        "42.0" : "Asie Orientale" ,
        "43.0" : "Asie du Sud-Est" ,
        "44.0": "Asie Méridionale" ,
        "51.0": "Océanie",
        "61.0": "Antarctique"}

      df_ctpzi['zone_geo'].replace(zones, inplace=True)
      fig = px.choropleth(locations=df_ctpzi['iso_code'],
                          color=df_ctpzi['zone_geo'],
                          hover_name=df_ctpzi['pays'],
                          projection="natural earth",
                          color_discrete_map={zone: color for zone, color in zip(df_ctpzi['zone_geo'].unique(), plotly.colors.qualitative.Light24)},
                          height=500, width=650, labels={'color':'zone_geo'}
                          )
      fig.update_layout(geo=dict(showframe=False))
      st.plotly_chart(fig)

    with st.expander("**Nettoyage**"):
      if st.checkbox("Afficher un extrait du dataset avec les zones géographiques ?") :
         st.dataframe(df_init.head())

      if st.checkbox("Afficher les statistiques de ce dataset ?") :
          st.dataframe(df_init.describe())

      st.markdown("""Nombre des données manquantes :""")
      col1, col2 = st.columns(2)
 
      with col1:
        st.dataframe(df_init.isna().sum(), width=500)
 
      with col2:
        st.markdown("""
        Actions pour nettoyer notre table :             
        - Réduire la période de temps >1970            
        - Supprimer les colonnes inutiles
        - Interpoler les données au maximum
        - Supprimer les colonnes avec trop de *NaN*
        """)

    with st.expander("**Jeu préparé**"):
      st.write("On obtient ainsi ce jeu de données :")
      tab = pd.DataFrame(df.head(15))
      tab['population'] = tab['population'].apply(lambda x: '{:.0f}'.format(x))
      tab['pib'] = tab['pib'].apply(lambda x: '{:.0f}'.format(x))
      tab['delta T°_dû_aux_ghg'] = tab['delta T°_dû_aux_ghg'].apply(lambda x: '{:.3f}'.format(x))
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
      # Afficher le dataframe du jeu de données
      st.table(styled_tab)
      st.write("Taille du tableau :", df.shape)
      st.write("6 variables *features* et une *target*")
      st.markdown("""
        Encodage des variables :                  
        - OneHotEncoding pour *zone_géo*          
        - Normalisation pour *année*
        - RobustScaler pour les autres variables
      """)

###################### Machine Learning
  if st.checkbox('Machine Learning'):
    st.markdown("""
          Plusieurs modèles de *Machine Learning* ont été testés :
          - *Linear Regression*
          - *Decision Tree Regressor*
          - *Random Forest Regressor*
          - *Gradient Boosting Regressor*
        """)

    with st.expander("**Choix du modèle**"):
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
      st.markdown("""
            ➽ Les modèles RandomForestRegressor et XGBoostRegressor ont des résultats de test similaires.
            """)
      st.write("")
      st.write("")

    with st.expander("**Courbes de résidus**"):
      st.markdown("""
            ➽ Regardons ici les figures pour le RandomForestRegressor et le XGBoostRegressor :
            """)
     
      all_ml_models = ["RandomForestRegressor","XGBoostRegressor"]
      model_choice = st.selectbox("Selectionner le modèle à étudier :",all_ml_models)
      if model_choice == "RandomForestRegressor":     
        st.image("Data/ML.jpg", width=700)
        # y_predRF = modeleRF.predict(X_test_processed)
        # residus(y_test, y_predRF, 'Résidus pour le RandomForest')
      elif model_choice == "XGBoostRegressor":
        st.image("Data/fleche.jpg", width=700)        
        # st.image("Data/ResidusXGB.jpg", width=700)
        # y_predXGB = modeleXGB.predict(X_test_processed)
        # residus(y_test, y_predXGB, 'Résidus pour le XGBoost')
      st.write("➽ Erreurs homogènes et distribution normale")

    with st.expander("**Courbes d'apprentissage**"):
      col1, col2 = st.columns(2)
 
      with col1:
          #  utiliser plt.savefig(save_path)
        st.image("Data/LCurveRF.jpg", width=380)
 
      with col2:  
        st.image("Data/LCurveXGB.jpg", width=380)
      
      # plot_learning_curve(modeleXGB, X_train_processed, y_train, "Courbe d'apprentissage pour le XGBoost")
      st.write("➽ Le modèle pourrait être amélioré avec un plus grand jeu d'apprentissage")
      st.write("")
      
      
    with st.expander("**Visualisation des variables d'importance**"):
      st.markdown("""
            Les graphes d'importances de RandomForest et XGBoost illustrent la différence de comportement des 2 modèles :
            """)
      nbImp=13
      importances(modeleRF,X_train_processed, "Variables les plus importantes pour le modèle RandomForestRegressor",nbImp)
      importances(modeleXGB,X_train_processed, "Variables les plus importantes pour le modèle XGBoostRegressor",nbImp)

      st.write("➽ La relation entre la température de la terre et les variables étudiées est complexe et peut être influencée par de nombreux autres facteurs externes.")

###################### Prédictions 
  if st.checkbox('Prédictions avec XGBoostRegressor'):
    # récupérer les variables pour la prédiction et les standardiser 
    # Afficher un menu déroulant pour sélectionner la zone géographique
    def get_code(region):
      codes = {
              "11 - Amérique du Nord  ":11,
              "12 - Amérique Centrale ":12,
              "13 - Caraïbes          ":13,
              "14 - Amérique du Sud   ":14,
              "21 - Europe de l'Ouest ":21,
              "22 - Europe de l'Est   ":22,
              "23 - Europe du Nord    ":23,
              "24 - Europe du Sud     ":24,
              "31 - Afrique de l'Ouest":31,
              "32 - Afrique de l'Est  ":32,
              "33 - Afrique du Nord   ":33,
              "34 - Afrique Centrale  ":34,
              "35 - Afrique Australe  ":35,
              "41 - Asie Centrale     ":41,
              "42 - Asie Orientale    ":42,
              "43 - Asie du Sud-Est   ":43,
              "44 - Asie Méridionale  ":44,
              "51 - Océanie           ":51,
      }
      return codes.get(region)

    selected_region = st.selectbox("Choisissez une région :", [
        "11 - Amérique du Nord  " ,
        "12 - Amérique Centrale " ,
        "13 - Caraïbes          " ,
        "14 - Amérique du Sud   " ,
        "21 - Europe de l'Ouest " ,
        "22 - Europe de l'Est   " ,
        "23 - Europe du Nord    " ,
        "24 - Europe du Sud     " ,
        "31 - Afrique de l'Ouest" ,
        "32 - Afrique de l'Est  " ,
        "33 - Afrique du Nord   " ,
        "34 - Afrique Centrale  " ,
        "35 - Afrique Australe  " ,
        "41 - Asie Centrale     " ,
        "42 - Asie Orientale    " ,
        "43 - Asie du Sud-Est   " ,
        "44 - Asie Méridionale  " ,
        "51 - Océanie           " 
        ])

    code = get_code(selected_region)
    df_code = pd.DataFrame({"zone_geo": [code]})

    cat_test_encoded = ohe.transform(df_code)    
  
    caracteristique1 = int(st.slider("Année ", 1970, 2017, 2000))
    caracteristique2 = int(st.slider("PIB ", int(min_pib), int(max_pib), int(moy_pib)))
    caracteristique3 = int(st.slider("Population ", int(min_pop), int(max_pop), int(moy_pop)))
    caracteristique4 = int(st.slider("Emissions CO2 (MT)", int(min_co2), int(max_co2), int(moy_co2)))
    caracteristique5 = moy_dt
    caracteristiques = np.array([[caracteristique1, caracteristique2, caracteristique3, caracteristique4,caracteristique5]])

    df_numerical = pd.DataFrame({
        "année": [caracteristique1],
        "population": [caracteristique3],
        "pib": [caracteristique2],
        "co2": [caracteristique4],
        "delta T°_dû_aux_ghg": [caracteristique5]
    })
    num_test_scaled = column_transformer.transform(df_numerical)

    # Concaténation des variables standardisees
    X_predict_processed = pd.concat([
        pd.DataFrame(num_test_scaled, columns=['année', 'population', 'pib', 'co2', 'delta_T°_dû_aux_ghg']),
        pd.DataFrame(cat_test_encoded, columns=ohe.get_feature_names_out(['zone_geo']))
    ], axis=1)

    # Calculer et afficher la prédiction
    prediction = modeleXGB.predict(X_predict_processed)
    prediction = prediction[0]
    prediction =  "{:.2f}".format(prediction)
    st.write(f"**L'évolution de température prédite est de : {prediction} °C**")
    
  st.write("")
  st.write("")
  st.write("")
  st.write("")
  st.write("")

