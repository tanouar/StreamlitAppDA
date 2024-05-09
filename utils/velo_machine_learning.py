from datetime import datetime
import pandas as pd
import json
import streamlit as st
import utils.velo_load_datas as datas
import pickle
from plotly import graph_objs as go
import plotly.express as px
from sklearn.metrics import r2_score

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# prédictions
@st.cache_resource
def load_model():
   gradient_model = pickle.load(open("data/gradient_model", 'rb'))
   return gradient_model

@st.cache_data
def pre_process(df_velo):
    df_velo_ml = datas.get_df_2023 (df_velo)
    df_velo_ml = df_velo_ml.drop(["Date d'installation du site de comptage", "Identifiant technique compteur", "mois_annee_comptage","lat", "lon", "Identifiant du site de comptage", "Nom du site de comptage", "annee", "jour_de_semaine", "station_meteo"], axis = 1)
    df_velo_ml = df_velo_ml.rename(columns={"Identifiant du compteur": "id_compteur","Week-end" : "week_end"," T" : "t","Numéro de semaine de l'année" : 'num_semaine',})
    df_velo_ml['pluvieux'] = df_velo_ml['pluvieux'].replace(['Pas de pluie', 'Pluie modérée', 'Pluie intense'], [0, 1, 2]).astype(int)
    return df_velo_ml

def filtered_datas(compteurs, date, heure, df_velo_ml):
   if heure == None:
      df_velo_ml_filtered = df_velo_ml[(df_velo_ml['Nom du compteur'].isin(compteurs)) & (pd.to_datetime(df_velo_ml["Date et heure de comptage"]).dt.date==date)]
   else:
    date_heure = datetime.combine(date, heure)
    df_velo_ml_filtered = df_velo_ml[(df_velo_ml['Nom du compteur'].isin(compteurs)) & (df_velo_ml["Date et heure de comptage"]==date_heure.strftime('%Y-%m-%d %H:%M:%S'))]
   return df_velo_ml_filtered

@st.cache_data
def factorize_id(df_ml):
    with open("data/id_compteurs.txt") as file:
        ids_dict = json.load(file)
    df_ml['id_compteur'] = df_ml['id_compteur'].replace(ids_dict)
    return df_ml

def predict_velo(df_velo_ml_filtered, gradient_model):
   df_ml = df_velo_ml_filtered[['id_compteur', 'heure', 'week_end', 't', "num_semaine"]].copy()
   df_ml_fact = factorize_id(df_ml)
   df_velo_ml_filtered["Prédiction"] = gradient_model.predict(df_ml_fact)
   r2 = r2_score(df_velo_ml_filtered["Comptage horaire"], df_velo_ml_filtered['Prédiction'])
   return df_velo_ml_filtered[["Nom du compteur", "Date et heure de comptage", "Comptage horaire", 'Prédiction']], r2

def preds_viz_jour(df_velo):
    if(len(df_velo["Nom du compteur"].unique())>1):
       df_velo = df_velo.groupby(['Date et heure de comptage'])[["Comptage horaire", "Prédiction"]].sum().reset_index()
    df_velo = df_velo.sort_values(by="Date et heure de comptage")
    fig = go.Figure()
    heures = pd.to_datetime(df_velo["Date et heure de comptage"]).dt.hour
    heures = [str(h)+'h' for h in heures]
    fig.add_trace(go.Scatter(x=heures, y=df_velo["Comptage horaire"], mode='markers', marker_color="#3D85C6", name="valeurs réelles"))
    fig.add_trace(go.Scatter(x=heures, y=df_velo["Prédiction"], mode='markers', marker_color="#DE8326", name="valeurs prédites"))
    fig.update_layout(legend=dict(orientation="h",yanchor="bottom", y=1.02,xanchor="right",x=1), xaxis = dict(tickmode = 'array', tickvals = list(range(0, 24)),ticktext = heures))
    return fig

def preds_viz_heure(df_velo):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_velo["Nom du compteur"], y=df_velo["Comptage horaire"], marker=dict(color="#3D85C6"),name="valeurs réelles"))
    fig.add_trace(go.Bar(x=df_velo["Nom du compteur"], y=df_velo["Prédiction"],  marker=dict(color="#DE8326"), name="valeurs prédites"))
    return fig



# MODELES


# DECISIONTREE

@st.cache_data
def decision_features():
    feats = pd.read_csv('data/modeles/decision_feats.csv')
    fig = go.Figure([go.Bar(x=feats["Caractéristiques"], y=feats["Importance"])])
    fig.update_traces(marker_color='#0b5394')
    fig.update_layout(title_text='Features importantes')
    return fig

@st.cache_data
def decision_preds():
    preds = pd.read_csv('data/modeles/decision_preds.csv')
    return preds

@st.cache_data
def decision_graph():
    df_decisions = decision_preds()
    fig = px.scatter(x=df_decisions['Données Réelles'], y=df_decisions['Données Prédites'], trendline="ols", trendline_color_override="#6fa8dc")
    fig.update_traces(marker=dict(color='#0b5394'))
    fig.update_layout(hovermode=False, title="Prédictions vs valeurs réelles", xaxis_title="Valeurs réelles", yaxis_title="Valeurs prédites", yaxis_range=[df_decisions['Données Prédites'].min(),1800])
    return fig



#RIDGE

@st.cache_data
def ridge_load_preds():
    ridge_preds = pd.read_csv('data/modeles/ridge_preds.csv')
    return ridge_preds
    
@st.cache_data
def ridge_graph():
    ridge_preds = ridge_load_preds()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    sns.scatterplot(x=ridge_preds['Pred'], y=ridge_preds['Residuals'], ax=axes[0, 0])
    axes[0, 0].set_title("Graphique de Dispersion des Résidus")
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    sns.histplot(x=ridge_preds['Residuals'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Histogramme des Résidus")
    sns.scatterplot(x=ridge_preds['Test'], y=ridge_preds['Pred'], ax=axes[1, 0])
    axes[1, 0].set_title("Comparaison Valeurs Réelles vs. Prédites")
    axes[1, 0].plot([min(ridge_preds['Test']), max(ridge_preds['Test'])], [min(ridge_preds['Test']), max(ridge_preds['Test'])], linestyle='--', color='red')
    stats.probplot(ridge_preds['Residuals'], plot=axes[1, 1])
    axes[1, 1].set_title("QQ Plot des Résidus")
    plt.tight_layout()
    return fig


#LASSO

@st.cache_data
def lasso_load_preds():
    lasso_preds = pd.read_csv('data/modeles/lasso_preds.csv')
    return lasso_preds
    
@st.cache_data
def lasso_graph():
    lasso_preds = lasso_load_preds()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    sns.scatterplot(x=lasso_preds['Pred'], y=lasso_preds['Residuals'], ax=axes[0, 0])
    axes[0, 0].set_title("Graphique de Dispersion des Résidus")
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    sns.histplot(x=lasso_preds['Residuals'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Histogramme des Résidus")
    sns.scatterplot(x=lasso_preds['Test'], y=lasso_preds['Pred'], ax=axes[1, 0])
    axes[1, 0].set_title("Comparaison Valeurs Réelles vs. Prédites")
    axes[1, 0].plot([min(lasso_preds['Test']), max(lasso_preds['Test'])], [min(lasso_preds['Test']), max(lasso_preds['Test'])], linestyle='--', color='red')
    stats.probplot(lasso_preds['Residuals'], plot=axes[1, 1])
    axes[1, 1].set_title("QQ Plot des Résidus")
    plt.tight_layout()
    return fig



# GradientBoost

@st.cache_data
def gradient_features():
    feats = pd.read_csv('data/modeles/gradient_feats.csv')
    fig = go.Figure([go.Bar(x=feats["Caractéristiques"], y=feats["Importance"])])
    fig.update_traces(marker_color='#0b5394')
    fig.update_layout(title_text='Features importantes')
    return fig

@st.cache_data
def gradient_preds():
    preds = pd.read_csv('data/modeles/gradient_preds.csv')
    return preds


@st.cache_data
def gradient_graph():
    df_gradient = gradient_preds()
    fig = px.scatter(x=df_gradient['Données Réelles'], y=df_gradient['Données Prédites'], trendline="ols", trendline_color_override="#6fa8dc")
    fig.update_traces(marker=dict(color='#0b5394'))
    fig.update_layout(hovermode=False, title="Prédictions vs valeurs réelles", xaxis_title="Valeurs réelles", yaxis_title="Valeurs prédites", yaxis_range=[df_gradient['Données Prédites'].min(),1800])
    return fig


#LINEAR REGRESSOR

@st.cache_data
def linear_preds():
    preds = pd.read_csv('data/modeles/linear_preds.csv')
    return preds

@st.cache_data
def linear_graph():
    df_linear = linear_preds()
    fig = px.scatter(x=df_linear['Valeurs Réelles'], y=df_linear['Prédictions'], trendline="ols", trendline_color_override="#6fa8dc")
    fig.update_traces(marker=dict(color='#0b5394'))
    fig.update_layout(hovermode=False, title="Prédictions vs valeurs réelles", xaxis_title="Valeurs réelles", yaxis_title="Valeurs prédites", yaxis_range=[df_linear['Prédictions'].min(),1200])
    return fig


#RANDOM FOREST REGRESSOR

@st.cache_data
def random_features():
    feats = pd.read_csv('data/modeles/random_feats.csv')
    fig = go.Figure([go.Bar(x=feats["Caractéristiques"], y=feats["Importance"])])
    fig.update_traces(marker_color='#0b5394')
    fig.update_layout(title_text='Features importantes')
    return fig

@st.cache_data
def random_preds():
    preds = pd.read_csv('data/modeles/random_preds.csv')
    return preds

@st.cache_data
def random_graph():
    df_random = random_preds()
    fig = px.scatter(x=df_random['Valeurs Réelles'], y=df_random['Prédictions'], trendline="ols", trendline_color_override="#6fa8dc")
    fig.update_traces(marker=dict(color='#0b5394'))
    fig.update_layout(hovermode=False, title="Prédictions vs valeurs réelles", xaxis_title="Valeurs réelles", yaxis_title="Valeurs prédites", yaxis_range=[df_random['Prédictions'].min(),1800])
    return fig

def random_max_depth():
    results_df = pd.read_csv('data/modeles/random_max_depth.csv')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results_df['max_depth'], y=results_df['train_score'], mode='lines+markers', marker_color="#3D85C6", name='score train'))
    fig.add_trace(go.Scatter(x=results_df['max_depth'], y=results_df['test_score'], mode='lines+markers', marker_color="#DE8326", name='score test'))
    fig.update_layout(title="Scores selon la max_depth", xaxis_title="max_depth", yaxis_title="Scores", legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))
    return fig


#KNN

def KNN_coude():
    with open('data/modeles/KNN_coude.json', 'rb') as fp:
        L = json.load(fp)

    fig = go.Figure(data=go.Scatter(x=[1,2,3,4,5,6,7,8,9], y=L))
    fig.add_annotation(text='n_clusters=4', x=4, y=100000, arrowhead=2, arrowwidth=2)
    fig.update_layout(title="Méthode du coude")
    return fig

def KNN_centroids(df_velo):
    df_velo_2023 = pre_process(df_velo)
    df_ml = factorize_id(df_velo_2023)
    moyennes=df_ml.groupby("id_compteur")["Comptage horaire"].mean()
    with open('data/modeles/KNN_centroids.txt', 'rb') as fp:
        centroids = json.load(fp)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=moyennes.index, y=moyennes,mode='markers', name="Comptage"))
    fig.add_trace(go.Scatter(x=list(centroids.keys()), y=list(centroids.values()), mode='markers', marker_color="red", marker_size=11, name="Centroids"))
    fig.update_layout(title="Centroids")
    return fig


#Méthode de prédiction sans machine learning

def msml_correlation():
    df_correlation = pd.read_csv('data/modeles/msml_correlation_matrix.csv')
    df_correlation['Corrélation'] = pd.to_numeric(df_correlation['Corrélation'])
    heatmap_data = df_correlation.pivot(index='Variable 1', columns='Variable 2', values='Corrélation')
    # Affichage de la matrice de corrélation
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(heatmap_data, cmap="coolwarm",fmt=".2f", ax=ax)
    # Ajout des annotations manuellement
    for i in range(len(heatmap_data)):
        for j in range(len(heatmap_data.columns)):
            ax.text(j + 0.5, i + 0.5, '{:.2f}'.format(heatmap_data.iloc[i, j]), ha='center', va='center', color='white')
    ax.set_title('Matrice de corrélation')
    ax.set_xlabel('Variables')
    ax.set_ylabel('Variables')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    return fig

def msml_preds():
    df_predictions_msml = pd.read_csv('data/modeles/msml_preds.csv')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_predictions_msml.index, y=df_predictions_msml["comptage_horaire"], mode='markers', marker_color="#3D85C6", name="valeurs réelles"))
    fig.add_trace(go.Scatter(x=df_predictions_msml.index, y=df_predictions_msml["Prédictions"], mode='markers', marker_color="#DE8326", name="valeurs prédites"))
    fig.update_layout(title="Comparaison entre les prédictions et les valeurs réelles", legend=dict(orientation="h",yanchor="bottom", y=1.02,xanchor="right",x=1))
    fig.update_yaxes(title_text='Comptage horaire')
    return fig


# global

@st.cache_data
def load_metrics():
    metrics = pd.read_csv('data/modeles/metrics.csv')
    return metrics