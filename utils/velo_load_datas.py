# Python
import pandas as pd
import numpy as np

def load_velo():
  df_2023=pd.read_csv('data/2023_comptage-velo-donnees-compteurs.csv', sep=';')
  df_2022=pd.read_csv('data/2022_comptage-velo-donnees-compteurs.csv', sep=';')
  df_velo_2022 = df_2022.copy()
  df_velo_2023 = df_2023.copy()
  return df_velo_2022, df_velo_2023

def clean_velo(df_velo_2022, df_velo_2023):
  """conversion datetime et colonnes dates"""
  # conversion en datetime (formats différents selon si 2022 ou 2023)
  # 2022-01-01T00:00:00
  df_velo_2022['Date et heure de comptage'] = pd.to_datetime(df_velo_2022['Date et heure de comptage'], format='%Y-%m-%dT%H:%M:%S')
  # 2023-01-01T07:00:00+01:00
  df_velo_2023['Date et heure de comptage'] = pd.to_datetime(df_velo_2023['Date et heure de comptage'].astype(str).str[:-6], format='%Y-%m-%dT%H:%M:%S')
  # creation d'un df pour les 2 ans sur lequel on va faire le nettoyage
  df_velo = pd.concat([df_velo_2022, df_velo_2023], axis =0)
  # suppression des colonnes liées aux images
  df_velo = df_velo.drop(['Lien vers photo du site de comptage', 'ID Photos', 'test_lien_vers_photos_du_site_de_comptage_', 'id_photo_1', 'url_sites', 'type_dimage'], axis = 1)
  # colonnes année et jours
  from datetime import datetime as dt
  df_velo['annee'] = df_velo['Date et heure de comptage'].dt.year
  df_velo['mois'] = df_velo['Date et heure de comptage'].dt.month
  df_velo['jour_semaine'] = df_velo['Date et heure de comptage'].dt.weekday
  df_velo['heure'] = df_velo['Date et heure de comptage'].dt.hour
  df_velo['Week-end'] = df_velo['jour_semaine'].isin([5, 6])
  df_velo['jour_de_semaine'] = df_velo['Date et heure de comptage'].dt.day_name()
  df_velo['Numéro de semaine de l\'année'] = df_velo['Date et heure de comptage'].dt.isocalendar().week
  #Ajout Colonnes Melissa
  df_velo['Matin'] = df_velo['heure'].apply(lambda x: True if 6 <= x < 12 else False)
  df_velo['Après-midi'] = df_velo['heure'].apply(lambda x: True if 12 <= x < 18 else False)
  df_velo['Soir'] = df_velo['heure'].apply(lambda x: True if 18 <= x < 24 else False)
  df_velo['Nuit'] = df_velo['heure'].apply(lambda x: True if 0 <= x < 6 else False)
  # vacances scolaires
  vacances = pd.date_range(start='2022-01-01', end='2022-01-02').append(
      pd.date_range(start='2022-02-19', end='2022-03-06')).append(
      pd.date_range(start='2022-04-23', end='2022-05-08')).append(
      pd.date_range(start='2022-07-07', end='2022-08-31')).append(
      pd.date_range(start='2022-10-22', end='2022-11-06')).append(
      pd.date_range(start='2022-12-17', end='2023-01-02')).append(
      pd.date_range(start='2023-02-18', end='2023-03-05')).append(
      pd.date_range(start='2023-04-22', end='2023-05-08')).append(
      pd.date_range(start='2023-07-08', end='2023-09-03')).append(
      pd.date_range(start='2023-10-21', end='2023-11-05')).append(
      pd.date_range(start='2023-12-23', end='2024-01-07'))
  # Création de la colonne "Vacances" dans le DataFrame
  df_velo['Vacances'] = df_velo['Date et heure de comptage'].dt.date.astype(str).isin(vacances.date.astype(str)).astype(int)
  """colonnes geographiques"""
  # séparation latitude et longitude
  df_velo[['lat', 'lon']] = df_velo['Coordonnées géographiques'].str.split(',', expand=True)
  df_velo[['lat', 'lon']] = df_velo[['lat', 'lon']].astype(float)
  df_velo = df_velo.drop('Coordonnées géographiques', axis=1)
  """gestion na"""
  # gestion des na sur 'Nom du site de comptage' et les données geographiques pour continuer (ne concerne que 2 compteurs au même site de comptage)
  df_velo['Nom du site de comptage'] = df_velo['Nom du site de comptage'].fillna(df_velo['Nom du compteur'].str[:-6])
  df_velo['lat'] = df_velo['lat'].fillna(48.846389)
  df_velo['lon'] = df_velo['lon'].fillna(2.315000)
  # retrait des compteurs qui n'obtiennent aucun comptage sur les 2 années 2022-2023
  compts = df_velo.groupby('Identifiant du compteur')['Comptage horaire'].sum().reset_index()
  df_compteurs_inactifs = df_velo[df_velo['Identifiant du compteur'].isin(compts[compts['Comptage horaire']==0]['Identifiant du compteur'])]
  df_velo = df_velo[~df_velo['Identifiant du compteur'].isin(compts[compts['Comptage horaire']==0]['Identifiant du compteur'])]

  # création des 2 df nettoyés pour 2022 et 2023
  df_velo_2022 = df_velo[df_velo['annee']==2022].copy()
  df_velo_2023 = df_velo[df_velo['annee']==2023].copy()
  return df_velo, df_velo_2022, df_velo_2023

def load_meteo():
  # création du df_meteo
  df_meteo_2023=pd.read_csv('../data/meteo/H_75_latest-2023-2024.csv', sep=';')
  df_meteo_2022=pd.read_csv('../data/meteo/H_75_previous-2020-2022.csv', sep=';')
  df_meteo_complet = pd.concat([df_meteo_2022, df_meteo_2023], axis =0)
  df_meteo = df_meteo_complet[['NUM_POSTE', 'NOM_USUEL', 'LAT', 'LON', 'AAAAMMJJHH', 'RR1', ' T']]
  return df_meteo

def add_meteo(df_velo, df_meteo):
  """Nettoyage"""
  # conversion date en datetime
  df_meteo.loc[:,'AAAAMMJJHH'] = pd.to_datetime(df_meteo['AAAAMMJJHH'], format='%Y%m%d%H')
  # suppression de la station 75114007 (en doublon, postée au même endroit, avec 75114001 mais sans les variables pluie 'RR1')
  df_meteo = df_meteo[df_meteo['NUM_POSTE']!=75114007]
  # assignation des labels selon l'intensité de la pluie
  df_meteo.loc[:,'pluvieux'] = pd.cut(df_meteo['RR1'], bins=[0, 0.1, 5.9, df_meteo['RR1'].max()], labels = ['Pas de pluie', 'Pluie modérée', 'Pluie intense'], include_lowest = True)
  """Assignation de la station meteo selon les coordonnées geo"""
  # creation de 2 df groupés par station de meteo et nom de site de comptage
  stations_meteo = df_meteo.groupby("NUM_POSTE")[['LAT', 'LON']].agg(pd.Series.mode)
  compteurs = df_velo.groupby("Nom du site de comptage")[['lat', 'lon']].agg(lambda x: x.mode()[0])
  # assigne pour chaque site de comptage unique (compteurs) la station meteo la plus proche (calcul via la formule de Haversine)
  
  def get_close_station(lat_compt, lon_compt):
    min_distance = 1000
    close_station = ''
    for index, station in stations_meteo.iterrows() :
      p = 0.017453292519943295 # Pi/180
      a = 0.5 - np.cos((station['LAT'] - lat_compt) * p)/2 + np.cos(lat_compt * p) * np.cos(station['LAT'] * p) * (1 - np.cos((station['LON'] - lon_compt) * p)) / 2
      distance = 0.6213712 * 12742 * np.arcsin(np.sqrt(a))
      if distance < min_distance:
        min_distance = distance
        close_station = index
    return close_station

  compteurs["station_meteo"] = compteurs.apply(lambda x: get_close_station(x['lat'], x['lon']), axis=1)
  """Attribution de la meteo à notre df_velo"""
  # à partir des données créées ci-dessus on applique notre station meteo sur chaque ligne de df_velo
  df_velo = pd.merge(df_velo, compteurs['station_meteo'], on = 'Nom du site de comptage', how='left')
  # merge la température et la pluie par heure et station meteo
  df_velo_meteo = pd.merge(df_velo, df_meteo[['AAAAMMJJHH', 'NUM_POSTE',' T', 'pluvieux']], left_on = ['Date et heure de comptage', 'station_meteo'], right_on = ['AAAAMMJJHH', 'NUM_POSTE'], how='left').drop(['AAAAMMJJHH', 'NUM_POSTE'], axis=1)
  # beaucoup de valeurs nulles, on les remplis selon l'intensité de la pluie des autres heures de la journée (on sort par heure et par station, et on associe la prochaine donnée "pluvieuse" à notre na)
  df_velo_meteo["pluvieux"] = df_velo_meteo.sort_values(by=['Date et heure de comptage', 'station_meteo'], ascending=False)["pluvieux"].bfill()
  return df_velo_meteo, stations_meteo, compteurs
