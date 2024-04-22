import pandas as pd
import numpy as np

def load_velo():
  df_velo=pd.read_csv('data/df_velo_meteo.zip', index_col=0)
  return df_velo

def get_loc_sites_comptage(df_velo):
  site_comptage = df_velo.groupby("Nom du site de comptage")[['lat', 'lon']].agg(lambda x: x.mode()[0])
  return site_comptage

def get_loc_compteurs(df_velo):
  compteurs = df_velo.groupby("Nom du compteur").agg({'lat': lambda x: x.mode()[0], 'lon': lambda x: x.mode()[0], 'Comptage horaire': 'mean'})
  compteurs = compteurs.sort_values(by='Comptage horaire', ascending=False).reset_index()
  return compteurs

def get_stations_meteo(df_velo):
  stations_meteo = df_velo.groupby("Nom du site de comptage")[['lat', 'lon', 'station_meteo']].agg(lambda x: x.mode()[0])
  return stations_meteo

def get_df_2022 (df_velo):
  df_velo_2022 = df_velo[df_velo['annee']==2022].copy()
  return df_velo_2022

def get_df_2023 (df_velo):
  df_velo_2023 = df_velo[df_velo['annee']==2023].copy()
  return df_velo_2023