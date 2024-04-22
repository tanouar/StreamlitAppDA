import pandas as pd
import numpy as np
import plotly.express as px
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import utils.velo_load_datas as datas

def get_boxplot(df_velo) :
    plt.figure(figsize=(10,6))
    plt.boxplot(df_velo['Comptage horaire'])
    plt.title('Boxplot représentant les valeurs extrêmes de la colonne Comptage horaire')
    plt.show()

def get_compt_temp(df_velo, temp):
    df_velo_2022 = datas.get_df_2022(df_velo)
    df_velo_2023 = datas.get_df_2023(df_velo)
    if temp == 'mois':

        passages_par_mois_2022 = df_velo_2022.groupby('mois')['Comptage horaire'].mean()
        passages_par_mois_2023 = df_velo_2023.groupby('mois')['Comptage horaire'].mean()
        plt.figure()
        passages_par_mois_2022.plot(kind='line', marker='o', color='#005F73',label='2022')
        passages_par_mois_2023.plot(kind='line', marker='o', color='#CA6702',label='2023')
        plt.xlabel('Mois')
        plt.ylabel('Nombre moyen de passages')
        plt.title('Nombre de passage moyen par mois')
        plt.xticks(range(1, 13), ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre'], rotation=45)
        plt.grid(True)
        plt.legend()
        return plt
    elif temp == 'jour':
        ordre_jours_semaine = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df_velo_2023['jour_de_semaine'] = pd.Categorical(df_velo_2023['jour_de_semaine'], categories=ordre_jours_semaine, ordered=True)
        df_velo_2023 = df_velo_2023.sort_values('jour_de_semaine')
        df_velo_2022['jour_de_semaine'] = pd.Categorical(df_velo_2022['jour_de_semaine'], categories=ordre_jours_semaine, ordered=True)
        df_velo_2022 = df_velo_2022.sort_values('jour_de_semaine')
        passages_par_jour_2023 = df_velo_2023.groupby('jour_de_semaine')['Comptage horaire'].mean()
        passages_par_jour_2022 = df_velo_2022.groupby('jour_de_semaine')['Comptage horaire'].mean()
        plt.figure()
        passages_par_jour_2022.plot(kind='line', marker='o', color='#005F73',label='2022')
        passages_par_jour_2023.plot(kind='line', marker='o', color='#CA6702',label='2023')
        plt.xlabel('Jour')
        plt.ylabel('Nombre moyen de passages')
        plt.title('Nombre de passages moyen par jour')
        plt.grid(True)
        plt.legend()
        return plt
    elif temp == 'heure':
        passages_par_heure_2023 = df_velo_2023.groupby('heure')['Comptage horaire'].mean()
        passages_par_heure_2022 = df_velo_2022.groupby('heure')['Comptage horaire'].mean()
        plt.figure()        
        passages_par_heure_2022.plot(kind='line', marker='o', color='#005F73', label='2022')
        passages_par_heure_2023.plot(kind='line', marker='o', color='#CA6702',label='2023')
        plt.xlabel('Heure')
        plt.ylabel('Nombre moyen de passages')
        plt.title('Nombre moyen de passages par heure')
        plt.grid(True)
        plt.legend()
        return plt
    elif temp == 'semaine':
        comptage_moyen_2022_par_semaine = df_velo_2022.groupby('Numéro de semaine de l\'année')['Comptage horaire'].mean()
        comptage_moyen_2023_par_semaine = df_velo_2023.groupby('Numéro de semaine de l\'année')['Comptage horaire'].mean()
        plt.figure()
        plt.plot(comptage_moyen_2022_par_semaine.index, comptage_moyen_2022_par_semaine.values, marker='o', linestyle='-', color='#005F73', label='2022')
        plt.plot(comptage_moyen_2023_par_semaine.index, comptage_moyen_2023_par_semaine.values, marker='o', linestyle='-', color='#CA6702', label='2023')
        plt.xlabel('Semaines de l\'année')
        plt.xticks(range(1, 53, 3))
        plt.ylabel('Comptage horaire moyen')
        plt.title('Fréquentation moyenne journalière à Paris')
        legende_vacances = False
        for semaine, vacances in df_velo.groupby('Numéro de semaine de l\'année')['Vacances'].max().items():
            if vacances and not legende_vacances:
                plt.axvspan(semaine - 0.5, semaine + 0.5, color='lightgrey', label='Vacances scolaires', alpha=0.5)
                legende_vacances = True
            elif vacances:
                plt.axvspan(semaine - 0.5, semaine + 0.5, color='lightgrey', alpha=0.5)
        plt.legend()
        return plt
    else :
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        df_velo_ordered=df_velo_2023.copy()
        df_velo_ordered['jour_de_semaine'] = pd.Categorical(df_velo_ordered['jour_de_semaine'], categories=order, ordered=True)
        df_grouped_heurejour=df_velo_ordered.groupby(["jour_de_semaine","heure"]).agg({"Comptage horaire":np.mean})
        sns.lineplot(x="heure",y="Comptage horaire",hue="jour_de_semaine",data=df_grouped_heurejour)
        plt.title("¨Comptage par heure et jour de la semaine - 2023")
        legend = plt.legend()
        legend.set_title('Jour de la semaine')
        plt.xticks(ticks=range(25),rotation=90)
        return plt

def get_map_compt(df_velo):
    compteurs = datas.get_loc_compteurs(df_velo)
    top_index = compteurs.head(5).index
    flop_index = compteurs.tail(5).index
    n_mean = compteurs['Comptage horaire'].mean()
    m = folium.Map(location=[48.862, 2.36], zoom_start=13)
    for index, compt in compteurs.iterrows():
        radius = (compt['Comptage horaire']/n_mean)*200
        color = 'orange'
        if index in top_index:
            color = 'red'
            folium.Marker(location=[compt['lat'],compt['lon']], icon=folium.Icon(icon="glyphicon glyphicon-plus",color=color), popup='top '+str(index+1),  tooltip=compt['Nom du compteur'], radius=2).add_to(m)
        elif index in flop_index:
            color = 'green'
            folium.Marker(location=[compt['lat'],compt['lon']], icon=folium.Icon(icon='glyphicon glyphicon-minus', color=color), popup='flop '+str(len(compteurs)-index),  tooltip=compt['Nom du compteur'], radius=2).add_to(m)
        folium.Circle(location=[compt['lat'], compt['lon']], radius=radius, color=color, weight=1, fill_opacity=0.4, opacity=1, fill=True, tooltip=compt['Nom du compteur']+': '+str(int(round(compt["Comptage horaire"], 0)))+' vélos par heure').add_to(m)
    return m, compteurs.head(5), compteurs.tail(5)

def get_meteo(df_velo, choice):
    if choice=='pluie':
        mean_by_rain_hour_site = df_velo[df_velo['annee']==2023].groupby(["pluvieux", "heure", "Nom du site de comptage"])['Comptage horaire'].mean()
        mean_by_rain_hour_site = mean_by_rain_hour_site.reset_index()
        mean_by_rain_hour = mean_by_rain_hour_site.groupby(["pluvieux", "heure"])["Comptage horaire"].mean()
        mean_by_rain_hour = mean_by_rain_hour.reset_index()
        mean_by_rain = mean_by_rain_hour.groupby(["pluvieux"])["Comptage horaire"].mean()
        fig = px.bar(x=mean_by_rain.index, y=mean_by_rain, title="Moyenne de comptage par heure et station pour chaque type de pluie sur l'année 2023", width=800)
        fig.update_xaxes(title="Types de pluie")
        fig.update_yaxes(title="Moyenne de comptage")
    else:
        fig = px.scatter(df_velo[df_velo['annee']==2023], x=' T', y='Comptage horaire', color="pluvieux", color_discrete_sequence=px.colors.qualitative.Prism, title="Comptages selon la température sur l'année 2023", width=800)
        fig.update_xaxes(title="Températures")
        fig.update_yaxes(range=[0, 3000], title="Comptage")
    return fig

def get_map_stations(compteurs, stations_meteo):
    colors = {75106001:"darkred", 75107005:"green", 75110001:"purple", 75114001:"blue", 75116008:"orange"}
    map = folium.Map(location=[48.85, 2.35], zoom_start=13)
    for index, compt in compteurs.iterrows():
        map.add_child(folium.RegularPolygonMarker(location=[compt[0],compt[1]], popup=index, color=colors[compt[2]], fill_color=colors[compt[2]], radius=6))
    for index, station in stations_meteo.iterrows() :
        map.add_child(folium.Marker(location=[station[0],station[1]], icon=folium.Icon(icon="cloud",color=colors[index]), popup=index,  radius=6))
    return map