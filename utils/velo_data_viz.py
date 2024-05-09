import plotly.express as px
from plotly import graph_objs as go
import matplotlib.pyplot as plt
import folium
import utils.velo_load_datas as datas
import streamlit as st

def get_boxplot(df_velo) :
    plt.figure(figsize=(10,6))
    plt.boxplot(df_velo['Comptage horaire'])
    plt.title('Boxplot représentant les valeurs extrêmes de la colonne Comptage horaire')
    plt.show()

def get_compt_temp(df_velo, temp):
    color_blue="#3D85C6"
    color_orange="#DE8326"
    df_velo_2022 = datas.get_df_2022(df_velo)
    df_velo_2023 = datas.get_df_2023(df_velo)
    if temp == 'mois':
        mois = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
        mois_2022 = df_velo_2022.groupby('mois')['Comptage horaire'].mean()
        mois_2023 = df_velo_2023.groupby('mois')['Comptage horaire'].mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mois_2022.index, y=mois_2022, line=dict(color=color_blue),name='2022', mode='lines'))
        fig.add_trace(go.Scatter(x=mois_2023.index, y=mois_2023, line=dict(color=color_orange),name='2023', mode='lines'))
        fig.update_layout(title='Nombre de passages moyen par mois',yaxis_title='Comptage horaire moyen', 
                          xaxis = dict(tickmode = 'array', tickvals = list(range(1, 13)),ticktext =mois))
        fig.add_vrect(x0=7, x1=8, annotation_text="vacances d'été", annotation_position="top left", fillcolor="lightgrey", opacity=0.35, line_width=0)
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        return fig
    elif temp == 'jour':
        jours = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        jour_2023 = df_velo_2023.groupby('jour_semaine')['Comptage horaire'].mean()
        jour_2022 = df_velo_2022.groupby('jour_semaine')['Comptage horaire'].mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=jour_2022.index, y=jour_2022, line=dict(color=color_blue),name='2022', mode='lines'))
        fig.add_trace(go.Scatter(x=jour_2023.index, y=jour_2023, line=dict(color=color_orange),name='2023', mode='lines'))
        fig.update_layout(title='Nombre de passages moyen par jour',yaxis_title='Comptage horaire moyen', 
                          xaxis = dict(tickmode = 'array', tickvals = list(range(0, 7)),ticktext = jours))
        fig.add_vrect(x0=5, x1=6, annotation_text="week-end", annotation_position="top right", fillcolor="lightgrey", opacity=0.35, line_width=0)
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        return fig
    elif temp == 'heure':
        heure_2023 = df_velo_2023.groupby('heure')['Comptage horaire'].mean()
        heure_2022 = df_velo_2022.groupby('heure')['Comptage horaire'].mean()
        heures = [str(h)+'h' for h in heure_2022.index ]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=heure_2022.index, y=heure_2022, line=dict(color=color_blue),name='2022', mode='lines'))
        fig.add_trace(go.Scatter(x=heure_2023.index, y=heure_2023, line=dict(color=color_orange),name='2023', mode='lines'))
        fig.update_layout(title='Nombre de passages moyen par heure',yaxis_title='Comptage horaire moyen', 
                          xaxis = dict(tickmode = 'array', tickvals = list(range(0, 24)),ticktext = heures))
        fig.add_vrect(x0=8, x1=9, annotation_text="heure de pointe", annotation_position="top left", fillcolor="lightgrey", opacity=0.35, line_width=0)
        fig.add_vrect(x0=18, x1=19, annotation_text="heure de pointe", annotation_position="top left", fillcolor="lightgrey", opacity=0.35, line_width=0)
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        return fig
    elif temp == 'semaine':
        semaine_2022 = df_velo_2022.groupby('Numéro de semaine de l\'année')['Comptage horaire'].mean()
        semaine_2023 = df_velo_2023.groupby('Numéro de semaine de l\'année')['Comptage horaire'].mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=semaine_2022.index, y=semaine_2022, line=dict(color=color_blue),name='2022', mode='lines'))
        fig.add_trace(go.Scatter(x=semaine_2023.index, y=semaine_2023, line=dict(color=color_orange),name='2023', mode='lines'))
        fig.update_layout(title='Nombre de passages moyen par semaine',yaxis_title='Comptage horaire moyen')
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        fig.update_xaxes(tickvals = list(range(1, 53, 3)))
        legende_vacances = False
        for semaine, vacances in df_velo.groupby('Numéro de semaine de l\'année')['Vacances'].max().items():
            if vacances and not legende_vacances:
                fig.add_vrect(x0=semaine - 0.5, x1=semaine + 0.5, annotation_text="vacances scolaires", annotation_position="top left", fillcolor="lightgrey", opacity=0.35, line_width=0)
                legende_vacances = True
            elif vacances:
                fig.add_vrect(x0=semaine - 0.5, x1=semaine + 0.5, fillcolor="lightgrey", opacity=0.35, line_width=0)
        

        return fig
    else :
        jour_2023=df_velo.groupby(["jour_semaine","heure"])["Comptage horaire"].mean().reset_index()
        jours = {'0':'Lundi', '1': 'Mardi',  '2': 'Mercredi', '3': 'Jeudi', '4': 'Vendredi', '5': 'Samedi', '6': 'Dimanche'}
        heures = [str(h)+'h' for h in jour_2023.heure]
        fig = px.line(jour_2023,x="heure",y="Comptage horaire", color='jour_semaine', color_discrete_sequence=px.colors.qualitative.Prism, labels={"jour_semaine": "jour"})
        fig.update_layout(title='Nombre de passages moyen par heure et par jour',yaxis_title='Comptage horaire moyen', xaxis_title=None,
                          xaxis = dict(tickmode = 'array', tickvals = list(range(0, 24)),ticktext = heures))
        fig.for_each_trace(lambda t: t.update(name = jours[t.name],legendgroup = jours[t.name],hovertemplate = t.hovertemplate.replace(t.name, jours[t.name])))
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        return fig

def get_map_compt(df_velo):
    compteurs = datas.get_loc_compteurs(df_velo)
    top_index = compteurs.head(5).index
    flop_index = compteurs.tail(5).index
    n_mean = compteurs['Comptage horaire'].mean()
    m = folium.Map(location=[48.862, 2.34], zoom_start=13, scrollWheelZoom=False)
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

@st.cache_data
def get_meteo(df_velo, choice):
    df_velo_2023 = datas.get_df_2023(df_velo)
    if choice=='pluie':
        mean_by_rain_hour_site = df_velo_2023.groupby(["pluvieux", "heure", "Nom du site de comptage"])['Comptage horaire'].mean()
        mean_by_rain_hour_site = mean_by_rain_hour_site.reset_index()
        mean_by_rain_hour = mean_by_rain_hour_site.groupby(["pluvieux", "heure"])["Comptage horaire"].mean()
        mean_by_rain_hour = mean_by_rain_hour.reset_index()
        mean_by_rain = mean_by_rain_hour.groupby(["pluvieux"])["Comptage horaire"].mean()
        
        fig = px.pie(values=mean_by_rain, names=mean_by_rain.index, color=mean_by_rain.index, color_discrete_map={'Pas de pluie':'#5f4690', 'Pluie modérée':'#1d6996', 'Pluie intense':'#38a6a5'}, title='Comptage horaire moyen par densité de pluie - 2023')
        fig.update_layout(hovermode=False, legend=dict(title=None, orientation="h",yanchor="bottom", y=1.02,xanchor="right",x=1))
    else:
        fig = px.scatter(df_velo_2023, x=' T', y='Comptage horaire', color="pluvieux", color_discrete_sequence=px.colors.qualitative.Prism, title="Comptages selon les températures - 2023")
        fig.update_xaxes(title="Températures")
        fig.update_yaxes(range=[0, 2500], title="Comptage")
        fig.update_layout(hovermode=False, legend=dict(title=None, orientation="h",yanchor="bottom", y=1.02,xanchor="right",x=1))
    return fig

def get_map_stations(compteurs, stations_meteo):
    colors = {75106001:"darkred", 75107005:"green", 75110001:"purple", 75114001:"blue", 75116008:"orange"}
    map = folium.Map(location=[48.85, 2.35], zoom_start=13)
    for index, compt in compteurs.iterrows():
        map.add_child(folium.RegularPolygonMarker(location=[compt[0],compt[1]], popup=index, color=colors[compt[2]], fill_color=colors[compt[2]], radius=6))
    for index, station in stations_meteo.iterrows() :
        map.add_child(folium.Marker(location=[station[0],station[1]], icon=folium.Icon(icon="cloud",color=colors[index]), popup=index,  radius=6))
    return map