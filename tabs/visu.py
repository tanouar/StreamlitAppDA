import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

title = "Températures terrestres"
sidebar_name = "Visualisation"

def run():
    st.write("  ")
    st.header("Data visualisation")
    st.markdown("---")
    df = pd.read_csv("Data/owid.csv")
   
    # Filtre pour ne garder que les pays du top 15
    top15_countries = ['United States', 'China', 'Russia', 'Germany', 'Japan', 'India', 'United Kingdom', 'Canada', 'France', 'Italy', 'Poland', 'South Africa', 'Mexico', 'South Korea', 'Ukraine', 'World']
    df_top15 = df[df['country'].isin(top15_countries)].copy()
    country_top_recent = df_top15.loc[(df_top15["year"]>=1950) &(df_top15.country != 'Monde') ]
   
    # Traduction des noms de pays en français
    country_translation = {'United States':'États-Unis', 'China':'Chine', 'Russia':'Russie','Germany': 'Allemagne', 'Japan':'Japon', 'India':'Inde', 'United Kingdom':'Royaume Uni', 'Canada':'Canada', 'France': 'France', 'Italy':'Italie', 'Poland':'Pologne', 'South Africa':'Afrique du Sud', 'Mexico':'Mexique', 'South Korea': 'Corée du Sud', 'Ukraine': 'Ukraine', 'World':'Monde'}
    df_top15['country'].replace(country_translation, inplace=True)

    # Création des graphiques
    fig_co2 = px.line(df_top15, x='year', y='co2', color='country', width=800, height=600)
    fig_ch4 = px.line(df_top15[df_top15['year'] >= 1990], x='year', y='methane', color='country', width=800, height=600)
    fig_n2o = px.line(df_top15[df_top15['year'] >= 1990], x='year', y='nitrous_oxide', color='country', width=800, height=600)

    country_top_recent['Autres sources'] = country_top_recent['cumulative_flaring_co2'] + country_top_recent['cumulative_other_co2'].copy()
    country_top_recent = country_top_recent.rename(columns={'cumulative_gas_co2': 'Combustion du gaz naturel', 'cumulative_oil_co2':'Combustion du pétrole', 'cumulative_coal_co2': 'Combustion du charbon'})
    fig_emissions = px.histogram(country_top_recent, x='country', y=['Combustion du gaz naturel', 'Combustion du pétrole', 'Combustion du charbon', 'Autres sources']).update_xaxes(categoryorder='total descending')
    fig_emissions.update_layout(xaxis_title='Pays', yaxis_title='Émissions cumulées de CO2 par combustible', width=800, height=600, legend_title_text='', legend=dict(y=1, x=0.68, bgcolor='rgba(255,255,255,0)'))

    st.write('**1- Contribution relative des pays les plus emetteurs de gaz à effet de serre**')
    
    select_graph = st.selectbox('Sélectionnez une figure à visualiser ', ['Émissions de CO2', 'Émissions de CH4', 'Émissions de NO2', 'Répartition des émissions de CO2 par combustible'])
    if select_graph == 'Émissions de CO2':
        st.plotly_chart(fig_co2)
        st.write("➽ Depuis les années 1900, le niveau mondial d'émissions de CO2 a augmenté très rapidement. Cette augmentation s'explique par la croissance économique, l'industrialisation et l'augmentation de la population mondiale.")
    elif select_graph == 'Émissions de CH4':
        st.plotly_chart(fig_ch4)
        st.write("➽ Tout comme pour les émissions de CO2, la Chine est responsable de la majeure partie des émissions de CH4, suivie par les Etats Unis, la Russie et l'Inde. Les émissions de CH4 sont associées à l'agriculture, à la production de combustibles fossiles et à la production de déchets.")
    elif select_graph == 'Émissions de NO2':
        st.plotly_chart(fig_n2o)
        st.write("➽ La Chine est le plus grand producteur d'émissions de N2O, suivis par les États-Unis et l'Inde. Les émissions de N2O sont particulièrement difficiles à diminuer car elles sont associées à l'utilisation de fertilisants azotés, qui sont essentiels à la production alimentaire.")
    else:
        st.plotly_chart(fig_emissions)
        st.write("➽ Le charbon est le principal combustible utilisé dans tous les pays examinés, excepté le Japon, le Canada, l'Italie et le Mexique. L'utilisation du pétrole est la plus élevée aux États-Unis tandis que la Chine utilise principalement le charbon. Le gaz naturel est plus utilisé en Russie et au Japon. Il est clair que la dépendance au charbon persiste malgré les efforts mondiaux pour réduire les émissions de gaz à effet de serre.")
    
    st.write("  ")

    st.write('**2- Contribution des gaz à effet de serre à l\'évolution des températures dans les pays les plus pollueurs**') 

    tmp=['United States', 'China', 'Russia','Germany', 'Japan', 'India', 'United Kingdom', 'Canada', 'France', 'Italy',
      'Poland', 'South Africa', 'Mexico', 'South Korea','Australia', 'Ukraine', 'Brazil','Argentina','Colombia','Indonesia']
    country_ext=df[(df['country'].isin(tmp)) & (df.year>=1950)]
    country_ext=country_ext.rename(columns={'temperature_change_from_co2': 'Variation de t° dûe au CO2',
                                                      'temperature_change_from_ch4':'Variation de t° dûe au CH4',
                                                      'temperature_change_from_n2o': 'Variation de t° dûe au N2O'})
    country_ext['country'].replace({'United States':'États-Unis', 'China':'Chine', 'Russia':'Russie','Germany': 'Allemagne',
                                'Japan':'Japon', 'India':'Inde', 'United Kingdom':'Royaume Uni', 'Canada':'Canada',
                                'France': 'France', 'Italy':'Italie', 'Poland':'Pologne', 'South Africa':'Afrique de Sud',
                                'Mexico':'Mexique', 'South Korea': 'Corée de Sud','Australia': 'Australie',
                                'Ukraine' : 'Ukraine', 'Brazil' : 'Brésil','Argentina' : 'Argentine',
                                'Colombia' : 'Colombie','Indonesia' :'Indonésie'}, inplace= True)
    country_grouped_mean = country_ext.groupby('country').mean(numeric_only=True)
    fig = px.bar(country_grouped_mean, y=['Variation de t° dûe au CO2','Variation de t° dûe au CH4','Variation de t° dûe au N2O'])
    fig.update_xaxes(categoryorder='total descending')
    fig.update_layout(xaxis_title='Pays', yaxis_title='Variation t°', width=650, height=600, legend_title_text='',
                  legend=dict(y=1, x=0.68, bgcolor='rgba(255,255,255,0)'))
    st.plotly_chart(fig)
    st.write("➽ Ici, on remarque à quel point les Etats-Unis ont agi sur le réchauffement des températures depuis 1950 via les émissions de gaz à effet de serre (et surtout via le CO2). En Inde, ce n'est pas le CO2 le plus gros responsable du réchauffement mais le méthane. Il est très important aussi en Chine.")

    st.write("  ")

    st.write('**3- Carte pour visualiser l\'évolution de températures par année et par pays**') 

    df_merged = pd.read_csv("Data/merged_owid_temp.csv")
    
    df_recent = df_merged[(df_merged.year>=1980) & (df_merged.year<=2017) ]
    fig_temp = px.choropleth(df_recent, locations="iso_code", color="temperature",
                   hover_name='country',
                   color_continuous_scale='ylorbr',
                   projection='natural earth',
                   height=500, width=650,
                   animation_frame = df_recent.year)

    fig_temp.update_layout(font={'size':15, 'color':'Black'})
    cmin = df_recent['temperature'].min()
    cmax = df_recent['temperature'].max()

    fig_temp.update_layout(coloraxis_colorbar=dict(
    title="T en °C",
    tickvals=np.linspace(cmin, cmax, num=10),  # Les valeurs des ticks de la barre de légende
    ticktext=np.linspace(cmin, cmax, num=10).astype(int),  # Les labels correspondants
))

# Spécifier les limites de l'échelle de couleur
    fig_temp.update_layout(geo=dict(showframe=False, showcoastlines=False), coloraxis=dict(
    cmin=cmin,  # Valeur minimale de l'échelle de couleur
    cmax=cmax,  # Valeur maximale de l'échelle de couleur
))
    st.plotly_chart(fig_temp)

    st.write("➽ On note qu'après les années 2010, les températures semblent augmenter visiblement sur l'ensemble du globe.")

    st.write("")
    st.write("")
    st.write("")
    st.write("")

   
        
    
        
   
       



    
    
