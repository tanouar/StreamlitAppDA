import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
# import statsmodels.api as sm

title = "Températures terrestres"
sidebar_name = "Statistiques"

def run():
    st.write("  ")
    st.header("Statistiques")
    st.markdown("---")
    df=pd.read_csv("Data/merged_owid_temp.csv", index_col=0) 
    top15_countries = ['United States', 'China', 'Russia', 'Germany', 'Japan', 'India', 'United Kingdom', 'Canada', 'France', 'Italy', 'Poland', 'South Africa', 'Mexico', 'South Korea', 'Ukraine', 'World']
    df_recent = df[(df.year>=1980) & (df.year<=2017) ]
    temp_world=df_recent[df_recent['country'].isin(top15_countries)].copy()
    df_corr=temp_world[['gdp', 'co2', 'population',
       'co2_per_capita', 'methane', 'nitrous_oxide', 'cumulative_gas_co2',
       'cumulative_oil_co2', 'cumulative_flaring_co2', 'cumulative_coal_co2',
       'cumulative_other_co2', 'temperature_change_from_ch4',
       'temperature_change_from_co2', 'temperature_change_from_ghg',
       'temperature_change_from_n2o', 'temperature']]
    df_corr=df_corr.rename(columns={'gdp': 'PIB','methane':'CH4', 'nitrous_oxide':'N2O', 'cumulative_gas_co2':'Combustion du gaz naturel',
       'cumulative_oil_co2':'Combustion du pétrol', 'cumulative_flaring_co2':'Le trochage', 'cumulative_coal_co2':'Combustion du charbon',
       'cumulative_other_co2': 'Autres combustibles', 'temperature_change_from_ch4':'Variation de t° dûe au CH4',
       'temperature_change_from_co2':'Variation de t° dûe au CO2', 'temperature_change_from_ghg':'Variation de t° dûe au ghg',
       'temperature_change_from_n2o': 'Variation de t° dûe au N2O', 'temperature':'Température'})
    df_corr=df_corr.dropna()
    df=df.dropna()

    if st.button("Distribution de la variable cible 'Température'"):
        plt.figure(figsize=(10, 6))
        sns.histplot(df['temperature'], kde=True, color='green')
        plt.ylabel('Fréquence')
        plt.xlabel('Température')
        fig = plt.gcf()
        st.pyplot(fig)
        p_value = stats.shapiro(df['temperature'])[1]
        if p_value > 0.05:
            st.write("➽ La p-value (= {:.2f}) est supérieure à 0.05 donc la variable température suit une distribution normale.".format(p_value))
        else:
            st.write("➽ La p-value (= {:.2f}) est inférieure à 0.05 donc la variable température ne suit pas une distribution normale.".format(p_value))
    
    if st.button("Matrice de corrélation") :
        correlation_matrix=df_corr.corr()
        fig_corr = px.imshow(correlation_matrix,labels=dict(color="Corrélation"), x=correlation_matrix.columns, y=correlation_matrix.columns, color_continuous_scale='Bluyl')
        fig_corr.update_layout(width=600, height=500)
        st.plotly_chart(fig_corr)
        st.write("➽ Présence de corrélation positive entre les températures dues aux gaz à effet de serre et leurs émissions.")
        st.write("➽ Absence de corrélation positive entre la variable température et le reste des variables.")

              
    st.write("")
    st.write("")
    st.write("")
    st.write("")

   
    
   
 

    
            
           


