import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io


title = "Températures terrestres"
sidebar_name = "Exploration des données"


def run():
    
    # st.image("Data/data.jpg", width=400)

    st.header(sidebar_name)
    st.markdown("---")

    st.write('**Approche suivie**')
    st.write("➽ Charger les données brutes depuis chaque fichier csv, et alimenter un dataframe par fichier source.\n\n➽ Afficher les caractéristiques des dataframes.\n\n➽ Rechercher et gérer les données manquantes et aberrantes.\n\n➽ Renommer les variables.\n\n➽ Fusionner les 2 datasets mis en qualité.")

    

    ########### dataset n°1 ########### 
    st.write("\n\n")
    st.write('**1- Dataset n°1**') 
    df=pd.read_csv("Data/owid.csv")

    with st.expander('**Origine des données**'):
        st.write("➽ Le dataset n°1 concerne les émissions annualisées de gaz à effet de serre depuis 1850 par zone géographique, majoritairement des pays, mais des zones plus larges ou plus petites y sont présentent.\n\n➽ Les données brutes sont disponibles aux formats csv, xlsx et json.\n\n➽ Un fichier additionnel au format csv comprend les métadonnées du fichier de données brutes\n\n➽ Ces fichiers sont téléchargeables via un [dépôt GitHub](https://github.com/owid/co2-data/tree/master)")

    # Affichage 5 premières lignes dataset n°1
    with st.expander('**Affichage données brutes**'):
        st.dataframe(df.head())

    with st.expander("**Caractéristiques**"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    with st.expander("**Données manquantes, superflues ou aberrantes**"):
        st.write("\n\n**Axes d'analyse et décision :**\n\n➽ Colonnes ayant un taux de données manquantes trop importantes, sauf si ces données sont nécessaires aux analyses à venir.\n\n➽ Lignes correspondantes à des territoires ou codifications non présentes dans la norme ISO-3166-1.\n\n➽ Données cumulatives de colonnes adjacentes.\n\n **Résultats :** \n\n➽ Réduction de 79 à 18 colonnes.\n\n➽ Réduction de 48 058 à 41 801 lignes.\n\n")
        python_code =  '''#Colonnes à garder dans le fichier
colonnes_a_garder = ['country', 'year', 'iso_code', 'population', 'gdp',
                     'co2', 'co2_per_capita',
                     'methane', 'nitrous_oxide',
                     'cumulative_gas_co2', 'cumulative_oil_co2','cumulative_flaring_co2','cumulative_coal_co2','cumulative_other_co2',
                     'temperature_change_from_ch4', 'temperature_change_from_co2',
                     'temperature_change_from_ghg', 'temperature_change_from_n2o']
df = df[colonnes_a_garder].copy()

#Les pays (lignes) à retirer du fichier
A_retirer = ['Africa (GCP)', 'Asia (GCP)',
       'Asia (excl. China and India)', 'Central America (GCP)',
       'Europe (GCP)', 'Europe (excl. EU-27)', 'Europe (excl. EU-28)',
       'European Union (27)',
       'French Equatorial Africa (Jones et al. 2023)',
       'French West Africa (Jones et al. 2023)', 'High-income countries',
       'International aviation', 'International shipping',
       'International transport', 'Kosovo', 'Kuwaiti Oil Fires (GCP)',
       'Kuwaiti Oil Fires (Jones et al. 2023)',
       'Least developed countries (Jones et al. 2023)',
       'Leeward Islands (GCP)', 'Leeward Islands (Jones et al. 2023)',
       'Low-income countries', 'Lower-middle-income countries',
       'Middle East (GCP)', 'Non-OECD (GCP)',
       'North America (GCP)', 'North America (excl. USA)', 'OECD (GCP)',
       'OECD (Jones et al. 2023)', 'Oceania (GCP)',
       'Panama Canal Zone (GCP)', 'Panama Canal Zone (Jones et al. 2023)',
       'Ryukyu Islands (GCP)', 'Ryukyu Islands (Jones et al. 2023)',
       'South America (GCP)',
       'St. Kitts-Nevis-Anguilla (GCP)',
       'St. Kitts-Nevis-Anguilla (Jones et al. 2023)',
       'Upper-middle-income countries']
df = df.loc[~df.country.isin(A_retirer)]
        '''
        st.code(python_code)

    ########### codebook dataset n°1 ########### 
    st.write("\n\n")
    st.write('**1.1- Codebook du dataset n°1**') 
    # Affichage 5 premières lignes codebook dataset n°1
    with st.expander("**Détails**"):
        df=pd.read_csv("Data/owid-co2-codebook.csv")
        st.dataframe(df.head())


    ########### dataset n°2 ########### 
    st.write("\n\n")

    st.write('**2- Dataset n°2**') 
    df=pd.read_csv("Data/hadcrut-surface-temperature-anomaly.csv")  

    with st.expander('**Origine des données**'):
        st.write("➽ Le dataset n°2 comprend les anomalies de températures annuelles de surface par année par pays depuis 1850. Ces données d'évolution de température sont basées sur les anomalies de température de surface mesurées (ou collectées) par le *Hadley Centre* et le *Climatic Research Unit* de l'Université d'East Anglia. Elles fournissent des informations sur les variations de température à long terme.\n\n➽ Ce fichier de données est téléchargeable sur le site de l'organisation *'Our World in Data'* via [une page dédiée](https://ourworldindata.org/grapher/hadcrut-surface-temperature-anomaly)")


    # Affichage 5 premières lignes dataset n°1
    with st.expander('**Affichage données brutes**'):
        st.dataframe(df.head())

    with st.expander("**Caractéristiques**"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    with st.expander("**Données manquantes, superflues ou aberrantes**"):
        st.write("\n\n**Axes d'analyse :**\n\n➽ Nature des lignes de données sans code alpha-3 ISO-3166-1.\n\n➽ Recherche de codes non standards.\n\n➽ Les seules données manquantes concernent la colonne 'Code',et deux codes absents de la norme ISO-3166-1 sont détectées, correspondant à des ajouts manuels par les propriétaires du dataset\n\n➽ Données superflues.\n\n**Résultats :**\n\n➽ Réduction de 4 à 3 colonnes\n\n➽ Réduction de 29 566 à 29 303 lignes.")
        python_code =  '''# Source : code alpha-3 depuis norme ISO-3166-1
alpha3_iso_codes = ['AFG','ZAF','ALA','ALB','DZA','DEU','AND','AGO','AIA','ATA','ATG','SAU','ARG','ARM','ABW','AUS','AUT','AZE','BHS','BHR','BGD','BRB','BLR','BEL','BLZ',
             'BEN','BMU','BTN','BOL','BES','BIH','BWA','BVT','BRA','BRN','BGR','BFA','BDI','CYM','KHM','CMR','CAN','CPV','CAF','CHL','CHN','CXR','CYP','CCK','COL',
             'COM','COG','COD','COK','KOR','PRK','CRI','CIV','HRV','CUB','CUW','DNK','DJI','DOM','DMA','EGY','SLV','ARE','ECU','ERI','ESP','EST','USA','ETH','FLK',
             'FRO','FJI','FIN','FRA','GAB','GMB','GEO','SGS','GHA','GIB','GRC','GRD','GRL','GLP','GUM','GTM','GGY','GIN','GNB','GNQ','GUY','GUF','HTI','HMD','HND',
             'HKG','HUN','IMN','UMI','VGB','VIR','IND','IDN','IRN','IRQ','IRL','ISL','ISR','ITA','JAM','JPN','JEY','JOR','KAZ','KEN','KGZ','KIR','KWT','LAO','LSO',
             'LVA','LBN','LBR','LBY','LIE','LTU','LUX','MAC','MKD','MDG','MYS','MWI','MDV','MLI','MLT','MNP','MAR','MHL','MTQ','MUS','MRT','MYT','MEX','FSM','MDA',
             'MCO','MNG','MNE','MSR','MOZ','MMR','NAM','NRU','NPL','NIC','NER','NGA','NIU','NFK','NOR','NCL','NZL','IOT','OMN','UGA','UZB','PAK','PLW','PSE','PAN',
             'PNG','PRY','NLD','PER','PHL','PCN','POL','PYF','PRI','PRT','QAT','REU','ROU','GBR','RUS','RWA','ESH','BLM','KNA','SMR','MAF','SXM','SPM','VAT','VCT',
             'SHN','LCA','SLB','WSM','ASM','STP','SEN','SRB','SYC','SLE','SGP','SVK','SVN','SOM','SDN','SSD','LKA','SWE','CHE','SUR','SJM','SWZ','SYR','TJK','TWN',
             'TZA','TCD','CZE','ATF','THA','TLS','TGO','TKL','TON','TTO','TUN','TKM','TCA','TUR','TUV','UKR','URY','VUT','VEN','VNM','WLF','YEM','ZMB','ZWE']

# Intialisation variable
illegal_codes = []
# Recherche des code "ISO"-like fantaisistes du fichier
for value in df2['Code'].unique():
  if value not in alpha3_iso_codes:
    illegal_codes.append(value)

print(illegal_codes)

# Suppression des lignes utilisant ces codes ISO non-standards
df2.drop(df2[df2['Code'] == 'OWID_KOS'].index, inplace = True)
df2.drop(df2[df2['Code'] == 'OWID_CYN'].index, inplace = True)
'''
        st.code(python_code)

        st.write("**Note :** ces données auraient de toute façon étaient exclues de la jointure interne entre les deux dataframes.")

    ########### données nettoyées et mergées dataset 1+2 ########### 
    st.write("\n\n")
    st.write('**3- Données après mise en qualité et fusion des datasets**') 
    with st.expander("**Détail de la fusion des datasets mis en qualité**"):
        st.write("La fusion est réalisée avec l'option inner pour ne garder que les données communes aux 2 datasets.\n\nNous ajoutons la colonne température issue du 2eme dataset.\n\nLa fusion est effectuée via les colonnes 'year' et 'iso_code'.")
        df=pd.read_csv("Data/merged_owid_temp.csv", index_col=0)  
        python_code =  '''
df2.rename(columns={'Code' : 'iso_code', 'Year':'year','Surface temperature anomaly': 'temperature'}, inplace=True)
df = pd.merge(df, df2,  how='inner',  on=['year', 'iso_code'])
'''
        st.code(python_code)

        if st.checkbox("Afficher les informations du dataset ?") :
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

        if st.checkbox("Afficher les statistiques du dataset ?") :
            st.dataframe(df.describe())

        if st.checkbox("Afficher le nombre des données manquantes par colonne du dataset ?") :
            st.dataframe(df.isna().sum())

    st.write("\n\n")

    ########### Ajout codification zones continentale et regroupement pays ########### 

    st.write('**4- Ajout codification zones continentale et regroupement pays**') 
    with st.expander("**Rationnel**"):
        st.write("\n\n**Axes d'analyse :**\n\n➽ L'ensemble des données sont liés aux pays, qui présentent plus de 200 valeurs différentes. Ce nombre est trop élevé pour être utilisé par un modèle de *Machine Learning*.\n\n**Décision :**\n\n➽ Nécessité de regrouper les pays en utilisant les données présentes de codification ISO-3166-1.\n\n➽ Choix de la table M-49 des Nations-Unies prévues à cet effet.")
        
    with st.expander("**Création de la table de regroupement**"):    
        python_code =  '''df_codes = pd.read_csv("./ctpzi.csv", encoding='latin-1')

df_codes.drop(columns=['Num-3', 'Alpha-2','Notes de bas de page'], inplace=True)
df_codes.Code = df_codes.Code // 1000
df_codes["continent"] = df_codes.Code // 10
df_codes = df_codes.dropna()
df_codes.rename(columns={'Alpha-3' : 'iso_code', 'Code':'zone_geo', 'Pays et zones d\'intérêt' : 'pays'}, inplace=True)
'''
        st.code(python_code)
        
        df_codes=pd.read_csv("Data/ctpzi_table.csv")

        st.dataframe(df_codes.drop(columns=['Unnamed: 0'], inplace=True))
        st.dataframe(df_codes.head())

    with st.expander("**Fusion de la table de regroupement avec le dataset mis en qualité**"):  
        python_code =  '''df_ML = df_codes.merge(df,  how='inner',  on=['iso_code'])  # du fait du "inner", on perd les données des continents et du monde
df_ML.drop(columns=['country','cumulative_gas_co2', 'cumulative_oil_co2','cumulative_flaring_co2','cumulative_coal_co2','cumulative_other_co2'], inplace=True)
df_ML = df_ML.loc[df_ML.year>=1970]
df_ML.zone_geo =df_ML.zone_geo.astype(int)
df_ML.continent =df_ML.continent.astype(int)
'''
        st.code(python_code)

        df=pd.read_csv("Data/merged_owid_temp_zones.csv")

        st.dataframe(df.drop(columns=['Unnamed: 0'], inplace=True))

        st.dataframe(df.head())


        if st.checkbox("Afficher les informations de ce dataset ?") :
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

        if st.checkbox("Afficher les statistiques de ce dataset ?") :
            st.dataframe(df.describe())

        if st.checkbox("Afficher le nombre des données manquantes par colonne de ce dataset ?") :
            st.dataframe(df.isna().sum())

    st.write("\n\n")
    st.write("\n\n")
    st.write("\n\n")





