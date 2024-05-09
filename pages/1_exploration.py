import streamlit as st
import utils.str_func as common

common.page_config("Données")
common.local_css("css_str.css")
common.menu()

if 'df_velo' not in st.session_state:
    common.load_velo()

df_velo = st.session_state.df_velo

st.html("<h1>Exploration des données</h1>")
col1, col2, col3 = st.columns((0.7,3,1))
col1.image("assets/paris_data.png")
col3.image("assets/meteo_gouv_data.png")

st.markdown("")
st.markdown("#### Jeux de données utilisés :")
st.write("Afin d'atteindre les objectifs de notre projet, nous avons utilisé deux jeux de données, disponibles sur [Paris Data](https://opendata.paris.fr/pages/home/) et [Meteo Data](https://meteo.data.gouv.fr/datasets) :")
st.markdown("""
            - Un jeu de données sur le **comptage de vélos** couvrant la période 2022 et 2023.
            - Un jeu de données traitant de la **météo** couvrant les périodes 2022 et 2023.
            """)

with st.expander("Jeu de données numéro 1 : Comptage vélo 2022 et 2023"):
    st.write("**Extrait du dataframe (5ères lignes):**")
    st.image('assets/dfhead5.png')

    
    colcompt1, colcompt2, colcompt3 = st.columns((1,1,1))
    colcompt1.markdown(f"""
    **Informations générales et statistiques :**
    <ul>
        <li> Nombre de lignes: <code>{2331870}</code></li>
        <li> Nombre de colonnes: <code>{16}</code></li>
    </ul>
    """, unsafe_allow_html=True)
    colcompt1.image('assets/dfstats.png')
    colcompt2.markdown("**Types des colonnes :**")
    colcompt2.image('assets/dfinfo.png')

    colcompt3.markdown("**Valeurs manquantes par colonnes :**")
    colcompt3.image('assets/dfvaleursmanquantes.png')

with st.expander("Jeu de données numéro 2 : Météo 2022 et 2023"):
    st.write("**Extrait du dataframe (5ères lignes):**")
    st.image('assets/meteohead.png')
    
    colmeteo1, colmeteo2, colmeteo3 = st.columns((1,1,1))
    colmeteo1.markdown(f"""
    **Informations générales et statistiques :**
    <ul>
        <li> Nombre de lignes: <code>{222431}</code></li>
        <li> Nombre de colonnes: <code>{7}</code></li>
    </ul>
    """, unsafe_allow_html=True)
    colmeteo1.image('assets/meteostats.png')
    
    colmeteo2.markdown("**Types des colonnes :**")
    colmeteo2.image('assets/meteoinfo.png')

    colmeteo3.markdown("**Valeurs manquantes par colonnes :**")
    colmeteo3.image('assets/meteomanquantes.png')



st.markdown("#### Nettoyage et modifications effectuées sur nos dataframe :")
st.write("Différentes tâches de nettoyage ont été nécessaires avant de débuter nos travaux de modélisation, avec des modifications et ajouts de colonnes. Le détail est indiqué ci-dessous, avec pour commencer les étapes effectuées sur le dataframe contenant le comptage horaire des passages de vélos, puis le dataframe météo.")

st.markdown("##### Focus sur le dataframe Comptage Velo :")
st.markdown("**Nettoyage du dataframe :**")
st.markdown(f"""
    <ul>
        <li> Conversion de la colonne <span class="colonne">Date et heure de comptage</span> en format adapté,</li>
        <li> Séparation de la colonne <span class="colonne">Coordonnées géographiques</span> en lattitude et longitude,</li>
        <li> Harmonisation de la nomenclature des colonnes,</li>
        <li> Suppression de colonnes non nécessaires,</li>
        <li> Gestion des N/A sur <span class="colonne">Nom du site de comptages</span>, sur <span class="colonne">Lat</span> et sur <span class="colonne">Lon</span> (ne concerne que 2 compteurs au même site de comptage),</li>
        <li> Retrait des compteurs qui n'obtiennent aucun comptage sur les 2 années 2022-2023 à partir des colonnes <span class="colonne">Identifiant du compteur</span> et <span class="colonne">Comptage horaire</span>. </li>
    </ul>
    """, unsafe_allow_html=True)
st.markdown("**Ajout de colonnes :**")
st.markdown(f"""
    <ul>
        <li> Ajout de différente variables à partir de la colonne <span class="colonne">Date et heure de comptage : année, mois, jour de semaine, heure, week-end, vacances scolaires, etc,</span> </li>
        <li> Ajout de différente variables à partir de la colonne <span class="colonne">Heure</span> créée précedemment: matin, après-midi, soir, nuit.</li>
    </ul>
    """, unsafe_allow_html=True)



st.markdown("##### Focus sur le dataframe Meteo :")
st.markdown("**Nettoyage du dataframe :**")
st.markdown(f"""
    <ul>
        <li> Conversion de la colonne <span class="colonne">AAAAMMJJHH</span> en format adapté,</li>
        <li> Suppression de la station 75114007 (en doublon, postée au même endroit, avec 75114001 mais sans les variables pluie 'RR1') à partir de la colonne <span class="colonne">NUM_POSTE</span>, </li>
        <li> Harmonisation de la nomenclature des colonnes.</li>
    </ul>
    """, unsafe_allow_html=True)
st.markdown("**Assignation des labels selon l'intensité de la pluie à partir de la colonne RR1**")


st.markdown("#### Dataframe final :")
st.write("Le dataframe final correspond à une consolidation des deux jeux de de données évoqués précedemment après leur nettoyage.")

with st.expander("Jeu de données final : Dataframe final"):
    st.write("Pour commencer, voici un extrait du dataframe (correspondant au 5 premieres lignes du dataframe) :")
    st.dataframe(df_velo.head(5))
    st.markdown(f"""
    **Informations générales et statistiques :**
    <ul>
        <li> Nombre de lignes: <code>{df_velo.shape[0]}</code></li>
        <li> Nombre de colonnes: <code>{df_velo.shape[1]}</code></li>
    </ul>
    """, unsafe_allow_html=True)
    st.dataframe(df_velo.describe())

    colfin1, colfin2, colfin3 = st.columns((1.2,1,1))
    colfin1.markdown("**Types de colonnes :**")
    colfin1.image('assets/dffinalinfo.png')

    colfin2.markdown("**Colonnes ayant des valeurs manquantes :**")
    missing_values = df_velo.isna().sum()
    colfin2.dataframe(missing_values, use_container_width=True)

st.markdown("#### Résultat de l’exploration des données :")
st.markdown("""
        Les variables qui serviront principalement à mener à bien notre analyse sont le **comptage horaire** (notre variable cible), la **date** et **heure du comptage** et les **ID compteurs**.\n
        Une particularité liée à notre jeu de données est que certains compteurs sont arrêtés en cours d’année (notamment pour cause de travaux) ou ajoutés. Ceci pourrait potentiellement avoir un impact sur nos analyses.
           """)
