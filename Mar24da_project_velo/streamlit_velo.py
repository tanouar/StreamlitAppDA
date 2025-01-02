import streamlit as st
import utils.str_func as common

common.page_config("Introduction")
common.menu()
common.local_css("css_str.css")

if 'df_velo' not in st.session_state:
    common.load_velo()

st.image('assets/logo2.png', caption='Cursus : Data Analyst - Mars 2024 | Lena GUILLOUX, Mélissa CHEMMAMA, Myriam MAHDJOUB, Eléonore HERMAND')

st.markdown("Selon Pierre Breteau dans son article [A Paris, la fréquentation des pistes cyclables a doublé en un an](https://www.lemonde.fr/les-decodeurs/article/2023/11/11/a-paris-la-frequentation-des-pistes-cyclables-a-double-en-un-an_6199510_4355770.html) (*novembre 2023 - Le Monde*), la fréquentation des pistes cyclables a fortement augmenté, jusqu'à doubler entre 2022 et 2023 aux heures de pointe. Une tendance, enclenchée en 2019 lors des grèves de transports en commun puis en 2020 lors du déconfinement, qui ne se dément pas.")
st.write("En effet, la mobilité urbaine est un domaine en constante évolution, notamment dans les grandes métropoles où la réduction du trafic automobile et l'amélioration de la qualité de vie sont devenues des priorités. Dans ce contexte, le vélo émerge comme un mode de transport crucial, favorisé par les politiques publiques et une prise de conscience écologique croissante.")

st.write("A partir de jeux de données disponibles librement sur le site de la [Mairie de Paris](https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/information/?disjunctive.id_compteur&disjunctive.nom_compteur&disjunctive.id&disjunctive.name), notre rapport présente une analyse de l'évolution du trafic vélo à Paris permettant de dégager des tendances.")

st.write("La réalisation de notre projet a été effectuée en plusieurs étapes:")

st.html("<h4>Etape 1 - Exploration des données 🔍:</h4> La première étape de ce travail a été l'exploration des données, où nous avons examiné les jeux de données à notre disposition. Cette étape inclut les modifications et les ajouts de features nécessaires pour atteindre notre dataframe final utilisé pour notre modélisation.")
st.html("<h4>Etape 2 - Data visualisation 📈:</h4> La deuxième étape a consisté en la visualisation des données, où nous avons utilisé des techniques de data visualisation pour représenter graphiquement les tendances du trafic vélo. Cette étape nous a permis de mieux comprendre les schémas et les relations entre les différentes variables, facilitant ainsi l'interprétation des résultats et la communication des conclusions.")
st.html("<h4>Etape 3 - Nettoyage et pre-processing des données 🧹:</h4> Le nettoyage et le pré-traitement des données ont constitué la troisième étape de notre analyse. Cette phase critique a impliqué la correction des anomalies, la gestion des valeurs manquantes et la standardisation des formats de données pour assurer la fiabilité de nos analyses et modélisations ultérieures.")
st.html("<h4>Etape 4 - Modelisation ⚙️:</h4> La quatrième étape a porté sur la modélisation via des techniques de machine learning. Nous avons appliqué plusieurs algorithmes pour prédire le trafic vélo, en tenant compte de différentes variables. Ce modèle vise à fournir une prediction sur le comptage horaire du trafic velo pour un compteur donné se situant dans la ville de Paris.")

st.markdown(
    """
    <style>
    .container {
        background-color: #f0f0f0;  # Couleur de fond gris
        padding: 10px;             # Espacement à l'intérieur de l'encadré
        border-radius: 5px;        # Coins arrondis
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Créer un container pour le texte
with st.container():
    st.markdown('<div class="container"><i>Pour accéder à nos analyses détaillées et interactives (Cartographie dynamique, Data visualisation, Machine learning), cliquez sur le menu à gauche.</i></div>', unsafe_allow_html=True)
  