import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import plotly.express as px
from io import StringIO
from scipy.stats import normaltest
from scipy.stats import norm
import pickle
import requests
from io import BytesIO
from PIL import Image



df_2021 = pd.read_csv("world-happiness-report-2021.csv", sep = ",")
df_all = pd.read_csv("world-happiness-report.csv", sep =",")
df_2021_bis = pd.read_csv("world-happiness-report-2021.csv", sep = ",")


print(df_all.head())
print(df_2021.head())

#Dimension
print("Il y a", len(df_all["Country name"].unique()), "pays différents dans df_all")
print("Il y a", len(df_2021["Country name"].unique()), "pays différents dans df_2021")
# suppression des colonnes inutiles

Explainedby = df_2021[["Explained by: Log GDP per capita", "Explained by: Social support",
                       "Explained by: Healthy life expectancy", "Explained by: Freedom to make life choices",
                       "Explained by: Generosity", "Explained by: Perceptions of corruption", "Dystopia + residual"]]



df_2021.head()
# création des nouvelles colonnes dans df_2021
df_2021['year'] = 2021
df_2021['Positive affect'] = np.nan
df_2021['Negative affect'] = np.nan
df_2021.head()

# Groupement par région et liste des pays pour chaque région
pays_par_region = df_2021.groupby('Regional indicator')['Country name'].unique()

# Afficher la liste des pays pour chaque région
for region, countries in pays_par_region.items():
    print(f"Région: {region}")
    print(f"Pays: {', '.join(countries)}\n")

# Creation map region
region_mapping = {
    'Central and Eastern Europe': ['Albania', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Czech Republic', 'Estonia', 'Hungary', 'Kosovo', 'Latvia', 'Lithuania', 'Montenegro', 'North Macedonia', 'Poland', 'Romania', 'Serbia', 'Slovakia', 'Slovenia'],
    'Commonwealth of Independent States': ['Armenia', 'Azerbaijan', 'Belarus', 'Georgia', 'Kazakhstan', 'Kyrgyzstan', 'Moldova', 'Russia', 'Tajikistan', 'Turkmenistan', 'Ukraine', 'Uzbekistan'],
    'East Asia': ['China', 'Hong Kong S.A.R. of China', 'Japan', 'Mongolia', 'South Korea', 'Taiwan Province of China'],
    'Latin America and Caribbean': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Costa Rica', 'Dominican Republic', 'Ecuador', 'El Salvador', 'Guatemala', 'Haiti', 'Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela'],
    'Middle East and North Africa': ['Algeria', 'Bahrain', 'Egypt', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Kuwait', 'Lebanon', 'Libya', 'Morocco', 'Palestinian Territories', 'Saudi Arabia', 'Tunisia', 'Turkey', 'United Arab Emirates', 'Yemen'],
    'North America and ANZ': ['Australia', 'Canada', 'New Zealand', 'United States'],
    'South Asia': ['Afghanistan', 'Bangladesh', 'India', 'Maldives', 'Nepal', 'Pakistan', 'Sri Lanka'],
    'Southeast Asia': ['Cambodia', 'Indonesia', 'Laos', 'Malaysia', 'Myanmar', 'Philippines', 'Singapore', 'Thailand', 'Vietnam'],
    'Sub-Saharan Africa': ['Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon', 'Chad', 'Comoros', 'Congo (Brazzaville)', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'Senegal', 'Sierra Leone', 'South Africa', 'Swaziland', 'Tanzania', 'Togo', 'Uganda', 'Zambia', 'Zimbabwe'],
    'Western Europe': ['Austria', 'Belgium', 'Cyprus', 'Denmark', 'Finland', 'France', 'Germany', 'Greece', 'Iceland', 'Ireland', 'Italy', 'Luxembourg', 'Malta', 'Netherlands', 'North Cyprus', 'Norway', 'Portugal', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom']
}

# Creation de la colonne Regional indicator pour df_all
df_all['Regional indicator'] = df_all['Country name'].map({country: region for region, countries in region_mapping.items() for country in countries})

# Reorganisation des colonnes de 2021
df_2021 = df_2021[['Country name', 'Regional indicator', 'year', 'Ladder score', 'Logged GDP per capita',
                   'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity',
                   'Perceptions of corruption', "Positive affect", "Negative affect"]]

# Reorganisation des colonnes de df_all
df_all = df_all[['Country name', 'Regional indicator', 'year', 'Life Ladder', 'Log GDP per capita',
                 'Social support', 'Healthy life expectancy at birth', 'Freedom to make life choices', 'Generosity',
                 'Perceptions of corruption', "Positive affect", "Negative affect"]]

# Renommer les colonnes pour df_2021
df_2021 = df_2021.rename(columns={
    'Country name': 'Country',
    'Regional indicator' : 'Region',
    'year': 'Year',
    'Ladder score': 'Ladder_score',
    'Logged GDP per capita': 'PIB_habitant',
    'Social support': 'Social_support',
    'Healthy life expectancy': 'Healthy_life_expectancy',
    'Freedom to make life choices': 'Freedom',
    'Generosity': 'Generosity',
    'Perceptions of corruption': 'Corruption',
    'Positive affect': 'Positive_affect',
    'Negative affect': 'Negative_affect'
})

# Renommer les colonnes pour df_all
df_all = df_all.rename(columns={
    'Country name': 'Country',
    'Regional indicator' : 'Region',
    'year': 'Year',
    'Life Ladder': 'Ladder_score',
    'Log GDP per capita': 'PIB_habitant',
    'Social support': 'Social_support',
    'Healthy life expectancy at birth': 'Healthy_life_expectancy',
    'Freedom to make life choices': 'Freedom',
    'Generosity': 'Generosity',
    'Perceptions of corruption': 'Corruption',
    'Positive affect': 'Positive_affect',
    'Negative affect': 'Negative_affect'
})

# Concaténer les deux DataFrames par ligne
df = pd.concat([df_2021, df_all], ignore_index=True)

# Trier les données par pays et par année
df = df.sort_values(by=['Country', 'Year'])

#on supprime les pays sans région
df = df.dropna(subset = ['Region'])

df.head()
df = df.reset_index(drop = True)
df.head()


#On test la normalité des échnatillons
stat, p_value = normaltest(df["Ladder_score"])

    # Interpréter les résultats
distribution_normale = "Oui" if p_value >= 0.05 else "Non"

resultats_tests_normalite_Ladder = []

    # Stocker les résultats dans la liste
resultats_tests_normalite_Ladder.append({
        'Colonne': "Ladder score",
        'Statistique de test': round(stat, 2),
        'p-value': p_value,
        'Distribution normale': distribution_normale
        })

# Créer un DataFrame à partir des résultats
df_resultats_normalite_Ladder = pd.DataFrame(resultats_tests_normalite_Ladder)
df_resultats_normalite_Ladder = df_resultats_normalite_Ladder.set_index(df_resultats_normalite_Ladder.columns[0])


# Configuration de la barre latérale
st.sidebar.title("Sommaire")
pages=["👋 Intro", "🔍 Exploration des données", "📊 Data Visualisation", "🧩 Modélisation", "🔮 Prédiction", "📌Conclusion"]
page=st.sidebar.radio("Aller vers", pages)
st.sidebar.markdown(
    """
    - **Cursus** : Data Analyst
    - **Formation** : Bootcamp
    - **Mois** : Nov.2023
    - **Groupe** : 
        - Baptiste DENOPCE
        - Danie FANANTENANA
        - Gilles TCHAPDA
        - Valentine FALCHERO""")

# Page d'introduction
if page == pages[0] :
    image_cover = "https://zupimages.net/up/24/03/ucks.png"
    st.image(image_cover, use_column_width=True)

    # Présentation projet
    st.caption("""**Cursus** : Data Analyst
    | **Formation** : Bootcamp
    | **Mois** : Nov.2023
    | **Groupe** : Baptiste DENOPCE, Danie FANANTENANA, Gilles TCHAPDA, Valentine FALCHERO
""")


    st.header("👋 Intro")
    st.markdown("""<style>h1 {color: #4629dd;  font-size: 70px;/* Changez la couleur du titre h1 ici */} h2 {color: #440154ff;    font-size: 50px /* Changez la couleur du titre h2 ici */} h3{color: #27dce0; font-size: 30px; /* Changez la couleur du titre h3 ici */}</style>""",unsafe_allow_html=True)
    st.markdown("""<style>body {background-color: #f4f4f4;</style>""",unsafe_allow_html=True)

    st.write("Nous vous proposons ici de découvrir ce qui rend les gens heureux et pouvoir prédire leur niveau de satisfaction. En d'autres termes, nous pouvons aujourd'hui vous livrer les clés du bonheur !")

    # Titre de la section du questionnaire
    st.write("**😁 Cela vous intéresse ?**")
    if st.checkbox("Oui, évidemment") :
        st.subheader("Pour en savoir plus")
        st.write("L'étude menée par le Gallup World Poll pour alimenter les **World Happiness Report** annuels estime le niveau de bonheur (note sur 10 appelée Ladder Score) des habitants d'un pays chaque année à partir de plusieurs indicateurs.")

        # Question sur ce qui fait le bonheur sur terre
        avis_bonheur = st.text_area("Quels peuvent être ces indicateurs selon vous ?")
        
        
        # Titre de la section en savoir plus
        if st.button("Je valide") :
            st.write("Merci pour votre réponse.")
            st.write("Pour cette étude, l'évaluation du Ladder Score s'est portée selon les variables suivantes :")
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                st.write("- le PIB par habitant,")
                st.write("- le soutien social,")
                st.write("- l'espérance de vie en bonne santé,")
            with col2:
                st.write("- la liberté de choisir,")
                st.write("- la générosité,")
                st.write("- la corruption,")
            with col3:
                st.write("- l'affect positif,")
                st.write("- l'affect négatif.")

            st.write("")
            st.write("Observons davantage les données d'origine de notre projet.")
            

# Page d'exploration des données
if page == pages[1] : 
    st.header("🔍 Exploration des Données")
    st.markdown("""<style>h1 {color: #4629dd;  font-size: 70px;/* Changez la couleur du titre h1 ici */} h2 {color: #440154ff;    font-size: 50px /* Changez la couleur du titre h2 ici */} h3{color: #27dce0; font-size: 30px; /* Changez la couleur du titre h3 ici */}</style>""",unsafe_allow_html=True)
    st.markdown("""<style>body {background-color: #f4f4f4;</style>""",unsafe_allow_html=True)


    #Afficher le Dataset df_all
    st.subheader("Dataset sur le bonheur dans le monde entre 2005 et 2020")
    st.write("**Voici les premières lignes de ce jeu de données:**")
    st.dataframe(df_all.head())
    st.write("**Informations principales sur ce jeu de données:**")
    st.write("- Nombre de lignes:", df_all.shape[0])
    st.write("- Nombre de colonnes:", df_all.shape[1])
    st.write("- Résumé statistique de tout le jeu de données :")
    st.dataframe(df_all.describe())
    st.write("")
    col1, col2 = st.columns([1.75,3])
    with col1 : 
        st.write("- Valeurs manquantes :", df_all.isnull().sum())

    with col2:
        st.write("- Informations :")
        info_str_io = StringIO()
        df_all.info(buf=info_str_io)
        info_str = info_str_io.getvalue()    
        st.text(info_str)

    st.write("")

    if st.button("Passer à 2021") :
        #Afficher le Dataset df_2021
        st.subheader("Dataset sur le bonheur dans le monde sur 2021")
        st.write("**Voici les premières lignes de ce jeu de données:**")
        st.dataframe(df_2021_bis.head())
        st.write("**Informations principales sur ce jeu de données:**")
        st.write("- Nombre de lignes:", df_2021_bis.shape[0])
        st.write("- Nombre de colonnes:", df_2021_bis.shape[1])
        st.write("- Résumé statistique de tout le jeu de données :")
        st.dataframe(df_2021_bis.describe())
        st.write("")
        col1, col2 = st.columns([1.75,3])
        with col1 : 
            st.write("- Valeurs manquantes :", df_2021_bis.isnull().sum())

        with col2:
            st.write("- Informations :")
            info_str_io = StringIO()
            df_2021.info(buf=info_str_io)
            info_str = info_str_io.getvalue()    
            st.text(info_str)


    #Afficher le Dataset df
    if st.button("Passer au dataframe concaténé") :
        st.write("Afin de continuer notre étude pour déterminer les clés du bonheur, nous avons réaliser des modifications sur ces 2 datasets pour ainsi obtenir un fichier exploitable et pertinent.")
        col4, col5 = st.columns([1,1])
        with col4:
            st.write("_Pour la partie Visualisation des données_")
            st.write("- Gestions des valeurs manquantes (suppression ou remplacement des données).")
        with col5:
            st.write("_Pour les parties Modélisation et Prédictions_")
            st.write("- Suppression de colonnes non nécessaires,")
            st.write("- Ajout de colonnes manquantes dans les dataframes selon pertinence,")
            st.write("- Harmonisation de la nomenclature des colonnes.")
        
        st.write("")
        st.write("")

        #Afficher le Dataset df
        st.subheader("Concaténation des datasets sur le bonheur dans le monde de 2005 à 2021")
        st.write("**Voici les premières lignes de ce jeu de données:**")

        st.dataframe(df.head())
        
        st.write("")
        st.write("")

        st.markdown("**Informations principales sur ce jeu de données:**")
        st.write("- Nombre de lignes:", df.shape[0])
        st.write("- Nombre de colonnes:", df.shape[1])
        st.write("- Résumé statistique de tout le jeu de données :")
        st.dataframe(df.describe())
        st.write("")
        col1, col2 = st.columns([1.75,3])
        with col1 : 
            st.write("- Valeurs manquantes :", df.isnull().sum())

        with col2:
            st.write("- Informations :")
            info_str_io = StringIO()
            df.info(buf=info_str_io)
            info_str = info_str_io.getvalue()    
            st.text(info_str)



if page == pages[2] :
    st.markdown("""<style>h1 {color: #4629dd;  font-size: 70px;/* Changez la couleur du titre h1 ici */} h2 {color: #440154ff;    font-size: 50px /* Changez la couleur du titre h2 ici */} h3{color: #27dce0; font-size: 30px; /* Changez la couleur du titre h3 ici */}</style>""",unsafe_allow_html=True)
    st.markdown("""<style>body {background-color: #f4f4f4;</style>""",unsafe_allow_html=True)

    st.header("📊 Data Visualisation")
    st.subheader("1. Distribution et loi normale de Ladder Score")

    ladder_scores = df["Ladder_score"]
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=ladder_scores, nbinsx=20, histnorm='probability', marker_color='#355F8D'))

    # Ajouter la courbe de densité normale
    mu, sigma = ladder_scores.mean(), ladder_scores.std()
    x = np.linspace(min(ladder_scores), max(ladder_scores), 100)
    y = norm.pdf(x, mu, sigma)

    hist_fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Densité normale', line=dict(color='green', width=2)))
    hist_fig.update_layout(
            xaxis_title='Ladder Score',
            yaxis_title='Densité de probabilité',
            showlegend=False,
        )
    hist_fig.update_layout(
        xaxis=dict(tickfont=dict(size=16), title_font=dict(size=16)),
        yaxis=dict(tickfont=dict(size=16), title_font=dict(size=16))
        )
    st.plotly_chart(hist_fig)

    if st.checkbox("Afficher statistique de la normalité") :
        st.dataframe(df_resultats_normalite_Ladder)
        st.write("La p-value est exactement de : 2.29e-22")

    st.write("")
    st.write("")
    st.write("")

    if st.button("Le Score du bonheur dans le monde") :

        st.subheader("2. Le Score du bonheur dans le monde")
        #Instanciation du graph
        fig = go.Figure(data=go.Choropleth(
            locations=df['Country'],
            locationmode='country names',
            z=df['Ladder_score'],
            text=df['Country'],
            colorbar={'title': 'Happiness'},
            colorscale='Viridis'
        ))
        #Caractéristiques du graph
        fig.update_layout(
            geo=dict(
                showframe=False,
                showocean=False,
                showlakes=True,
                showcoastlines=True,
                projection={'type': 'natural earth'}
            ),
            height=600
        )
        #Afficher le graph
        st.plotly_chart(fig)

        #if st.checkbox("N°3"):
        st.markdown("""
                    ##### Points à retenir :
                    - Ladder score de chaque pays plutôt similaire au sein d'une même région
                    - Différence de valeurs entre les pays développés et ceux en voie de développement
                    """)
        
        st.write("")
        st.write("")
        st.write("")

    if st.button("Etude graphique du Ladder Score par région") :

        st.subheader("3. Etude graphique du Ladder Score par région")

        #Grouper et trier par mediane
        region_order = df.groupby("Region")["Ladder_score"].median().sort_values(ascending=False).index

        # Créer une palette de couleurs Viridis
        viridis_colors = px.colors.sequential.Viridis[:len(region_order)]

        # Créer un boxplot trié par mediane
        fig = px.box(df, y='Region', x='Ladder_score', category_orders={'Region': region_order},
                    orientation='h', color='Region', hover_name=df["Country"],
                    color_discrete_map=dict(zip(region_order, viridis_colors)),
                    title="Boxplot du Ladder Score par région",
                    labels={'Ladder_score': 'Ladder Score', 'Region': 'Région'},
                    width=800, height=400)
        #Caractéristiques du graph
        fig.update_layout(
            showlegend=False,
            xaxis=dict(tickfont=dict(size=14), title_font=dict(size=14)),
            yaxis=dict(tickfont=dict(size=14), title_font=dict(size=14)),
            title_font=dict(size=20)
        )
        #Afficher le graph
        st.plotly_chart(fig)


        # Créer un barplot du "Ladder score" median par région
        #Grouper par mediane puis trier
        region_order = df.groupby(["Region"], as_index = False)["Ladder_score"].median()
        region_order_sorted = region_order.sort_values(by='Ladder_score', ascending=False)

        col1, col2 = st.columns([3,1])
        
        #Instanciation du graph dans col1
        with col1:
            fig = px.bar(region_order_sorted, x='Ladder_score', y='Region', color='Region',
                        color_discrete_map=dict(zip(region_order_sorted['Region'].unique(), viridis_colors)),
                        title='Mediane des Ladder Scores par Région',
                        labels={'Ladder_score': 'Ladder Score (Mediane)', 'Region': 'Région'},
                        width=600, height=450)
            #Caractéristiques du graph
            fig.update_layout(
                showlegend=False,
                xaxis=dict(tickfont=dict(size=14), title_font=dict(size=14)),
                yaxis=dict(tickfont=dict(size=14), title_font=dict(size=14)),
                title_font=dict(size=20)
            )
            #Afficher le graph
            st.plotly_chart(fig)


        # Créer un barplot du "Ladder score" moyen par région
        #Grouper par moyenne puis trier
        ladder_mean = df.groupby(["Region"], as_index=False)["Ladder_score"].mean()
        ladder_mean_sorted = ladder_mean.sort_values(by='Ladder_score', ascending=False)

        #Instanciation du graph dans col2
        with col2:
            fig = px.bar(ladder_mean_sorted, x='Ladder_score', y='Region', color="Region",
                        color_discrete_map=dict(zip(ladder_mean_sorted['Region'].unique(), viridis_colors)),
                        title='Moyenne des Ladder Scores par Région',
                        labels={'Ladder_score': 'Ladder Score (Moyenne)'},
                        width =600, height = 450)
            #Caractéristiques du graph
            fig.update_layout(
                showlegend=False,
                xaxis=dict(tickfont=dict(size=14), title_font=dict(size=14)),
                yaxis=dict(title='', tickfont=dict(size=14), title_font=dict(size=14)),
                title_font=dict(size=20)
            )
            #Afficher le graph
            st.plotly_chart(fig)

        st.write("")

        #if st.checkbox("N°2"):
        st.markdown("""
                    ##### Points à retenir :
                    - Représentation assez homogène du Ladder Score au sein des régions
                    - Valeurs du Ladder Score différentes d'une région à l'autre
                    - Classement du Ladder Score des régions similaires en moyenne et médiane
                    """)

        st.write("")
        st.write("")
        st.write("")

    if st.button("Evolution du Ladder Score au fil des années") :

        st.subheader("4. Evolution du Ladder Score au fil des années selon la région du monde")

        #Instanciation de la couleur viridis
        viridis_colors = px.colors.sequential.Viridis[:len(df['Region'].unique())] 
    
        #Avoir le mediane
        region_order = df.groupby(["Region"], as_index = False)["Ladder_score"].median()
        region_order_sorted = region_order.sort_values(by='Ladder_score', ascending=False)

        #Avoir la moyenne
        ladder_mean_year = df.groupby(["Year", "Region"], as_index=False)["Ladder_score"].mean()

        #Instanciation du graph
        fig = px.line(ladder_mean_year, x='Year', y='Ladder_score', color='Region',
                    labels={'Ladder_score': 'Ladder Score', 'Year': 'Année', 'Region': 'Région'},
                    color_discrete_map=dict(zip(region_order_sorted['Region'].unique(), viridis_colors))
                    )
        #Caractéristiques du graph
        fig.update_layout(
            width=900, height=600,
            xaxis=dict(tickfont=dict(size=16), title_font=dict(size=16)),
            yaxis=dict(tickfont=dict(size=16), title_font=dict(size=16)),
            legend_title=dict(font=dict(size=16)),
            legend_font=dict(size=14)
        )
        #Afficher le graph
        st.plotly_chart(fig)

        st.write("")

        #if st.checkbox("N°4"):
        st.markdown("""
                    ##### Point à retenir :
                    - Insensibilité relative du Ladder Score dans le temps
                    """)
            
        st.write("")
        st.write("")
        st.write("")

    if st.button("Heatmap des corrélations") :

        #Graph de la heatmap
        st.subheader("5. Heatmap des corrélations")

        heatmap_data = df[["Ladder_score", "PIB_habitant", "Social_support", "Healthy_life_expectancy", "Freedom", "Generosity", "Corruption", "Positive_affect", "Negative_affect"]]

        cor = heatmap_data.corr()  #Calcul des correlations
        
        #Instanciation du graphique
        fig = px.imshow(cor, labels=dict(x="Variables", y="Variables", color="Correlation"), x=cor.index, y=cor.columns,
                    color_continuous_scale="viridis", width =900, height = 700)

        fig.update_layout(
            xaxis=dict(tickfont=dict(size=16), title_font=dict(size=16)),
            yaxis=dict(tickfont=dict(size=16), title_font=dict(size=16))
        )
        # Ajout des valeurs dans les cases
        for i in range(len(cor)):
            for j in range(len(cor.columns)):
                fig.add_annotation(
                    x=i,
                    y=j,
                    text=f"{cor.iloc[i, j]:.2f}",
                    showarrow=False,
                    font=dict(size=14),
                    xanchor='center',
                    yanchor='middle',  # Correction ici
                )

        # Afficher le graphique
        st.plotly_chart(fig)

        st.write("")
        
        #if st.checkbox("N°7"):
        st.markdown("""
                    ##### Point à retenir :
                    - Corrélation forte avec le Ladder Score pour le PIB / Habitant, l'espérance de vie en bonne santé et le soutien social
                    """) 
        
        st.write("")
        st.write("")
        st.write("")

    if st.button("Importance des variables") :

        st.subheader("6. Importance des variables dans le Ladder Score en 2021")

        #Utiliser la moyenne de chaque colonne contenu dans Explainedby
        x = Explainedby.mean()
        
        #Instanciation du graphique
        fig_1 = go.Figure()
        fig_1.add_trace(go.Pie(labels = x.index,
                            values = x,
                            marker_line = dict(color = 'black', width = 1.5), # Couleur et épaisseur de la ligne 
                            marker_colors =["#440154","#482475","#414487","#355F8D","#2A788E","#21918C","#22A784"], 
                            pull = [0,0,0,0,0,0,0] # Partie à éloigner du camembert 
                            ))
        #Caractéristiques du graph
        fig_1.update_layout(
            width=800, height=600,
            xaxis=dict(tickfont=dict(size=16), title_font=dict(size=16)),
            yaxis=dict(tickfont=dict(size=16), title_font=dict(size=16)),
            legend_title=dict(font=dict(size=16)),
            legend_font=dict(size=14)
        )
        #Afficher le graph
        st.plotly_chart(fig_1)    

        st.write("")

        #if st.checkbox("N°5"):
        st.markdown("""
                    ##### Points à retenir :
                    - Forte influence du PIB par habitant et du Soutien Social pour expliquer le Ladder Score, outre la valeur _'Dystopia + résidual'_
                    - Faible influence pour la générosité ou la perception de la corruption
                    """)
        
        st.write("")
        st.write("")
        st.write("")

    if st.button("Relation des variables") :

        st.subheader("7. Relation des variables avec le Ladder Score")

        #Instanciation de la couleur viridis
        viridis_colors = px.colors.sequential.Viridis[:len(df['Region'].unique())] 
        
        # Graphique 1
        fig = px.scatter(df, x="PIB_habitant", y="Ladder_score",  color="Region", hover_name=df["Country"], size_max=20, 
                        color_discrete_map=dict(zip(df['Region'].unique(), viridis_colors)))
        #Caractéristiques du graph
        fig.update_layout(
            width=800, height=600,
            xaxis=dict(tickfont=dict(size=16), title_font=dict(size=16)),
            yaxis=dict(tickfont=dict(size=16), title_font=dict(size=16)),
            legend_title=dict(font=dict(size=16)),
            legend_font=dict(size=14)
        )
        #Afficher graph
        st.plotly_chart(fig)

        # Graphique 2
        fig = px.scatter(df, x="Social_support", y="Ladder_score", color="Region", hover_name=df["Country"], size_max=20, 
                        color_discrete_map=dict(zip(df['Region'].unique(), viridis_colors)))
        #Caractéristiques du graph
        fig.update_layout(
            width=800, height=600,
            xaxis=dict(tickfont=dict(size=16), title_font=dict(size=16)),
            yaxis=dict(tickfont=dict(size=16), title_font=dict(size=16)),
            legend_title=dict(font=dict(size=16)),
            legend_font=dict(size=14)
        )
        #Afficher le graph
        st.plotly_chart(fig)

        # Graphique 3
        fig = px.scatter(df, x="Healthy_life_expectancy", y="Ladder_score", color="Region", hover_name=df["Country"], size_max=20, 
                        color_discrete_map=dict(zip(df['Region'].unique(), viridis_colors)))
        #Caractéristiques du graph
        fig.update_layout(
            width=800, height=600,
            xaxis=dict(tickfont=dict(size=16), title_font=dict(size=16)),
            yaxis=dict(tickfont=dict(size=16), title_font=dict(size=16)),
            legend_title=dict(font=dict(size=16)),
            legend_font=dict(size=14)
        )
        #Afficher le graph
        st.plotly_chart(fig)

        
        # Graphique 5
        fig = px.scatter(df, x="Generosity", y="Ladder_score", color="Region", hover_name=df["Country"], size_max=20, 
                        color_discrete_map=dict(zip(df['Region'].unique(), viridis_colors)))
        
        #Caractéristiques du graph
        fig.update_layout(
            width=800, height=600,
            xaxis=dict(tickfont=dict(size=16), title_font=dict(size=16)),
            yaxis=dict(tickfont=dict(size=16), title_font=dict(size=16)),
            legend_title=dict(font=dict(size=16)),
            legend_font=dict(size=14)
        )
        st.plotly_chart(fig)


        st.write("")
        

        #if st.checkbox("N°6"):
        st.markdown("""
                    ##### Points à retenir :
                    - Corrélation linéaire avec le Ladder Score pour le PIB/Habitant, l'espérance de vie en bonne santé et le soutien social avec une pente positive plus ou moins forte
                    - Clusters de régions bien distincts dans le scatterplot de l’espérance de vie en bonne santé
                    """)

        st.write("")
        st.write("")
        st.write("")


        st.write("##### La modélisation nous permettra de mettre en évidence cela ")

if page == pages[3]:
    st.header("🧩 Modélisation")
    st.markdown("""<style>h1 {color: #4629dd;  font-size: 70px;/* Changez la couleur du titre h1 ici */} h2 {color: #440154ff;    font-size: 50px /* Changez la couleur du titre h2 ici */} h3{color: #27dce0; font-size: 30px; /* Changez la couleur du titre h3 ici */}</style>""",unsafe_allow_html=True)
    st.markdown("""<style>body {background-color: #f4f4f4;</style>""",unsafe_allow_html=True)

    
    st.subheader("Objectif")
    st.write("Prédire le Ladder-score moyen d'une région (variable continue) en fonction des données contenues dans les variables explicatives qui le composent.")
    
    st.write("")
    st.write("")

    if st.button("Modèles de régression") :
        st.subheader("Choix des modèles")
        st.markdown("""
                    Afin d'adresser la prédiction du bien-être, nous avons étudié la performance de plusieurs modèles de machine learning :
                    - Régression linéaire
                    - Decision Tree Regressor
                    - Random Forest Regressor
                    - Ridge
                    - LASSO
                    - ElasticNet
                    - Gradient Boosting Regressor.
        """)

        st.write("")
        st.write("")

        st.subheader("Exécution des modèles")
        st.markdown("""
                    Pour chaque modèle appliqué, nous avons suivi les étapes suivantes :
                    1. Instanciation du modèle
                    2. Entrainement du modèle sur l ensemble du jeu d entraînement X_train et y_train (80%, 20%)
                    3. Prédictions sur l ensemble du jeu de test X_test et y_test
                    4. Evaluation de la performance des modèles en utilisant les métriques appropriées
                    5. Interprétation des coefficients pour comprendre l impact de chaque caractéristique sur la variable cible
                    6. Visualisation et analyse des résultats.
                """)
    
    
    if st.button("Notre choix") :
        data = {
        'Modèles': ['Linear Regression', 'DecisionTree Regressor', 'Random Forest', 'Ridge', 'Lasso', 'ElasticNet', 'Gradient Boosting'],
        'R² train': [0.7919, 0.9018, 0.9843, 0.7919, 0.7910, 0.7919, 0.9999],
        'R² test': [0.7988, 0.8247, 0.8954, 0.7987, 0.7968, 0.7992, 0.9009],
        'MAE train': [0.3900, 0.2646, 0.0038, 0.3901, 0.3911, 0.3900, 0.0012],
        'MAE test': [0.3672, 0.3346, 0.2472, 0.3678, 0.3702, 0.3678, 0.2487],
        'MSE train': [0.2577, 0.1216, 0.0001, 0.2578, 0.2590, 0.2579, 3.80e-06],
        'MSE test': [0.2362, 0.2058, 0.1069, 0.2363, 0.2385, 0.2358, 0.1163],
        'RMSE train': [0.5077, 0.3488, 0.0085, 0.5078, 0.5089, 0.5078, 0.0019],
        'RMSE test': [0.4860, 0.4537, 0.3271, 0.4861, 0.4884, 0.4856, 0.3411]
                }

        # Création du DataFrame
        tab = pd.DataFrame(data)
        tab.index = tab.index #+ 1
        # Trouver l'index de la ligne correspondant à "Random Forest"
        rf_index = tab[tab['Modèles'] == 'Random Forest'].index

        # Appliquer un style personnalisé à la ligne spécifique
        styled_tab = tab.style.apply(lambda x: ['background: #27dce0' if x.name in rf_index else '' for i in x], axis=1)
        
        st.write("")
        st.write("")

        # Afficher le tableau avec le style appliqué
        st.subheader("Métriques de Performance Optimisées")
        st.table(styled_tab)
        st.markdown("""
                        ##### Points à retenir :
                        - Critères de choix : Valeurs R2 élevés, Valeurs autres métriques basses
                        - GradientBoosting OVERFITTING donc Random Forest Regressor
                        - Choix du modèle Random Forest Regressor.
                        """)
        
        st.write("")
        st.write("#### Modèle retenue : Random Forest Regressor.")


    if st.button("Evaluation graphique du modèle") :
        
        #Courbe d'apprentissage
        st.subheader("Courbe d'apprentissage du modèle")
        image_apprentissage = "https://zupimages.net/up/24/04/tr5z.png"
        st.image(image_apprentissage, use_column_width=True)

        st.markdown("""
                    ##### Points à retenir :         
                    - Modèle capable de s'ajuster aux données d'entraînement
                    - Pas de soupçon d'overfitting
                    """)
        
        st.write("")
        st.write("")

        #QQplot, residu et prediction vs vraies
        st.subheader("Graphique des résidus et QQ-plot")
        image_qqplot = "https://zupimages.net/up/24/04/tna6.png"
        st.image(image_qqplot, use_column_width=True)

        st.write("")
        st.write("")

        st.subheader("Distribution cumulative des résidus")
        image_cumul_residus = "https://zupimages.net/up/24/04/l83v.png"
        st.image(image_cumul_residus, use_column_width=True)
        
        st.markdown("""
                    ##### Points à retenir :         
                    - Distributions relativement centrées autour de zéro
                    - Distribution normales des résidus
                    - Très peu de points au dela de +/-1
                    - Variabilité moyenne des résidus pas trop importantes : 18.21%
                    - Discussion ouverte sur l'erreur acceptable par l'organisation.
                    """)
        
        st.write("")
        st.write("")


    if st.button("Features importance") :

        # Visualiser les importances des caractéristiques
        st.subheader("Importance des variables du RandomForestRegressor")
        
        df_coef = pd.read_csv("CoefML_RFR.csv",
                        index_col = 0)
        
        fig = px.bar(df_coef, x='Importance', y='Variable', color="Variable",
                        color_discrete_sequence=px.colors.sequential.Viridis,
                        width =800, height = 600)
        #Caractéristiques du graph
        fig.update_layout(
                showlegend=False,
                xaxis=dict(tickfont=dict(size=16), title_font=dict(size=16)),
                yaxis=dict(tickfont=dict(size=16), title_font=dict(size=16))
            )
            #Afficher le graph
        st.plotly_chart(fig)
   

if page == pages[4]:
    st.header("🔮 Prédiction")
    st.markdown("""<style>h1 {color: #4629dd;  font-size: 70px;/* Changez la couleur du titre h1 ici */} h2 {color: #440154ff;    font-size: 50px /* Changez la couleur du titre h2 ici */} h3{color: #27dce0; font-size: 30px; /* Changez la couleur du titre h3 ici */}</style>""",unsafe_allow_html=True)
    st.markdown("""<style>body {background-color: #f4f4f4;</style>""",unsafe_allow_html=True)

    # Interface utilisateur Streamlit
    st.subheader('Simulation de Prédiction avec Random Forest Regressor')

    def charger_modele():
        # Charger le modèle à partir du fichier Pickle
        with open('modele_rfr.pkl', 'rb') as fichier_modele:
            modele = pickle.load(fichier_modele)
        return modele


    #Selection de la région
    regions = ["Region_Central_and_Eastern_Europe", "Region_Commonwealth_of_Independent_States", 
           "Region_East_Asia", "Region_Latin_America_and_Caribbean", "Region_Middle_East_and_North_Africa",
           "Region_North_America_and_ANZ", "Region_South_Asia", "Region_Southeast_Asia", 
           "Region_Sub-Saharan_Africa", "Region_Western_Europe"]

    region_values = {region: 0 for region in regions}

    # Afficher un menu déroulant pour sélectionner la région
    selected_region = st.selectbox('Sélectionnez une région:', regions)

    # Mettre à jour la valeur de la région sélectionnée à 1
    region_values[selected_region] = 1

    # Ajouter des boutons radio pour choisir la valeur pour chaque région
    st.write("Choisir la valeur pour chaque région :")
    for region in regions:
        region_values[region] = index=0 if region != selected_region else 1
    #st.radio(f"{region} :", [0, 1], key=region,
        
    region_values_list = list(region_values.values())
    
    # Ajouter les widgets pour l'entrée des caractéristiques
    caracteristique1 = float(st.slider("PIB_habitant ", 6.0, 13.0, 9.4))
    caracteristique2 = float(st.slider("Social_support ", 0.0, 1.0, 0.8))
    caracteristique3 = float(st.slider("Healthy_life_expectancy ", 30.0, 80.0, 63.8))
    caracteristique4 = st.slider("Positive_affect ", 0.0, 1.0, 0.7)

    # Prétraitement des caractéristiques avec StandardScaler
    caracteristiques = np.array([[caracteristique1, caracteristique2, caracteristique3, caracteristique4,]+region_values_list])


    def charger_modele():
        # Charger le modèle à partir du fichier Pickle
        with open('modele_rfr_up.pkl', 'rb') as fichier_modele:
            model = pickle.load(fichier_modele)
        return model
      
    # Prévoir la classe avec le modèle
    modele = charger_modele()
    prediction = modele.predict(caracteristiques)
    prediction = np.round(prediction, 3)
    # Afficher la prédiction
    st.markdown(f"<p style='font-size:24px; font-weight:bold;'>Le Ladder Score serait de : {prediction[0]}</p>", unsafe_allow_html=True)

    data_pred = {
        'Variables': ['PIB habitant', 'Social support', 'Healthy life expectancy', 'Positive affect', 'Ladder Score'],
        'Afghanistan (2012 - South Asia)': [7.70, 0.52, 52.24, 0.71, 3.783],
        'Finland (2012 - Western Europe)': [10.74, 0.93, 70.90, 0.80, 7.420]
    }

    st.write("")

    if st.checkbox("Cas concret de prédiction"):
        #st.write("##### Cas concret de prédiction :")
        tab = pd.DataFrame.from_dict(data_pred, orient='index')
        # Définir les colonnes en utilisant la première ligne du DataFrame
        tab.columns = tab.iloc[0]
        # Exclure la première ligne du DataFrame
        tab = tab[1:]    
        st.table(tab)

if page == pages[5]:
    st.header("📌 Conclusion")
    st.markdown("""<style>h1 {color: #4629DD;  font-size: 70px;/* Changez la couleur du titre h1 ici */} h2 {color: #440154ff;    font-size: 50px /* Changez la couleur du titre h2 ici */} h3{color: #27DCE0; font-size: 30px; /* Changez la couleur du titre h3 ici */}</style>""",unsafe_allow_html=True)
    st.markdown("""<style>body {background-color: #F4F4F4;</style>""",unsafe_allow_html=True)
    
    st.write("**Finalement, est-ce que l'argent fait le bonheur ?**")
    st.checkbox("Oui")
    st.checkbox("Non")
    if st.checkbox("Peut-être"):
        st.write("### En synthèse :")
        st.write("Cette analyse du bien-être mondial met en lumière des corrélations significatives, notamment avec l'espérance de vie en bonne santé, le PIB par habitant et la perception du soutien social.")

        image_urls = [
        "https://i.pinimg.com/474x/18/76/94/1876949407d8edf95aac2afbf07374b9.jpg",
        "https://media.lesechos.com/api/v1/images/view/64fb1af8ee8d530d0e61d469/1280x720/0901909959806-web-tete.jpg",
        "https://www.positiveprime.com/wp-content/uploads/2019/07/positiv-min.jpg",
        "https://www.futuribles.com/wp-content/uploads/2018/10/EsperanceVie_X6mLOwl.jpg"
        ]
        #Redimensionner pour toutes la meme tailles
        new_size = (400, 300)
        #Dire que l'on voudra 2 colonnes
        col1, col2 = st.columns(2)
        # Afficher chaque image redimensionnée
        for i, image_url in enumerate(image_urls):          #Parcourir toute les images
            response = requests.get(image_url)              #Récupérer l'image à partir de l'url
            image = Image.open(BytesIO(response.content))   #Lire les données de l'image et l'ouvrir
            resized_image = image.resize(new_size)          #Redimensionner l'image
            if i % 2 == 0: #Si image 0 ou 2 alors col1 et 1 ou 3 col2
               with col1:
                   st.image(resized_image, use_column_width=True)
            else:
                with col2:
                    st.image(resized_image, use_column_width=True)
        st.write("_Attention, ces données datent de 2021, et le monde évolue très vite... Peut-être qu'avec des données à jour, d'autres facteurs entreraient en compte, l'influence du digital avec les réseaux sociaux notamment..._")
    if st.button("Merci") :
        st.write("Nous souhaiterions remercier pour nous avoir aidé sur ce projet :")
        st.write("- Notre mentor sur le projet Tarik Anouar,")
        st.write("- DataScientest dont les animateurs des masterclasses et notre chef de cohorte, Yazid,")
        st.write("- Les données trouvées sur World Happiness Record et sur Kaggle,")
        st.write("- ... ChatGPT!")
