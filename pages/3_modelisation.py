import streamlit as st
import utils.str_func as common
import utils.velo_machine_learning as ml

common.page_config("Données")
common.local_css("css_str.css")
common.menu()

if 'df_velo' not in st.session_state:
    common.load_velo()

df_velo = st.session_state.df_velo


st.html("<h1>Modélisation</h1>")


opt = st.radio("Temporalité", options=['Modèles','Métriques et comparaison'], horizontal=True, label_visibility='hidden')

if (opt=="Modèles"):
    
    modeles = st.selectbox('Choisissez un modèle',('KNN', 'Lasso', 'Ridge', 'Linear Regressor', 'Decision Tree', 'Random Forest Regressor', 'Gradient Boost', 'Méthode de prédiction sans machine learning'))
    
    if (modeles=='KNN'):
        st.markdown("Nous avons exploré l'utilisation d'un modèle non supervisé de type “KNN” pour identifier d'éventuels groupes de compteurs ayant des comportements similaires au niveau du comptage horaire.") 
        colknn1, colknn2 = st.columns((1,1))
        colknn1.plotly_chart(ml.KNN_coude())
        colknn1.markdown("Nous avons ainsi fait appel à la méthode du coude pour déterminer le nombre idéal de clusters.") 
        colknn2.plotly_chart(ml.KNN_centroids(df_velo)) 
        colknn2.markdown("Centroïdes des clusters trouvés par le modèle : à vue d'œil, ce clustering ne semble pas efficace.")
        st.markdown("**Performances**")
        st.markdown("""
                    - *distance Euclidienne moyenne:* `89.72120801832347`
                    - *silhouette score:* `0.43495817245041263`
                    """)
        
    elif (modeles=='Lasso'):
        collasso1, collasso2 = st.columns((1,2))
        with collasso1:
            st.markdown("**Méthodologie**")
            st.markdown("Pour ce modèle nous avons fait le choix d’ajouter une caractéristique booléenne à notre jeu de données qui indique si le nombre de comptages est inférieur ou supérieur à 500.")
            st.markdown("Les ID compteurs ont été encodés avec `pandas.factorize()`.")
            st.markdown("**Performances**")
            st.markdown("Les performances d'autres modèles étaient plus concluantes.")
            st.markdown("""
                    - *alpha:* `0.1`
                    - *score train:* `0.4471265471171989`
                    - *score test:* `0.4600346337363961`
                    """)
        collasso2.pyplot(ml.lasso_graph())
        
    elif(modeles=='Ridge'):
        colridge1, colridge2 = st.columns((1,2))
        with colridge1:
            st.markdown("**Méthodologie**")
            st.markdown("Pour ce modèle nous avons fait le choix d’ajouter une caractéristique booléenne à notre jeu de données qui indique si le nombre de comptages est inférieur ou supérieur à 500.")
            st.markdown("Les ID compteurs ont été encodés avec `pandas.factorize()`.")
            st.markdown("**Performances**")
            st.markdown("Les performances d'autres modèles étaient plus concluantes.")
            st.markdown("""
                    - *alpha:* `0.1`
                    - *score train:* `0.4471265471171989`
                    - *score test:* `0.4600346337363961`
                    """)
        colridge2.pyplot(ml.ridge_graph())

    elif(modeles=='Decision Tree'):
        coldecision1, coldecision2 = st.columns((1,1))
        with coldecision1:
            st.markdown("**Méthodologie**")
            st.markdown("Scores sans hyperparamètres : overfitting.")
            st.markdown("""
                    - *score train:* `1.0`
                    - *score test:* `0.887047592339389`
                    """)
            st.markdown("Pour réduire l'overfitting nous utilisons le diagramme des features importantes pour réduire notre jeux de données.")
            st.markdown("Nous utilisons ensuite GridSearchCV sur le jeux de données réduit pour optimiser les paramètres.")
            st.markdown("Le modèle est réentraîné, nous obtenons de meilleurs scores.")
            st.markdown("""
                    - *score train:* `0.923198647827235`
                    - *score test:* `0.9003758718632677`
                    """)
            
        with coldecision2:
            st.plotly_chart(ml.decision_features())

        colgradient2, colgradient3 = st.columns((1, 1)) 
        colgradient2.plotly_chart(ml.decision_graph())
        with colgradient3 :
            st.markdown("")
            st.markdown("")
            st.markdown("**Interprétation**")
            st.markdown("""
                              Le modèle est assez précis : les données collent plutôt bien à la droite.\n
                              Cependant le modèle a du mal à interpréter les valeurs extrêmes.
                              """)
        
    elif(modeles=='Gradient Boost'):

        colgradient1, colgradient2 = st.columns((1,1))
        with colgradient1:
            st.markdown("**Méthodologie**")
            st.markdown("Les scores obtenus sans hyperparamètres ne dépassant pas les 0.60 nous décidons de tester les paramètres avec GridSearchCV.")
            st.markdown("Avec ces nouveaux paramètres le modèle fait de l'overfitting.")
            st.markdown("""
                    - *score train:* `0.9942219681894201`
                    - *score test:* `0.9596049436424086`
                    """)
            st.markdown("Pour réduire l'overfitting nous utilisons le diagramme des features importantes pour réduire notre jeux de données.")
            st.markdown("Nous utilisons ensuite GridSearchCV sur le jeux de données réduit pour optimiser les paramètres.")
            st.markdown("Le modèle est réentraîné, nous obtenons de meilleurs scores.")
            st.markdown("""
                    - *score train:* `0.9654593624693203`
                    - *score test:* `0.9423245118907382`
                    """)
            
        with colgradient2:
            st.plotly_chart(ml.gradient_features())

        colgradient2, colgradient3 = st.columns((1, 1)) 
        colgradient2.plotly_chart(ml.gradient_graph())
        with colgradient3 :
            st.markdown("")
            st.markdown("")
            st.markdown("**Interprétation**")
            st.markdown("""
                              Le modèle est précis : les données collent bien à la droite.\n
                              Cependant le modèle a du mal à interpréter les valeurs extrêmes.
                              """)
     
    elif(modeles=='Linear Regressor'):
        collinear1, collinear2 = st.columns((1,1))
        with collinear1:
            st.markdown("**Méthodologie**")
            st.markdown("Scores sans hyperparamètres : résultats faibles")
            st.markdown("""
                    - *score train:* `0.17590005722385393`
                    - *score test:* `0.17844050894480878`
                    """)
            st.markdown("Nous en déduisons que ce modèle n'est pas adapté à notre jeux de données.")
            
        with collinear2:
            st.plotly_chart(ml.linear_graph())

    elif(modeles=='Random Forest Regressor'):

        colrand1, colrand2 = st.columns((1,1))
        with colrand1:
            st.markdown("**Méthodologie**")
            st.markdown("Scores sans hyperparamètres : overfitting.")
            st.markdown("""
                    - *score train:* `0.9908590373355587`
                    - *score test:* `0.938203172048873`
                    """)
            st.markdown("Pour réduire l'overfitting nous utilisons le diagramme des features importantes pour réduire notre jeux de données.")
            st.markdown("A l'aide d'une boucle for nous testons différentes 'max_depth' sur le jeux réduit.")
            st.markdown("Le modèle semble donner de bons résultats avec une profondeur maximale autour de 17, nous réentraînons le modèle avec ce paramètre.")
            st.markdown("Les scores obtenus sont plus équilibrés.")
            st.markdown("""
                    - *score train:* `0.8769382121351791`
                    - *score test:* `0.852489391204668`
                    """)
            st.markdown("Le modèle a du mal à interpréter les valeurs extrêmes et reste moins précis que d'autres modèles.")
            
        with colrand2:
            st.plotly_chart(ml.random_features())
        
        colrand3, colrand4 = st.columns((1,1))
        colrand3.plotly_chart(ml.random_max_depth())
        colrand4.plotly_chart(ml.random_graph())
        
    else:

        colelse1, colelse2 = st.columns((1,1))
        with colelse1:
            st.markdown("**Méthodologie**")
            st.markdown("Ce modèle a la forme d’un modèle de régression linéaire pour lequel nous avons déterminé les variables intéressantes à prendre en compte et les coefficients de corrélation qui leur sont associés.")
            st.markdown("La matrice de corrélation met en évidence des coefficients de corrélation intéressants mais pas directement exploitables en l’état : nous avons utilisé la métrique de performance R^2 pour déterminer quelles variables conserver dans notre modèle.")
            st.markdown("Le score R^2 de notre modèle varie naturellement selon l’échantillon choisi, voici ci-dessous sa valeur prise la plus élevée :")
            st.markdown("""
                    - *R2:* `0.2805931139630379`
                    """)
            st.markdown("Le modèle ne semble pas très performant, les “outliers” sont particulièrement mal prédits:")
            st.plotly_chart(ml.msml_preds())
            
        with colelse2:
            st.pyplot(ml.msml_correlation())
            
        

else:
    colmetrics1, colmetrics2 = st.columns((1,1.5))
    df = ml.load_metrics()
    with colmetrics1:
        st.markdown("**Choix du modèle**")
        st.markdown("Le modèle KNN détient un 'silhouette score' peu élevé, nous n’avons ainsi pas pu le considérer comme concluant.")
        st.markdown("La Régression Linéaire, Ridge, Lasso et le modèle de prédiction sans 'machine learning' ont également été rejetée en raison du manque de précision.")
        st.markdown("Les modèles Decision Tree et RandomForest ont montré de bons résultats mais tendait à sous-évaluer les comptages extrêmes ainsi qu'un score R2 plus bas que le GradientBoost.")
        st.markdown("Nous avons donc décidé de conserver le modèle **GradientBoost** pour nos prédictions.")

    with colmetrics2:
        st.markdown("**Métriques**")
        st.dataframe(df.style.applymap(lambda _: "background-color: #bbdefe;", subset=([0], slice(None))), hide_index=True)

