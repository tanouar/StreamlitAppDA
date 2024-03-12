import streamlit as st


title = "<h1 style='color: green;'>Températures terrestres</h1>"
sidebar_name = "Conclusions & Perspectives"


def run():
    st.image("Data/rechauffement-climatique.jpg", width=400)


    st.header(sidebar_name)
    st.markdown("---")


    with st.expander("**Contexte**"):
       st.write("Le sujet de l’évolution des températures en corollaire de la production des gaz à effet de serre est très largement étudié depuis des décennies, mettant en évidence un impact des activités humaines sur ce qu’il est commun d’appeler le réchauffement climatique.")
    
    with st.expander("**Rappel des objectifs**"):
       st.write("➽ Collecter et prétraiter des données sur les températures terrestres et les gaz à effet de serre.\n\n➽ Utiliser des techniques de visualisation pour représenter graphiquement les variations de température au fil du temps et leur relation avec les concentrations de gaz à effet de serre.\n\n➽ Effectuer des analyses statistiques pour quantifier les relations entre les températures et les gaz à effet de serre.\n\n➽ Développer des modèles de prédiction en utilisant des techniques de modélisation et de *Machine Learning*.")
    
    with st.expander("**Sources de données**"):
       st.write("Nous disposons de sources de données publiques fiables, combinant d’un côté les productions annuelles par pays des différents gaz à effet de serre et l’évolution relative de la température terrestre qui en découle, et d’autre part les données d’évolution relative de la température terrestre par pays.\n\nCes données sont par ailleurs mesurées ou extrapolées depuis l’ère préindustrielle jusqu’à nos jours.")
    
    with st.expander("**Conclusions**"):
       st.write("➽ Une augmentation des températures globales et locales est détectée sur la période considérée, de l’ordre de 1,5 °C.\n\n➽ Une augmentation massive des productions de gaz à effet de serre depuis le début de l’ère industrielle. Bien que le phénomène soit global sur la planète, cette production est très majoritairement liée à une quinzaine de pays qui représentent à eux-seuls les ¾ des émissions de CO2.\n\n➽ Dans les dernières décennies, une inflexion de la production de gaz à effet de serre par certains pays est détectée, sans impact détectable sur l’évolution globale ou locale de la température terrestre.\n\n➽ La production locale de gaz à effet de serre et l’évolution locale relative de la température terrestre ne sont pas corrélées.\n\n➽ Nos prédictions par *Machine Learning* de l’évolution locale relative de la température terrestre en fonction de la production locale de gaz à effet de serre sur la période considérée ont montré un succès relatif, avec les modèles *RandomForestRegressor* et *XGBoostRegressor*.")
    
    with st.expander("**Perspectives**"):
       st.write("➽ Utiliser les mesures atmosphériques disponibles des différents gaz à effet de serre pour recherche de corrélation et prédiction d’évolution de températures terrestres.\n\n➽ Utiliser des regroupements de pays sur d’autres critères que géographiques (par le niveau de développement, profils de production des gaz à effet de serre, …).\n\n➽ Obtenir une prédiction meilleure pour les gaz à effet de serre hors CO2, uniquement pour les périodes récentes et à venir, ces données étant largement manquantes par pays à part pour les dernières décennies.\n\n➽ Prendre en compte de multiples scénarios de stabilisation de la température terrestre globale (tels que ratifiés dans l’accord de Paris), que ce soit sur les années ou décennies futures comme récentes (cf. inflexion détectée depuis quelques décennies de production de certains gaz à effet de serre par certains pays).")


