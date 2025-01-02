import streamlit as st

title = "<h1 style='color: green;'>Projet 'Températures terrestres'</h1>"
sidebar_name = "Introduction"

def run():
    st.markdown(title, unsafe_allow_html=True)
    st.header(sidebar_name)
    st.markdown("---")

    with st.expander("Quelle est la principale cause de l'augmentation de la température terrestre ?"):
        st.write("L'accumulation de gaz à effet de serre dans l'atmosphère, causée par les activités humaines telles que la combustion des combustibles fossiles, la déforestation et l'agriculture intensive.")
    with st.expander("Comment les gaz à effet de serre contribuent-ils au réchauffement de la planète ?"):
        st.write("Les gaz à effet de serre sont des composés présents dans l'atmosphère qui absorbent et émettent le rayonnement infrarouge, contribuant ainsi au réchauffement de la planète. Les principaux gaz à effet de serre d'origine humaine sont le dioxyde de carbone (CO2), le méthane (CH4) et le protoxyde d'azote (N2O).")
    
    st.write("**Ce projet vise à comprendre les variations de température à l'échelle mondiale et leur corrélation avec les gaz à effet de serre**.")
    
    st.write('**Les objectifs** :')
    st.write('➽ Collecter et prétraiter des données sur les températures terrestres et les gaz à effet de serre')
    st.write('➽ Utiliser des techniques de visualisation pour représenter graphiquement les variations de température au fil du temps et leur relation avec les concentrations de gaz à effet de serre')
    st.write('➽ Effectuer des analyses statistiques pour quantifier les relations entre les températures et les gaz à effet de serre')
    st.write('➽ Développer des modèles de prédiction en utilisant des techniques de modélisation et de *Machine Learning*')

    st.write("\n\n")
    st.write("\n\n")
    st.write("\n\n")



