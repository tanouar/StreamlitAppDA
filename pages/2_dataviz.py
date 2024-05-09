import streamlit as st
from streamlit_folium import st_folium
import utils.velo_data_viz as viz
import utils.str_func as common

common.page_config("Données")
common.local_css("css_str.css")
common.menu()


if 'df_velo' not in st.session_state:
    common.load_velo()

df_velo = st.session_state.df_velo

st.html("<h1>Data Visualisation</h1>")

types = ['Comptages par temporalité', 'Classification des compteurs', 'Météo']
type_graph = st.selectbox('',types)

#temporalité
if(type_graph==types[0]):
    temp = st.radio("Temporalité", options=['mois','semaine','jour et heure'], horizontal=True, label_visibility='hidden')
    st.write("")
    
    if(temp=="mois"):
        col1, col2 = st.columns((1, 1))
        with col1:
            st.markdown("#### **Mois**")
            st.markdown("Le graphique représente le nombre moyen de passages de vélos par mois au cours des années 2022 et 2023. Une nette augmentation du nombre moyen de passages de vélos est observée au mois de juin, coïncidant avec l'arrivée des températures estivales. Cette période voit une augmentation significative de l'utilisation des vélos.")
            st.markdown("En revanche, une baisse marquée est constatée en juillet et en août. Ces mois correspondent à la période estivale où de nombreuses personnes partent en vacances, réduisant ainsi l'utilisation des vélos.")
            st.markdown("À partir du mois de septembre, on observe une remontée des courbes, suivie d'une nouvelle baisse à partir de novembre, avec le début de la baisse des températures.")
            st.markdown("Cette tendance suggère une corrélation entre l'utilisation des vélos et les conditions météorologiques et les vacances estivales.")
        col2.plotly_chart(viz.get_compt_temp(df_velo, temp))
    elif(temp=="semaine"):
        col1, col2 = st.columns((1, 1))
        with col1:
            st.markdown("#### **Semaine**")
            st.markdown("Ce graphique représente la fréquentation moyenne hebdomadaire des cyclistes à Paris au cours des années 2022 et 2023, en mettant en évidence les périodes creuses ainsi que les vacances scolaires.")
            st.markdown("Des fluctuations dans la fréquentation cycliste sont observées, avec des variations significatives pendant les vacances scolaires, indiquées par les zones grises, suggérant une réduction des déplacements domicile-école ou domicile-travail, et donc une baisse de l'utilisation du vélo pendant ces périodes.")
            st.markdown("Notamment, une forte baisse de la fréquentation est observée pendant les vacances de Noël, probablement en raison des conditions météorologiques défavorables du début de l’hiver.")
        col2.plotly_chart(viz.get_compt_temp(df_velo, temp))
    else:
        st.markdown("#### **Jour et heure**")
        col1_1, col2_1 = st.columns((1, 1))
        col1_1.plotly_chart(viz.get_compt_temp(df_velo, "jour"))
        col1_1.markdown("Concernant maintenant la fréquence par jour dans la semaine, nous observons une plus grande fréquence sur les jours ouvrés (du lundi au vendredi), avec une plus faible fréquence durant les weekends")
        col2_1.plotly_chart(viz.get_compt_temp(df_velo, "heure")) 
        col2_1.markdown("Concernant la fréquence par heure, nous observons une forte affluence aux heures de pointes, c’est à dire entre 8h et 9h ainsi qu’entre 17h et 19h. L’affluence baisse également drastiquement pendant la nuit, entre minuit et 5h du matin")
        st.markdown("")
        st.markdown("#### **Résumé**")
        col1_2, col2_2 = st.columns((1, 1))
        col1_2.plotly_chart(viz.get_compt_temp(df_velo, "jour et heure"))
        col2_2.markdown("**Le graphique ci-présent qui regroupe les deux visualisations précédentes confirme la tendance observée : nous comprenons que l’usage du vélo se fait principalement pour le trajet domicile / lieu de travail, il est donc un moyen de transport quotidien à part entière.**")
        
#compteurs
elif(type_graph==types[1]):
    m,top,flop = viz.get_map_compt(df_velo)
    top = top[["Nom du compteur", "Comptage horaire"]]
    flop = flop.sort_values(by="Comptage horaire")[["Nom du compteur", "Comptage horaire"]]

    colmap1, colmap2 = st.columns((2, 3))
    with colmap1:
        st.markdown("#### **Classement par comptage horaire moyen**")
        st.markdown("")
        st.markdown("")
        st.markdown("##### **🔺 Top 5**")
        st.dataframe(top, hide_index=True, use_container_width=True)
        st.markdown("##### **🔻 Flop 5**")
        st.dataframe(flop, hide_index=True, use_container_width=True)
    
    with colmap2:
        st_folium(m, use_container_width=True)
    
    st.markdown("")
    st.markdown("La carte ci-dessus nous permet de visualiser les emplacements des compteurs, pondérés par leur moyenne de comptage horaire.")
    st.markdown("Les compteurs top 5 sont précisés par les marqueurs :red[rouges] ➕, les compteurs flop 5 par les marqueurs :green[verts] ➖.")
    st.markdown("On constate que les compteurs comptabilisant le comptage horaire moyen le plus élevé sont principalement situés dans le **centre** (**1er, 2ème, 3ème et 4ème arrondissements**) et le **centre-nord** de Paris (à cheval entre le **9ème et 10ème arrondissement**). Il y a également un trafic élevé entre le pont des invalides et pont de la concorde à l’ouest, ainsi que dans le 11ème arrondissement à l’est.")
    st.markdown("Le flop 5 se situe surtout dans le **17ème et 18ème arrondissement**.")

#meteo
else:
    st.write("#### **Météo - pluie et températures**")
    col1_1, col2_1 = st.columns((1, 1))
    col1_1.plotly_chart(viz.get_meteo(df_velo, "pluie"))
    col1_1.markdown("Nous pouvons constater que le comptage horaire moyen est plus élevé s’il ne pleut pas.")
    col1_1.markdown("Cependant ce graphique peut-être biaisé par le déséquilibre des classes : la grande majorité des heures sont considérées comme non pluvieuses et sont donc sujettes à un pourcentage plus élevé de comptage à 0 que la catégorie Pluie intense.")
    col2_1.plotly_chart(viz.get_meteo(df_velo, "temperatures"))
    col2_1.markdown("Ce graphique nous permet de visualiser la relation entre les températures et les comptages : les comptages augmentent légèrement avec la températures et une intensité de pluie réduite.")
    col2_1.markdown("Ce résultat est confirmé par l'analyse de la régression linéaire entre le comptage horaire et les températures avec un coefficient de 2.529.")