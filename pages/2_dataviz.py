import streamlit as st
from streamlit_folium import st_folium
import streamlit_velo as main
import utils.velo_data_viz as viz
import utils.velo_load_datas as datas

st.set_page_config(
    page_title="Visualisation",
    page_icon="🚴‍♀️",
    layout="wide"
)

main.menu()
main.page_css()

if 'df_velo' not in st.session_state:
    main.load_velo()

df_velo = st.session_state.df_velo

st.html("<h1>Data Visualisation</h1>")

types = ['Comptages par temporalité', 'Classification des compteurs', 'Météo']
type_graph = st.selectbox('',types)

if(type_graph==types[0]):
    temp = st.radio("Temporalité", options=['mois','semaine','jour', 'heure', 'jour et heure'], horizontal=True, label_visibility='hidden')
    st.write("")
    col1, col2 = st.columns((1, 1))
    col1.write("Interprétation")
    col2.pyplot(viz.get_compt_temp(df_velo, temp))

elif(type_graph==types[1]):
    colmap1, colmap2 = st.columns((1, 2))
    m,top,flop = viz.get_map_compt(df_velo)

    with colmap1:
        st.html("<h4 style='color:#71ae24'>Top 5</h1>")
        st.write("1. ", top.iloc[0]["Nom du compteur"], ": ", str(int(top.iloc[0]["Comptage horaire"])))
        st.write("2. ", top.iloc[1]["Nom du compteur"], ": ", str(int(top.iloc[1]["Comptage horaire"])))
        st.write("3. ", top.iloc[2]["Nom du compteur"], ": ", str(int(top.iloc[2]["Comptage horaire"])))
        st.write("4. ", top.iloc[3]["Nom du compteur"], ": ", str(int(top.iloc[3]["Comptage horaire"])))
        st.write("5. ", top.iloc[4]["Nom du compteur"], ": ", str(int(top.iloc[4]["Comptage horaire"])))
        
        st.html("<h4 style='color:#d03c28'>Flop 5</h1>")
        st.write("1. ", flop.iloc[0]["Nom du compteur"], ": ", str(int(flop.iloc[0]["Comptage horaire"])))
        st.write("2. ", flop.iloc[1]["Nom du compteur"], ": ", str(int(flop.iloc[1]["Comptage horaire"])))
        st.write("3. ", flop.iloc[2]["Nom du compteur"], ": ", str(int(flop.iloc[2]["Comptage horaire"])))
        st.write("4. ", flop.iloc[3]["Nom du compteur"], ": ", str(int(flop.iloc[3]["Comptage horaire"])))
        st.write("5. ", flop.iloc[4]["Nom du compteur"], ": ", str(int(flop.iloc[4]["Comptage horaire"])))
    with colmap2:
        st_folium(m, width=1200)

else:
    st.write("meteo")
