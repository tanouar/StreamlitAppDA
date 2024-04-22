import streamlit as st
import streamlit_velo as main
import utils.velo_load_datas as datas
import datetime

main.page_config("Prédiction")
main.menu()
main.page_css()

if 'df_velo' not in st.session_state:
    main.load_velo()

df_velo = st.session_state.df_velo

st.html("<h1>Prédictions</h1>")

countries = st.multiselect("Compteur", list(df_velo["Nom du compteur"].unique()), ["97 avenue Denfert Rochereau SO-NE"])
date = st.date_input("Date", datetime.date(2023, 1, 1), min_value=datetime.date(2023, 1, 1), max_value=datetime.date(2023, 12, 31))
hour = st.time_input('Heure', value=datetime.time(8), step=3600)