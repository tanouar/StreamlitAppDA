import streamlit as st
import utils.str_func as common

common.page_config("Données")
common.menu()
common.page_css()

if 'df_velo' not in st.session_state:
    common.load_velo()

df_velo = st.session_state.df_velo

st.html("<h1>Exploration des données</h1>")

st.dataframe(df_velo.head(10))