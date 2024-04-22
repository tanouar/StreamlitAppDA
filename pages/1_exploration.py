import streamlit as st
import streamlit_velo as main
import utils.velo_load_datas as datas

main.page_config("Données")
main.menu()
main.page_css()

if 'df_velo' not in st.session_state:
    main.load_velo()

df_velo = st.session_state.df_velo

st.html("<h1>Exploration des données</h1>")

st.dataframe(df_velo.head(10))