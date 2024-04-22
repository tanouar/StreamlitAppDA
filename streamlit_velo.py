import streamlit as st
import utils.str_func as common

common.page_config("Introduction")
common.menu()
common.page_css()

if 'df_velo' not in st.session_state:
    common.load_velo()

st.image('assets/logo.jpg', caption='Cursus : Data Analyst - Mars 2024 | Lena GUILLOUX, Mélissa CHEMMAMA, Myriam MAHDJOUB, Eléonore HERMAND')

st.html("<h1>Introduction</h1>")