import streamlit as st
import streamlit_velo as main

st.set_page_config(
    page_title="Modélisation",
    page_icon="🚴‍♀️",
    layout="wide"
)

main.menu()
main.page_css()

st.html("<h1>Modélisation</h1>")