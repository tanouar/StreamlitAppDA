import streamlit as st
import utils.velo_load_datas as datas

st.set_page_config(
    page_title="Velos",
    page_icon="🚴‍♀️",
    layout="wide"
)

def page_css():
        st.markdown(
            f'''
            <style>
                h1 {{
                  color: #001219;
                }}
                div.block-container {{
                    padding-top: 25px;
                }}
            </style>
            ''',
            unsafe_allow_html=True,
        )

def load_velo():
    df_velo = datas.load_velo()
    st.session_state.df_velo = df_velo

def menu():
  with st.sidebar:
      st.image('assets/logo-velo2.png', caption='Lena GUILLOUX, Mélissa CHEMMAMA, Myriam MAHDJOUB, Eléonore HERMAND')
      st.page_link('streamlit_velo.py', label="Introduction", icon="🏠")
      st.page_link('pages/1_exploration.py', label="Exploration des données", icon="📑")
      st.page_link('pages/2_dataviz.py', label="Data Visualisation", icon="📈")
      st.page_link('pages/3_modelisation.py', label="Modélisation", icon="🤖")
      st.page_link('pages/4_predictions.py', label="Prédictions", icon="🗓️")
      st.page_link('pages/5_conclusion.py', label="Conclusion", icon="📌")

menu()
page_css()

if 'df_velo' not in st.session_state:
    load_velo()

st.image('assets/logo.jpg', caption='Cursus : Data Analyst - Mars 2024 | Lena GUILLOUX, Mélissa CHEMMAMA, Myriam MAHDJOUB, Eléonore HERMAND')

st.html("<h1>Introduction</h1>")

#@st.cache_data
#def load():
#  df_velo = datas.load_velo()
#  return df_velo
#df_velo = load()