import streamlit as st
import utils.velo_load_datas as datas
from streamlit_extras.stylable_container import stylable_container

def page_config(title):
    st.set_page_config(
    page_title=title,
    page_icon="🚴‍♀️",
    layout="wide"
)

def local_css(file_name):
    with open('css/'+file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def load_velo():
    df_velo = datas.load_velo()
    st.session_state.df_velo = df_velo

def menu():
  
  with st.sidebar:
    local_css('css_nav.css')
    st.image('assets/stats.png', use_column_width=True)
    st.image('assets/menu.png')
    st.page_link('streamlit_velo.py', label="Introduction", icon="🏠")
    st.page_link('pages/1_exploration.py', label="Exploration des données", icon="🔍")
    st.page_link('pages/2_dataviz.py', label="Data Visualisation", icon="📈")
    st.page_link('pages/3_modelisation.py', label="Modélisation", icon="⚙️")
    st.page_link('pages/4_predictions.py', label="Prédictions", icon="🗓️")
    st.page_link('pages/5_conclusion.py', label="Conclusion", icon="📌")
    
    with stylable_container(key="navbar_team", css_styles="""{background-color:#0b5394;color:white;border-radius:4px;padding-top:20px;padding-bottom:20px;padding-left:10px;margin-top:45px}""",):
        col1, col2 = st.columns((1, 3))
        col1.image('assets/team.png')
        col2.markdown("")
        col2.markdown("""**EQUIPE**""")
        st.markdown("""
                    - **Cursus** : *Data Analyst* 
                    - **Formation** : *Bootcamp*
                    - **Promotion** : *Mars 2024*
                    """)

@st.cache_data(show_spinner=False)
def split_frame(input_df, rows):
    df = [input_df.loc[i : i + rows - 1, :] for i in range(0, len(input_df), rows)]
    return df