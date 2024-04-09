import streamlit as st
import utils.velo_load_datas as datas

st.set_page_config(
    page_title="Velos",
    page_icon="🚴‍♀️"
)

st.write("# Traffic cycliste à Paris 🚴‍♀️")
st.markdown(
      """
        - Analyser les comptages sur les années 2022 et 2023 
        - Prédire les comptages pour l'année 2024
    """
    )
st.sidebar.success("Pages")

@st.cache_data
def load():
  df_velo_2022, df_velo_2023 = datas.load_velo()
  return df_velo_2022

df_velo_2022 = load()
st.dataframe(df_velo_2022.head())
