import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


title = "Températures terrestres"
sidebar_name = "Exploration des données"


def run():
    
    st.image("Data/data.jpg", width=400)

    st.header("Exploration des données")

    df=pd.read_csv("Data/merged_owid_temp.csv", index_col=0)  

    st.dataframe(df.head(10))

    st.write(df.shape)
    st.dataframe(df.describe())

    if st.checkbox("Afficher les NA") :
        st.dataframe(df.isna().sum())
