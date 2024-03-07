import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


title = "Températures terrestres"
sidebar_name = "Statistiques"


def run():
    st.image("Data/fleche.jpg", width=300)

    st.header("Statistiques")

    df=pd.read_csv("Data/merged_owid_temp.csv", index_col=0)  
