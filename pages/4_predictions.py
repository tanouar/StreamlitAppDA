import streamlit as st
import datetime
import utils.str_func as common
import utils.velo_machine_learning as ml

common.page_config("Données")
common.menu()
common.local_css("css_str.css")

if 'df_velo' not in st.session_state:
    common.load_velo()

df_velo = st.session_state.df_velo
df_velo_ml = ml.pre_process(df_velo)
gradient_model = ml.load_model()

st.html("<h1>Prédictions</h1>")

compteurs = st.multiselect("Compteur", list(df_velo["Nom du compteur"].unique()), ["97 avenue Denfert Rochereau SO-NE"])
left, right = st.columns(2)
with left: 
    date = st.date_input("Date", datetime.date(2023, 3, 6), min_value=datetime.date(2023, 1, 1), max_value=datetime.date(2023, 12, 31))
with right:
    heure = st.time_input('Heure', value=None, step=3600)

df_velo_ml_filtered = ml.filtered_datas(compteurs, date, heure, df_velo_ml)

if(len(df_velo_ml_filtered) != 0):

    df_velo_ml_preds, r2 = ml.predict_velo(df_velo_ml_filtered, gradient_model)
    df_display = df_velo_ml_preds.sort_values(by=["Date et heure de comptage", "Nom du compteur"]).reset_index(drop=True)
    col1, col2 = st.columns(2)
    if(heure==None):
        col1.markdown(f"""<b>R² score :</b> <code>{r2}</code>""", unsafe_allow_html=True)
    else:
        col1.markdown("")
        col1.markdown("")
        col1.markdown("")

    with col1:
        pagination = st.container()
        bottom_menu = st.columns((4, 1))
        batch_size = 12
        with bottom_menu[1]:
            total_pages = (
                int(len(df_display)/batch_size)+(len(df_display) % batch_size>0) if int(len(df_display)/batch_size) > 0 else 1
            )
            current_page = st.number_input("Page", min_value=1, max_value=total_pages, step=1, label_visibility='collapsed')
        with bottom_menu[0]:
            st.markdown(f"Page **{current_page}** of **{total_pages}** ")
        pages = common.split_frame(df_display,batch_size)
        pagination.table(data=pages[current_page - 1])

    
    if(heure==None):
        col2.plotly_chart(ml.preds_viz_jour(df_velo_ml_preds))
    else:
        col2.plotly_chart(ml.preds_viz_heure(df_velo_ml_preds))
else:
    st.write("Aucune valeur réelle à cette date et pour ce compteur.")
