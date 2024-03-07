
from collections import OrderedDict
import streamlit as st

# TODO : change TITLE, TEAM_MEMBERS and PROMOTION values in config.py.
import config

# TODO : you can (and should) rename and add tabs in the ./tabs folder, and import them here.
from tabs import intro, data, visu, stats, model, ccl


st.set_page_config(
    page_title=config.TITLE,
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
)

with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


# TODO: add new and/or renamed tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (data.sidebar_name, data),
        (visu.sidebar_name, visu),
        (stats.sidebar_name, stats),
        (model.sidebar_name, model),
        (ccl.sidebar_name, ccl)
    ]
)


def run():
    
    st.sidebar.image(
        "Data/rechauffement-climatique.jpg",
        width=250,
        )
    
    tab_name = st.sidebar.radio("Sommaire", list(TABS.keys()), 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("### Team members:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    st.sidebar.markdown(
        """
        ----
        Crédit image :
        ©LIONEL BONAVENTURE / AFP
        """,
        unsafe_allow_html=True
        )
    
    tab = TABS[tab_name]

    tab.run()


if __name__ == "__main__":
    run()
