
#Intro
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
import zipfile

def wide_space_default():
    st.set_page_config(layout="wide")
wide_space_default()

# Main Code
github_co2_data = pd.read_csv("Aug24_WorldTemp/owid-co2-data.csv")
github_explanation = pd.read_csv("Aug24_WorldTemp/owid-co2-codebook.csv")
owid_surf_temp_anom = pd.read_csv("Aug24_WorldTemp/OWID_01_Surface_Temp_anomaly_historical_data.csv")
kaggle_temp_change_NOFLAG = pd.read_csv("Aug24_WorldTemp/Environment_Temperature_change_E_All_Data_NOFLAG.csv", encoding='cp1252')
owid_country_infos = pd.read_csv("Aug24_WorldTemp/OWID_02_CountryInfos.csv")
kaggle_temp_change_1 = pd.read_csv("Aug24_WorldTemp/FAOSTAT_data_1-10-2022_part1.csv")
kaggle_temp_change_2 = pd.read_csv("Aug24_WorldTemp/FAOSTAT_data_1-10-2022_part1.csv")
kaggle_temp_change = pd.concat([kaggle_temp_change_1, kaggle_temp_change_2], ignore_index=True)



#Creating Main Structure
st.sidebar.title("Summary")
pages=["Introduction", "Data Exploration", "Data Vizualization","Target Variable Choice","Pre-processing and Data Cleaning" ,"Modelling", "Prediction", "Conclusion"]
page=st.sidebar.radio("Select Section", pages)

st.sidebar.markdown(
    """
    - **Course** : Data Analyst
    - **Formation** : Bootcamp
    - **Month** : AUG.2024
    - **Group** : 
        - Carmine Saffioti
        - Michaela Lange
        - Khaldoun Zaal""")

# Page 0 Introduction
if page == pages[0] :
    st.markdown(
    """
    <style>
    .centered-title {
        font-size: 32px;
        text-align: center;
        border-top: 2px solid black;
        border-bottom: 2px solid black;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.markdown('<h1 class="centered-title">DataScientest - Wolrd Temperature Project - AUG24</h1>', unsafe_allow_html=True)

    st.markdown(
        """
        <div style='text-align: center;'>
            <img src='https://www.esa.int/var/esa/storage/images/esa_multimedia/images/2019/12/sea-surface_temperature_change/21499291-3-eng-GB/Sea-surface_temperature_change_pillars.gif' width='600'/>
        </div>
        """, 
        unsafe_allow_html=True
    )
   
    st.header("Introduction")
    text = """Climate change stands as one of the most urgent challenges of our era, marked by significant shifts in weather patterns and rising global temperatures.
    A key aspect of grasping and addressing its effects is the analysis of surface temperature anomalies, which represent the difference between actual surface temperatures and long-term averages.
    Examining these anomalies helps uncover the driving forces behind climate change, including greenhouse gas emissions, population expansion, and economic growth.
    This study applies machine learning techniques to model and forecast surface temperature anomalies, leveraging socioeconomic and environmental factors."""

    st.markdown(f"""
    <div style="text-align: justify;">
    <p>{text}</p>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("""
    <div style="text-align: justify;">
    <p><strong>Understanding what impacts our planet's temperature changes over time is vital for understanding the dynamics of climate change.</strong></p>
    </div>
    """, unsafe_allow_html=True)


    st.header("Objective")
    st.markdown("""
    <div style="text-align: justify; margin-top: 20px;">
    The main goal of this project was to build a predictive model for surface temperature anomalies using a dataset that incorporates variables like GDP, CO2 emissions, population, and other relevant factors.
    The project involved identifying and proactively processing data structures to ensure data quality, visualizing the data to identify trends and relationships,and using machine learning methods to model the relationship between CO2 emissions and abnormal temperatures.
    </div>
    """, unsafe_allow_html=True)


# Page 1 Data Exploration
if page == pages[1] :
    st.markdown(
    """
    <style>
    .centered-title {
        font-size: 28px;
        text-align: center;
        border-top: 2px solid black;
        border-bottom: 2px solid black;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.markdown('<h1 class="centered-title">Data Exploration</h1>', unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.write("To perform the data analysis and the study of World temperature change, the following dataset available online were selected.")
    st.markdown("""
        1. **[Nasa Dataset: Zonal Annual Temperature Anomalies](https://data.giss.nasa.gov/gistemp/)** 
        2. **[FAOSTAT Temperature Change - from Kaggle](https://www.kaggle.com/datasets/sevgisarac/temperature-change/?select=FAOSTAT_data_1-10-2022.csv)**
        3. **[FAOSTAT Temperature Change - from Kaggle - NO Flag](https://www.kaggle.com/datasets/sevgisarac/temperature-change/?select=Environment_Temperature_change_E_All_Data_NOFLAG.csv)** 
        4. **[CO2 and Greenhouse Gas Emissions database by Our World in Data - from GitHub](https://github.com/owid/co2-data)**
        5. **[HARDCRUT Information about Surf Temperature, historical data till 2017 - From Our World in Data](https://ourworldindata.org/grapher/hadcrut-surface-temperature-anomaly )**
        6. **[World Regions - According to Our World in Data](https://ourworldindata.org/grapher/continents-according-to-our-world-in-data)**
        7. **[Annual CO2 Emissions by region - from Our World in Data](https://ourworldindata.org/grapher/annual-co-emissions-by-region)**
        """)



    st.write("This Streamlit App has the objective to show only the relevant work that brought the team to a relevant conclusion.")
    text = """
    The datasets varied in size and complexity, ranging from a few hundred rows to over two hundred thousand rows. 
    They included various features, such as annual CO₂ emissions, temperature anomalies, and geographical data, providing a comprehensive view of global warming trends.
    """
    st.markdown(f"""
    <div style="text-align: justify;; margin-top: 20px;">
    <p>{text}</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("## Data Audit for the main Sources")
    st.write("### GitHub Data")
    st.write("This Database was used as our primary source of data for the Machine Learning part, as it included all explanatory variables. ")
    with st.expander("**Expand GitHub Findings**"):
        st.write("Preview of the GitHub File:")
        st.dataframe(github_co2_data.head())
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Description:")
            st.write("""
            This dataset provides comprehensive data on global carbon dioxide (CO2) emissions, greenhouse gas (GHG) emissions, and related climate metrics across countries over time.
            It includes information on sector-specific CO2 emissions (e.g., from coal, oil, and land-use change), energy consumption, GDP, population, and temperature changes attributed to GHGs.
            The dataset is useful for analyzing trends in emissions, their drivers, and their impact on climate change.""")

            st.write("In order to give a better explanation of the variables included in this dataset, A codebook that explains all columns is provided below:")
            st.dataframe(github_explanation)
        with col2:
            st.markdown(
            """
            - **Rows** : 47415
            - **Columns** : 79
            - **Description** :  Included comprehensive data on CO₂ emissions and other greenhouse gases, such as methane, for various countries
            - **Highlights**: 
                - Not only countries were listed. 
                - Missing values are not equally spread for each column.
                - To better provide data visualization for this dataset, it will be divided based on the analysis we wanted to perform. 
                - Time Span:  From 1750 to 2022, Most of data are concentrated from 1860 to 2022
            """)

            #Na values per Column - CUMULATIVE 
            github_co2_data_na = github_co2_data.isna().sum(axis=0)
            github_co2_data_na_perc = round(github_co2_data_na/github_co2_data.shape[0]*100,2)
            NaN_df = pd.DataFrame([github_co2_data_na, github_co2_data_na_perc], index=["Abs value NaN", "Perc NaN"]).T
            st.dataframe(NaN_df, use_container_width=True)
    
    
    st.divider()
    st.write("### Surface Temperature Anomalies")
    st.write(f"This dataset was used as the **Target Variable** for our Machine Learning Models.")
    with st.expander("**Expand Surface Temperature Anomalies Findings**"):
        st.write("Preview of the Surface Temperature Anomalies DataFrame:")

        col1, col2 = st.columns(2)


        with col1:
            st.data_editor(kaggle_temp_change_NOFLAG,
                           column_config={"Year": st.column_config.NumberColumn(format="%d"), "Entity" : st.column_config.Column(width="large")},
                            hide_index=True,
                            )

            st.write("### Description:")
            text = """
            This dataset records temperature changes and standard deviations across different countries/regions from 1961 to 2019, broken down by months. 
            It provides yearly temperature variation data for each area, allowing for analysis of climate trends over time. 
            The measurements are expressed in degrees Celsius, indicating changes in temperature for each month and region.
            Data is based on the GISTEMP data (Global Surface Temperature Change) 
            distributed by the National Aeronautics and Space Administration Goddard Institute for Space Studies (NASA-GISS).
            """
            st.markdown(f"""
            <div style="text-align: justify;; margin-top: 20px;">
            <p>{text}</p>
            </div>
            """, unsafe_allow_html=True)

            
        with col2:
            #Na values per Column - CUMULATIVE 
            owid_surf_temp_anom_na = owid_surf_temp_anom.isna().sum(axis=0)
            owid_surf_temp_anom_na_perc = round(owid_surf_temp_anom_na/owid_surf_temp_anom.shape[0]*100,2)
            NaN_df = pd.DataFrame([owid_surf_temp_anom_na, owid_surf_temp_anom_na_perc], index=["Abs value NaN", "Perc NaN"]).T
            st.dataframe(NaN_df, use_container_width=True)

            st.markdown(
            """
            - **Rows** : 29566
            - **Columns** : 4
            - **Description** :  Contained surface temperature anomaly data from 1860 to 2017.
            - **Highlights**: 
                - Only Countries are listed as Entity, No grouping. 
                - Surface temperature anomaly data range from -6.84 to 4.66, with a mean of -0.01, indicating variations in temperature anomalies.
                - Since this dataset is focused on the surface temperature anomaly for each year for each country, it will be likely to be used as a reference (target quantity) for future Data Analysis.
            - **Time Span**:  1860 to 2017
            """)



    st.divider()
    st.write("### Kaggle Surface Temp Anomalies NO FLAG")
    st.write(f"This dataset was used as the **Target Variable** for our Machine Learning Models, to maximize the performance of our models.")
    with st.expander("**Kaggle Surface Temp Anomalies - NO FLAG**"):
        st.write("Preview of the Kaggle Surface Temp Anomalies DataFrame:")
        col1, col2 = st.columns(2)

        with col1:
            st.data_editor(kaggle_temp_change_NOFLAG,
                           column_config={"Area" : st.column_config.Column(width="large"), "Months Code": st.column_config.NumberColumn(format="%d")},
                            hide_index=True,
                            )

            st.write("### Description:")
            text = """
            This dataset contains historical surface temperature anomalies for various entities (countries) from 1850 to 2017. It tracks deviations in surface temperatures from a baseline, providing insights into climate change over time.
            It has been downloaded from Our World in Data as it is a publicly available source.
            The dataset contains infos about Country and Country Regions.
            The surface temperature anomaly is measured in degrees Celsius and it  is relative to the 1951-1980 global average temperature.
            Data is based on theHadCRUT analysis from the Climatic Research Unit (University of East Anglia) in conjunction with the Hadley Centre (UK Met Office)
            """
            st.markdown(f"""
            <div style="text-align: justify;; margin-top: 20px;">
            <p>{text}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            #Na values per Column - CUMULATIVE 
            kaggle_temp_change_NOFLAG_na = kaggle_temp_change_NOFLAG.isna().sum(axis=0)
            kaggle_temp_change_NOFLAG_na_perc = round(kaggle_temp_change_NOFLAG_na/kaggle_temp_change_NOFLAG.shape[0]*100,2)
            NaN_df = pd.DataFrame([kaggle_temp_change_NOFLAG_na, kaggle_temp_change_NOFLAG_na_perc], index=["Abs value NaN", "Perc NaN"]).T
            st.dataframe(NaN_df, use_container_width=True)

            st.markdown(
            """
            - **Rows** : 9656
            - **Columns** : 66
            - **Description** :  Temperature changes and standard deviations across different countries/regions from 1961 to 2019.
            - **Highlights**: 
                - Values about temperature change are stored per month, per entity. 
                - Many of the columns are years.  
                - For better use of this dataset, data manipulation was needed.
                -  this dataset is focused on the surface temperature anomaly for each year for each country, it will be likely to be used as a reference (target quantity) for future Data Analysis.
            - **Time Span**:  1961 to 2019
            """)

    st.divider()
    st.write("### OWID Country Information")
    st.write(f"This dataset was for Data Visualization Purposes and to analyze trends, highlights on a continental level.")
    with st.expander("**OWID Country Information**"):
        st.write("Preview of the OWID Country Information DataFrame:")
        col1, col2 = st.columns(2)

        with col1:
            st.data_editor( owid_country_infos,
                            column_config={"Entity" : st.column_config.Column(width="large"),  "Year": st.column_config.NumberColumn(format="%d")},
                            hide_index=True,
                            use_container_width=True  # This will make the dataframe occupy the entire width
                            )

            st.write("### Description:")
            text = """
            This dataset provides information on 285 countries or regions, including their names, unique codes, associated continentsIt categorizes entities by continent and contains no missing values. 
            The dataset appears to serve as a reference for country and region information, likely used in conjunction with other data sources.
            """
            st.markdown(f"""
            <div style="text-align: justify;; margin-top: 20px;">
            <p>{text}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            #Na values per Column - CUMULATIVE 
            owid_country_infos_na = owid_country_infos.isna().sum(axis=0)
            owid_country_infos_na_perc = round(owid_country_infos_na/owid_country_infos.shape[0]*100,2)
            NaN_df = pd.DataFrame([owid_country_infos_na, owid_country_infos_na_perc], index=["Abs value NaN", "Perc NaN"]).T
            st.dataframe(NaN_df, use_container_width=True)

            st.markdown(
            """
            - **Rows** : 285
            - **Columns** : 4
            - **Description** :  Country Name associated with Continents
            - **Highlights**: 
                - No missing Values 
                - Since this is only a grouping, no further analysis will be performed on this database alone.
                - For better use of this dataset, data manipulation was needed.
            """)

# Page 2 - Data Visualization - Page Text and Calculations
if page == pages[2] :
    st.markdown(
    """
    <style>
    .centered-title {
        font-size: 28px;
        text-align: center;
        border-top: 2px solid black;
        border-bottom: 2px solid black;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.markdown('<h1 class="centered-title">Data Visualization</h1>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    text = """
    In order to provide an idea of the Data we used for our project, in this section we will present all the relevant graphs and Tables. 
    """
    st.markdown(f"""
    <div style="text-align: justify;; margin-top: 20px;">
    <p>{text}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## GitHub Data Exploration")
    owid_country_infos = owid_country_infos.drop(columns="Year") 
    github_co2_data_countries = github_co2_data[github_co2_data["iso_code"].notnull()]
    github_co2_data_countries = github_co2_data_countries.rename(columns={"country" : "Entity"})
    github_co2_data_countries = pd.merge(github_co2_data_countries, owid_country_infos[['Entity', 'Continent']], on='Entity', how='inner')
    # Define the categories and their corresponding columns
    categories = {
        "General Information": #1
          ["Entity", "year", "iso_code"],

        "Population and Economy": #2
          ["population", "gdp"],

        "Climate Change": #3
          ["share_of_temperature_change_from_ghg", "temperature_change_from_ch4", "temperature_change_from_co2", "temperature_change_from_ghg", "temperature_change_from_n2o"],


        "Source-specific CO2 Emissions": #4
          ["co2", "co2_including_luc", "cement_co2", "coal_co2", "flaring_co2", "gas_co2", "land_use_change_co2", "oil_co2", "other_industry_co2"],


        "CO2 Emissions per Capita": #5
          ["co2_per_capita", "co2_including_luc_per_capita", "cement_co2_per_capita", "coal_co2_per_capita", "flaring_co2_per_capita", "gas_co2_per_capita", "land_use_change_co2_per_capita", "oil_co2_per_capita", "other_co2_per_capita"],

        "CO2 Emissions per GDP": #6
          ["co2_per_gdp", "co2_including_luc_per_gdp", "consumption_co2_per_gdp"],

        "CO2 Emissions per Unit of Energy": #7
          ["co2_per_unit_energy", "co2_including_luc_per_unit_energy"],

        "CO2 Emissions from Consumption": #8
          ["consumption_co2", "consumption_co2_per_capita"],

        "CO2 Emissions from Trade": #9
          ["trade_co2", "trade_co2_share"],

        "Cumulative CO2 Emissions": #10
          ["cumulative_cement_co2", "cumulative_co2", "cumulative_co2_including_luc", "cumulative_coal_co2", "cumulative_flaring_co2", "cumulative_gas_co2", "cumulative_luc_co2", "cumulative_oil_co2", "cumulative_other_co2"],

        "Share of Global CO2 Emissions": #11
          ["share_global_cement_co2", "share_global_co2", "share_global_co2_including_luc", "share_global_coal_co2", "share_global_flaring_co2", "share_global_gas_co2", "share_global_luc_co2", "share_global_oil_co2", "share_global_other_co2", "share_global_cumulative_cement_co2", "share_global_cumulative_co2", "share_global_cumulative_co2_including_luc", "share_global_cumulative_coal_co2", "share_global_cumulative_flaring_co2", "share_global_cumulative_gas_co2", "share_global_cumulative_luc_co2", "share_global_cumulative_oil_co2", "share_global_cumulative_other_co2"],

        "CO2 Emissions Growth": #12
          ["co2_growth_abs", "co2_growth_prct", "co2_including_luc_growth_abs", "co2_including_luc_growth_prct"],

        "Total GHG Emission": #13
          ["total_ghg", "total_ghg_excluding_lucf"],

        "GHG per Capita": #14
          ["ghg_per_capita", "ghg_excluding_lucf_per_capita"],

        "Other Greenhouse Gases (GHG)": #15
          ["methane", "nitrous_oxide"],

        "Other GHG per Capita": #16
          ["methane_per_capita", "nitrous_oxide_per_capita"],

        "Energy": #17
         ["primary_energy_consumption", "energy_per_capita", "energy_per_gdp"]
    }

    # Chech if category division worked correctly
    total_values = sum(len(v) for v in categories.values())
    print("Total number of values in the dictionary - categories- :", total_values)
    del total_values


    # Function to combine columns from one or multiple categories into a single DataFrame
    def combine_categories(df, categories, *cats):
        combined_columns = []

        for cat in cats:
            combined_columns.extend(categories[cat])

        # Create a new DataFrame with the combined columns
        combined_df = df[combined_columns]
        return combined_df

    df_years_1850= github_co2_data_countries[(github_co2_data_countries['year'] >= 1850)]

    #df with annual values
    df_annual_co2= combine_categories(df_years_1850, categories, "General Information","Population and Economy", "Climate Change","Source-specific CO2 Emissions")

    #columns to remove because they are not relevant
    columns_to_remove = ['other_industry_co2', 'cumulative_other_co2', 'iso_code',"share_of_temperature_change_from_ghg","cumulative_co2_including_luc","co2_including_luc"]


    df_annual_co2=df_annual_co2.drop(columns=columns_to_remove, errors='ignore')

    # Printing Correlation Matrix with NON filled Database

    # Select numeric columns
    numeric_df = df_annual_co2.select_dtypes(include=[np.number])

    #  Calculate the correlation matrix
    correlation_matrix = numeric_df.corr()

# Page 2 - Data Visualization - GitHub Graphs
if page == pages[2] :
    # Visualize the correlation matrix
    with st.expander("**Graph 01 Annual CO2 Emissions including land use change for Top 5 Countries**"):
        col1, col2 = st.columns([7,4])  # Adjust the ratio as needed
        with col1:
            df_annual_co2_dropped = df_annual_co2.dropna()
            top_countries = df_annual_co2_dropped.groupby('Entity')['co2'].sum().nlargest(5).index

            # Step 2: Filter the dataframe to include only the top 5 countries
            df_top_countries = df_annual_co2_dropped[df_annual_co2_dropped['Entity'].isin(top_countries)]

            # Define a palette with consistent colors
            palette = sns.color_palette("husl", len(top_countries))
            country_colors = dict(zip(top_countries, palette))

            # Step 3: Analyze the annual CO2 emissions for these countries
            plt.figure(figsize=(10, 8))  # Set the size of the plot (width, height)
            sns.lineplot(data=df_top_countries, x='year', y='co2', hue='Entity', marker='o', palette=country_colors)
            plt.title('Annual CO2 Emissions including land use change for Top 5 Countries')
            plt.xlabel('Year')
            plt.ylabel('CO2 Emissions (MtCO2)')
            plt.legend(title='Country')
            plt.grid(True)
            st.pyplot(plt)    
            # Step 4: Analyze the cumulative CO2 emissions for these countries
            plt.figure(figsize=(10, 8))  # Set the size of the plot (width, height)
            for country in top_countries:
                country_data = df_top_countries[df_top_countries['Entity'] == country]
                plt.plot(country_data['year'], country_data['co2'].cumsum(), label=country, color=country_colors[country])
            plt.xlabel('Year')
            plt.ylabel('Cumulative CO2 Emissions')
            plt.title('Sum of annual CO2 Emissions Over the Years for Top 5 Countries')
            plt.legend(title='Country')
            plt.grid(True)
            st.pyplot(plt)

        with col2:
            text = """
            Interpretation:

            * The two charts illustrate the annual and cumulative CO2 emissions for the top five countries.

            Annual CO2 Emissions:
            * China has seen a rapid and substantial rise in annual CO2 emissions in recent decades, far exceeding the other countries.
            * The United States peaked in emissions around 2000, followed by a slight decline.
            * Russia, Germany, and the United Kingdom show more stable or slightly declining trends over time.

            Cumulative CO2 Emissions:
            * The United States leads in cumulative CO2 emissions, reflecting its significant historical contribution to climate change.
            * China’s cumulative emissions have also surged, underscoring its growing recent impact.
            * Russia, Germany, and the United Kingdom have much lower cumulative emissions compared to the United States and China.
            """
            st.markdown(f"""
            <div style="text-align: justify;; margin-top: 20px;">
            <p>{text}</p>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("**Graph 02 - Distribution of CO2 Emissions by Category**"):
        col1, col2 = st.columns([7,4])  # Adjust the ratio as needed
        with col1:
            co2_categories = ['oil_co2', 'gas_co2', 'coal_co2', 'flaring_co2', 'cement_co2', 'land_use_change_co2']
            co2_sums = df_annual_co2_dropped[co2_categories].sum()

            # Create custom labels for the legend with the category name and percentage
            labels = [f'{category}: {percentage:.1f}%' for category, percentage in zip(co2_sums.index, 
                        100 * co2_sums / co2_sums.sum())]

            # Create a pie chart
            plt.figure(figsize=(10, 7))
            wedges, texts, autotexts = plt.pie(co2_sums, 
                                                autopct='%1.1f%%', # Adds percentages inside slices
                                                startangle=140, 
                                                wedgeprops={'edgecolor': 'black'})

            # Style the autotext (percentages inside slices)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(10)

            # Add a legend to the side with the custom labels
            plt.legend(wedges, labels, title="CO₂ Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

            # Add title
            plt.title('Distribution of CO2 Emissions by Category')

            # Show the plot
            st.pyplot(plt)
        with col2:
            text = """
            Interpretation:

            * The pie chart displays the distribution of CO2 emissions by source of category. 
            * The largest contributor is land use change, which accounts for 33.8% of total emissions. Coal follows closely at 28.9%, underscoring its role as a major source of CO2 emissions. 
            * Other categories, such as oil, gas, cement, and flaring, contribute smaller shares: oil makes up 23.1%, gas 11.8%, while cement and flaring represent only minor portions of the total.
            """
            st.markdown(f"""
            <div style="text-align: justify;; margin-top: 20px;">
            <p>{text}</p>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("**Graph 03 - Distribution of Temperature Change by Category**"):
        col1, col2 = st.columns([7,4])  # Adjust the ratio as needed
        with col1:
            Temp_categories = ['temperature_change_from_co2', 'temperature_change_from_ch4', 'temperature_change_from_n2o',]
            Temp_sums = df_annual_co2_dropped[Temp_categories].sum()

            co2_categories = ['oil_co2', 'gas_co2', 'coal_co2', 'flaring_co2', 'cement_co2', 'land_use_change_co2']
            co2_sums = df_annual_co2_dropped[co2_categories].sum()

            # Create custom labels for the legend with the category name and percentage
            labels = [f'{category}: {percentage:.1f}%' for category, percentage in zip(Temp_sums.index, 
                        100 * Temp_sums / Temp_sums.sum())]

            # Create a pie chart
            plt.figure(figsize=(10, 7))
            wedges, texts, autotexts = plt.pie(Temp_sums, 
                                                autopct='%1.1f%%', # Adds percentages inside slices
                                                startangle=140, 
                                                wedgeprops={'edgecolor': 'black'})

            # Style the autotext (percentages inside slices)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(10)

            # Add a legend to the side with the custom labels
            plt.legend(wedges, labels, title="CO₂ Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

            # Add title
            plt.title('Distribution of Temperature Change by Category')
            # Show the plot
            st.pyplot(plt)
        with col2:
            text = """
            Interpretation:

            The pie chart helps in understanding the relative impact of these gases on global temperature change and highlights the areas where mitigation efforts could be most effective.

            1. Temperature Change from CO2 (68.9%):
            The largest portion of the temperature change is attributed to CO2 emissions.
            This indicates that CO2 is the most significant contributor to the observed temperature changes.

            2. Temperature Change from CH4 (27.8%):
            Methane (CH4) is the second largest contributor.
            Methane, while less abundant than CO2, has a much higher global warming potential, making its contribution significant.

            3. Temperature Change from N2O (34%):
            Nitrous oxide (N2O) contributes 3.4% to the total temperature change.
            Although its contribution is smaller compared to CO2 and CH4, it still plays a role in the overall warming effect.

            """
            st.markdown(f"""
            <div style="text-align: justify;; margin-top: 20px;">
            <p>{text}</p>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("**Graph 04 - Correlation Matrix**"):
        col1, col2 = st.columns([7,4])  # Adjust the ratio as needed
        with col1:  # Centering the plot in the second column
            plt.figure(figsize=(15, 12))  # Set the size of the plot (width, height)
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Correlation Matrix of Df Annual CO2 without Filling NAs')

            # Display the plot in the specified column width
            st.pyplot(plt)

            # Sum the relevant CO2 categories
            co2_categories = ['oil_co2', 'gas_co2', 'coal_co2', 'flaring_co2', 'cement_co2', 'land_use_change_co2']
            co2_sums = df_annual_co2_dropped[co2_categories].sum()

            # Create custom labels for the legend with the category name and percentage
            labels = [f'{category}: {percentage:.1f}%' for category, percentage in zip(co2_sums.index, 
                       100 * co2_sums / co2_sums.sum())]
        with col2:
            text = """
            Interpretation:
            
            The Correlation Matrix shows the correlation between a set of variables included in the GitbHub database. 
            We can see a clear area where correlations are stronger. The main correlation is between the temperature change attributed to the GHG emisssion and the various CO2 sources. 

            """
            st.markdown(f"""
            <div style="text-align: justify;; margin-top: 20px;">
            <p>{text}</p>
            </div>
            """, unsafe_allow_html=True)

    # ------------- KAggle Manipulation
    columns_to_drop = ["Domain Code", "Domain", "Element Code", "Element", "Unit"]
    kaggle_temp_change_cleaned = kaggle_temp_change.drop(columns=columns_to_drop)
    kaggle_temp_yearly = kaggle_temp_change_cleaned.loc [kaggle_temp_change_cleaned["Months Code"] == 7020, ["Area", "Year", "Value"]]
    kaggle_temp_yearly = kaggle_temp_yearly.rename(columns={"Area": "Entity"})
    kaggle_temp_yearly = pd.merge(kaggle_temp_yearly, owid_country_infos, on=["Entity"], how='left')
    # Assigning the remaining Continents to the Countries - Creating a Function 
    def assign_continents(country):
        asia = ['Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei Darussalam', 'Cambodia',
                'China, mainland','China', 'China, Hong Kong SAR', 'China, Macao SAR', 'China, Taiwan Province of', 'Democratic People\'s Republic of Korea',
                'Georgia', 'India', 'Indonesia', 'Iran (Islamic Republic of)', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan',
                'Kuwait', 'Kyrgyzstan', 'Lao People\'s Democratic Republic', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar',
                'Nepal', 'Oman', 'Pakistan', 'Palestine', 'Philippines', 'Qatar', 'Republic of Korea', 'Saudi Arabia', 'Singapore',
                'Sri Lanka', 'Syrian Arab Republic', 'Tajikistan', 'Thailand', 'Timor-Leste', 'Turkey', 'Turkmenistan', 'USSR', 'United Arab Emirates',
                'Uzbekistan', 'Viet Nam', 'Yemen']

        europe = ['Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Belgium-Luxembourg', 'Bosnia and Herzegovina', 'Bulgaria',
                  'Channel Islands', 'Croatia', 'Cyprus', 'Czechia', 'Czechoslovakia', 'Denmark', 'Estonia', 'Faroe Islands', 'Finland',
                  'France', 'Germany', 'Gibraltar', 'Greece', 'Holy See', 'Hungary', 'Iceland', 'Ireland', 'Isle of Man', 'Italy', 'Latvia',
                  'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway',
                  'Poland', 'Portugal', 'Republic of Moldova', 'Romania', 'Russian Federation', 'San Marino', 'Serbia', 'Serbia and Montenegro',
                  'Slovakia', 'Slovenia', 'Spain', 'Svalbard and Jan Mayen Islands', 'Sweden', 'Switzerland', 'Ukraine', 'United Kingdom of Great Britain and Northern Ireland', "United Kingdom",
                  'Yugoslav SFR']

        north_america = ['Anguilla', 'Antigua and Barbuda', 'Aruba', 'Bahamas', 'Barbados', 'Belize', 'Bermuda', 'British Virgin Islands',
                         'Canada', 'Cayman Islands', 'Costa Rica', 'Cuba', 'Dominica', 'Dominican Republic', 'El Salvador', 'Greenland',
                         'Grenada', 'Guadeloupe', 'Guatemala', 'Haiti', 'Honduras', 'Jamaica', 'Martinique', 'Mexico', 'Montserrat',
                         'Netherlands Antilles (former)', 'Nicaragua', 'Panama', 'Puerto Rico', 'Saint Kitts and Nevis', 'Saint Lucia',
                         'Saint Pierre and Miquelon', 'Saint Vincent and the Grenadines', 'Trinidad and Tobago', 'Turks and Caicos Islands',
                         'United States of America', 'United States Virgin Islands']

        south_america = ['Argentina', 'Bolivia (Plurinational State of)', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Falkland Islands (Malvinas)',
                         'French Guyana', 'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela (Bolivarian Republic of)',
                         'South Georgia and the South Sandwich Islands']

        africa = ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cameroon', 'Central African Republic',
                  'Chad', 'Comoros', 'Congo', 'Côte d\'Ivoire', "C?te d'Ivoire", 'Democratic Republic of the Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea',
                  'Eritrea', 'Eswatini', 'Ethiopia', 'Ethiopia PDR', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya', 'Lesotho',
                  'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Mayotte', 'Morocco', 'Mozambique', 'Namibia',
                  'Niger', 'Nigeria', 'Rwanda', 'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa',
                  'South Sudan', 'Sudan', 'Sudan (former)', 'Togo', 'Tunisia', 'Uganda', 'United Republic of Tanzania', 'Zambia', 'Zimbabwe',
                  'Réunion', 'R?union', 'Saint Helena, Ascension and Tristan da Cunha', 'Western Sahara']

        oceania = ['American Samoa', 'Australia', 'Christmas Island', 'Cocos (Keeling) Islands', 'Cook Islands', 'Fiji', 'French Polynesia',
                   'Kiribati', 'Marshall Islands', 'Micronesia (Federated States of)', 'Midway Island', 'Nauru', 'New Caledonia', 'New Zealand',
                   'Niue', 'Norfolk Island', 'Pacific Islands Trust Territory', 'Palau', 'Papua New Guinea', 'Pitcairn', 'Samoa', 'Solomon Islands',
                   'Tokelau', 'Tonga', 'Tuvalu', 'Vanuatu', 'Wallis and Futuna Islands', 'Wake Island']

        antarctica = ['Antarctica', 'French Southern Territories']

        if country in asia:
            return 'Asia'
        elif country in europe:
            return 'Europe'
        elif country in north_america:
            return 'North America'
        elif country in south_america:
            return 'South America'
        elif country in africa:
            return 'Africa'
        elif country in oceania:
            return 'Oceania'
        elif country in antarctica:
            return 'Antarctica'
        else:
            return 'Unknown'

    df = kaggle_temp_yearly.copy()
    df['Continent'] = df['Entity'].apply(assign_continents)
    annual_avg_temp_change = df.groupby('Year')['Value'].mean().dropna()

# Page 2 - Data Visualization - Kaggle Graphs - Surf Temp
if page == pages[2] :
    st.markdown("## Kaggle - Surface Temperature Anomalies ")

    with st.expander("**Graph 01 - Annual Average Temperature Change (Global)**"):
        col1, col2 = st.columns([7,4])  # Adjust the ratio as needed
        with col1:
            plt.figure(figsize=(14, 7))
            plt.plot(annual_avg_temp_change.index, annual_avg_temp_change.values, marker='o', linestyle='-', color='b')
            plt.xlabel('Year')
            plt.ylabel('Average Temperature Change (°C)')
            plt.title('Annual Average Temperature Change (Global)')
            plt.grid(True)
            st.pyplot(plt)
        with col2:
            text = """
            Interpretation:
            
            The graph depicts the average global temperature change from 1960 to 2020, highlighting a clear upward trend. 
            In the earlier years, temperature fluctuations are more balanced, with periods of both warming and cooling that largely offset each other. 
            However, starting in the 1990s, there is a noticeable and consistent rise in average temperatures, which becomes even more pronounced in the last two decades. 
            This pattern indicates significant global warming, likely driven by increased greenhouse gas emissions and other human activities.
            """
            st.markdown(f"""
            <div style="text-align: justify;; margin-top: 20px;">
            <p>{text}</p>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("**Graph 02 - Distribution of Temperature Changes by Continent**"):
        col1, col2 = st.columns([7,4])  # Adjust the ratio as needed
        with col1:
            # Create a boxplot for the temperature change in Celsius per continent
            plt.figure(figsize=(12, 6))
            sns.boxplot(    x='Continent', 
                            y='Value', 
                            data=df)
            plt.title('Distribution of Temperature Changes by Continent')
            plt.xlabel('Continent')
            plt.ylabel('Temperature Change in Celsius')
            plt.grid(True)
            st.pyplot(plt)

        with col2:
            text = """
            Interpretation:
            
            The boxplot highlights the differences in how temperature changes are distributed across continents, revealing insights into regional climate variability and stability. 
            This information is crucial for understanding the impacts of climate change on different parts of the world.
            """
            st.markdown(f"""
            <div style="text-align: justify;; margin-top: 20px;">
            <p>{text}</p>
            </div>
            """, unsafe_allow_html=True)

            text = """
            - **Variability**: Europe and Asia exhibit the highest variability in temperature changes, as evidenced by a wide range of outliers.
            - **Consistency**: Africa shows relatively consistent temperature changes with fewer outliers.
            - **Stability**: Antarctica appears to have the most stable temperature changes with the smallest IQR and fewer outliers.
            - **Median Temperature Changes**: The median temperature changes are fairly similar across most continents, generally hovering around 0.5°C to 1.5°C, except for Antarctica, which has a median around 0°C.
            """
            st.markdown(text)

    with st.expander("**Graph 03 - Mean Temperature Change by Continent in 2020**"):
        col1, col2 = st.columns([7,4])  # Adjust the ratio as needed
        with col1:
            df_2020 = df[df['Year'] == df['Year'].max()]
            continent_avg_2020 = df_2020.groupby('Continent')['Value'].mean().sort_values(ascending=True )
            max_continent = continent_avg_2020.idxmax()

            # Step 4: Plot the horizontal bar graph
            plt.figure(figsize=(10, 6))
            bars = plt.barh(continent_avg_2020.index, continent_avg_2020.values, color='lightblue')

            # Highlight the bar of the continent with the highest temperature change in red
            for bar in bars:
                if bar.get_y() + bar.get_height() / 2 == continent_avg_2020.index.get_loc(max_continent):
                    bar.set_color('red')

            # Step 5: Annotate each bar with the mean value
            for bar in bars:
                plt.text(
                    bar.get_width() + 0.01,   # x-coordinate for the text
                    bar.get_y() + bar.get_height()/2,   # y-coordinate for the text
                    f'{bar.get_width():.2f}',   # text to display (formatted mean value)
                    va='center'   # vertical alignment
                )

            # Step 6: Add labels and title
            plt.xlabel('Mean Temperature Change (2020)')
            plt.title('Mean Temperature Change by Continent in 2020')
            st.pyplot(plt)

        with col2:
            text = """
            Interpretation:

            This bar chart displays the mean temperature change by continent in 2020. 
            Europe experienced the highest increase, exceeding 2 degrees, highlighted by the red bar. 
            Asia and North America had notable increases of around 1.5 degrees, while Oceania, South America, and Africa saw smaller changes between 1.0 and 1.5 degrees.
            Antarctica had the lowest temperature change, slightly above 0.5 degrees. 
            The chart underscores significant regional differences in warming, with Europe being particularly impacted.
            """
            st.markdown(f"""
            <div style="text-align: justify;; margin-top: 20px;">
            <p>{text}</p>
            </div>
            """, unsafe_allow_html=True)

# Page 2 - Data Visualization - OWID Graphs - Surf Temp
if page == pages[2] :
    st.markdown("## OWID - Surface Temperature Anomalies ")
    with st.expander("**Graph 01 - Annual Average Temperature Change (Global)**"):
        col1, col2 = st.columns([7,4])  # Adjust the ratio as needed
        with col1:
            owid_surf_temp_anom.loc[ owid_surf_temp_anom["Entity"]== "Micronesia", "Entity" ] = "Micronesia (country)" 
            owid_surf_temp_anom = pd.merge(owid_surf_temp_anom, owid_country_infos, on=['Entity', "Code"], how='left')

            # Plotting Global  Temp
            annual_avg_temp_change = owid_surf_temp_anom.groupby('Year')['Surface temperature anomaly'].mean().dropna()
            annual_avg_temp_change_1960 = annual_avg_temp_change 
            # Create a figure and axis for the correct plot
            plt.figure(figsize=(15, 7))
            plt.plot(annual_avg_temp_change.index, annual_avg_temp_change.values, linestyle='-', color='b')
            plt.xlabel('Year')
            plt.ylabel('Average Temperature Change (°C)')
            plt.title('Annual Average Temperature Change (Global)')
            plt.grid(True) 
            st.pyplot(plt)
        with col2:
            text = """
            Interpretation:
            
            The graph shows global annual average temperature changes from 1850 to 2017.
            Initially, temperatures fluctuated below the 0°C baseline, indicating cooler years before 1925. 
            A clear upward trend begins around the 1940s, with temperatures rising more sharply after 1975. 
            In recent years, temperatures have frequently exceeded 0.4°C, reaching nearly 1.0°C by the 2020s. 
            The overall trend reflects significant global warming, especially in the past few decades.            """
            st.markdown(f"""
            <div style="text-align: justify;; margin-top: 20px;">
            <p>{text}</p>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("**Graph 02 - Mean Temperature Change by Continent in 2017**"):
        col1, col2 = st.columns([7,4])  # Adjust the ratio as needed
        with col1:
            # PLOTTING  MEAN temperature change per continent 
            # Step 1: Filter the data for the highest Year
            owid_surf_temp_anom_2017 = owid_surf_temp_anom[owid_surf_temp_anom['Year'] == owid_surf_temp_anom['Year'].max()]

            # Step 2: Group by 'Continent' and calculate the mean 'Surface temperature anomaly'
            continent_avg_2017 = owid_surf_temp_anom_2017.groupby('Continent')['Surface temperature anomaly'].mean().sort_values(ascending=True )
            # Step 3: Find the continent with the highest mean value
            max_continent = continent_avg_2017.idxmax()

            # Step 4: Plot the horizontal bar graph
            plt.figure(figsize=(11, 6))
            bars = plt.barh(continent_avg_2017.index, continent_avg_2017.values, color='lightblue')

            # Highlight the bar of the continent with the highest temperature change in red
            for bar in bars:
                if bar.get_y() + bar.get_height() / 2 == continent_avg_2017.index.get_loc(max_continent):
                    bar.set_color('red')

            # Step 5: Annotate each bar with the mean value
            for bar in bars:
                plt.text(
                    bar.get_width() - 0.10,   # x-coordinate for the text
                    bar.get_y() + bar.get_height()/2,   # y-coordinate for the text
                    f'{bar.get_width():.2f}',   # text to display (formatted mean value)
                    va='center'   # vertical alignment
                )

            # Step 6: Add labels and title
            plt.xlabel('Mean Temperature Change (2017)')
            plt.title('Mean Temperature Change by Continent in 2017')
            plt.tight_layout()
            st.pyplot(plt)

        with col2:
            text = """
            Interpretation:
            
            The graph shows the mean temperature change by continent in 2017. 
            Europe had the highest increase at 1.35°C, while Antarctica had the lowest at 0.15°C. 
            All other continents showed intermediate temperature changes, with Africa at 0.96°C and Oceania at 0.55°C.
            """
            st.markdown(f"""
            <div style="text-align: justify;; margin-top: 20px;">
            <p>{text}</p>
            </div>
            """, unsafe_allow_html=True)

# Target Variable Choice
if page == pages[3] :
    st.markdown(
    """
    <style>
    .centered-title {
        font-size: 28px;
        text-align: center;
        border-top: 2px solid black;
        border-bottom: 2px solid black;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.markdown('<h1 class="centered-title">Target Variable Choice</h1>', unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    Kaggle_mean_surf_temp_2022 = pd.read_csv("Aug24_WorldTemp/CS_Kaggle_mean_surf_temp_2022_03.csv")
    Kaggle_mean_surf_temp_NoFlag = pd.read_csv("Aug24_WorldTemp/CS_Kaggle_mean_surf_temp_NoFlag_04.csv")
    owid_surf_temp_anom  = pd.read_csv("Aug24_WorldTemp/CS_owid_surface_temp_anom_countries_02.csv")

    text = """
    For our Machine Learning models, the team had to choose which target variable for the Surface Temperture Anomalies to predict. 
    Given our initial databases, we had the chance to select between three main sources: 

    """
    st.markdown(f"""
    <div style="text-align: justify;; margin-top: 20px;">
    <p>{text}</p>
    </div>
    """, unsafe_allow_html=True)
    text= """
        1. **FAOSTAT Temperature Change - from Kaggle**
        2. **FAOSTAT Temperature Change - from Kaggle - NO Flag** 
        3. **HARDCRUT Information about Surf Temperature, historical data till 2017 - From Our World in Data**
        """
    st.markdown(text)


    with st.expander("Show DB preview"):
        st.markdown("### FAOSTAT Temperature Change - from Kaggle")
        st.data_editor( Kaggle_mean_surf_temp_2022,
                column_config={"Entity" : st.column_config.Column(width="medium"),  "Year": st.column_config.NumberColumn(format="%d")},
                hide_index=True,
                use_container_width=True  # This will make the dataframe occupy the entire width
                )
        st.markdown("### FAOSTAT Temperature Change - from Kaggle - NO Flag")
        st.data_editor( Kaggle_mean_surf_temp_NoFlag,
                column_config={"Entity" : st.column_config.Column(width="medium"),  "Year": st.column_config.NumberColumn(format="%d")},
                hide_index=True,
                use_container_width=True  # This will make the dataframe occupy the entire width
                )

        st.markdown("### HARDCRUT Information about Surf Temperature, historical data till 2017 - From Our World in Data")
        st.data_editor( owid_surf_temp_anom,
                column_config={"Entity" : st.column_config.Column(width="medium"),  "Year": st.column_config.NumberColumn(format="%d")},
                hide_index=True,
                use_container_width=True  # This will make the dataframe occupy the entire width
                )

    col1, col2, col3  = st.columns([1,1,1])  # Adjust the ratio as needed
    with col1:
        st.markdown("**Number of missing values**")
        text = """
        - **Kaggle**: 481 missing values
        - **Kaggle (No Flag)** : 0 missing values
        - **OWID**: 0 missing values
        """
        st.markdown(text)
    with col2:
        st.markdown("**Number of countries in the dataset**")
        text = """
        - **Kaggle**: 247 countries
        - **Kaggle (No Flag)** : 208 countries
        - **OWID**: 199 countries
        """
        st.markdown(text)
    with col3:
        st.markdown("**Time span of data**")
        text = """

        - **Kaggle**: 1961 - 2020
        - **Kaggle (No Flag)** : 1961 - 2019
        - **OWID**: 1850 - 2017
        """
        st.markdown(text)
    with st.expander("Graph Comparison - China"):
        col1, col2 = st.columns([7,4])  # Adjust the ratio as needed
        with col1:
            temp = pd.merge(Kaggle_mean_surf_temp_2022,Kaggle_mean_surf_temp_NoFlag, on=['Entity', "Code", 'Year', 'Continent'],  how='outer')
            temp = temp.drop(columns="Mean_surf_temp_change_kaggle_Noflag_std_dev")
            temp = pd.merge(temp,owid_surf_temp_anom, on=['Entity', "Code", 'Year', 'Continent'],  how='outer')
            plt.figure(figsize=(14,6))

            country_check = temp.loc[temp["Entity"] == "China"]

            sns.lineplot(country_check, x="Year", y="Mean_surf_temp_change_kaggle", errorbar=None, label="Kaggle Mean Temp Change")
            sns.lineplot(country_check, x="Year", y="Mean_surf_temp_change_kaggle_Noflag", errorbar=None, label="Kaggle Mean Temp Change NoFlag")
            sns.lineplot(country_check, x="Year", y="Mean_Surf_temp_anomaly_owid", errorbar="ci", label="OWID Temp Anomaly")

            plt.legend(title="Temperature Data")
            plt.title(' Mean Surface Temperature - Database Comparison - China')
            # Add labels to the axes
            plt.xlabel("Year")
            plt.grid()
            plt.ylabel("Temperature Change (°C)")

            # Display the plot
            st.pyplot(plt)
        with col2:
            text = """
            To determine the most suitable dataset for further testing, we decided to compare the three and evaluate which offered the best starting point for our analysis.
            We compared the mean surface temperature anomaly data across various countries to identify strong justifications for selecting one of the three datasets.
            """
            st.markdown(f"""
            <div style="text-align: justify;; margin-top: 20px;">
            <p>{text}</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("**Interpretation:**")
            text = """In the graph on the left, the mean surface temperature anomaly is plotted over time. 
            It's clear that the OWID database (HADCRUT information on surface temperature, with historical data until 2017) 
            provides data starting from 1850, while the other two datasets begin in 1960. 
            To make a fair comparison between the datasets, we chose to focus only on the years they all had in common.
            """
            st.markdown(f"""
            <div style="text-align: justify;; margin-top: 20px;">
            {text}
            </div>
            """, unsafe_allow_html=True)
    with st.expander("Graph Comparison - Germany"):
        col1, col2 = st.columns([7,4])  # Adjust the ratio as needed
        with col1:
            plt.figure(figsize=(14,6))
            country_check = temp.loc[(temp["Entity"] == "Germany") & (temp["Year"] >= 1960)]

            sns.lineplot(country_check, x="Year", y="Mean_surf_temp_change_kaggle", errorbar=None, label="Kaggle Mean Temp Change")
            sns.lineplot(country_check, x="Year", y="Mean_surf_temp_change_kaggle_Noflag", errorbar=None, label="Kaggle Mean Temp Change NoFlag")
            sns.lineplot(country_check, x="Year", y="Mean_Surf_temp_anomaly_owid", errorbar="ci", label="OWID Temp Anomaly")

            plt.legend(title="Temperature Data")
            plt.title(' Mean Surface Temperature - Database Comparison - Germany')
            # Add labels to the axes
            plt.xlabel("Year")
            plt.grid()
            plt.ylabel("Temperature Change (°C)")
            st.pyplot(plt)
        with col2:
            st.markdown("**Interpretation:**")
            text = """
            To support this hypothesis, the graph on the left shows the data for Germany.
            Once again, the two Kaggle datasets display identical values. 
            Additionally, it’s worth noting that the "No Flag" dataset is missing values for 2020.
            """
            st.markdown(f"""
            <div style="text-align: justify;; margin-top: 20px;">
            <p>{text}</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("### Final Decision")
    text = """
    To ensure a good range of the datasets we used 2 merges with one focused on the **"HADCRUT Information about Surface Temperature, historical data until 2017"** and the second one the **“Kaggle Temperature change (No Flag)”**
    dataset. 
    For consistency between the explanatory variables and the target variable, we chose a similar time span for both.
    """
    st.markdown(text)
    
# Preprocessing 
if page == pages[4] :
    st.markdown(
        """
        <style>
        .centered-title {
            font-size: 28px;
            text-align: center;
            border-top: 2px solid black;
            border-bottom: 2px solid black;
            padding: 10px;
        }
        .blue-box {
            border: 1px solid #d6d6d6;
            padding: 10px;
            border-radius: 5px;
            background-color: #e0f7fa;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<h1 class="centered-title">Pre-processing and Data Cleaning</h1>', unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    with st.expander("Datasets"):
        st.markdown("""
        - Nasa Dataset: Zonal Annual Temperature Anomalies
        - FAOSTAT Temperature Change - from Kaggle
        - FAOSTAT Temperature Change - from Kaggle - NO Flag
        - CO2 and Greenhouse Gas Emissions database by Our World in Data - from GitHub
        - HARDCRUT Information about Surf Temperature, historical data till 2017 - From Our World in Data
        - World Regions - According to Our World in Data
        - Annual CO2 Emissions by region - from Our World in Data
        """)

    st.markdown("""
    ### Pre-processing Steps
     1. **Dataset decisions:**

         - First Dataset: Base: CO2 and Greenhouse Gas Emissions
         - Second Dataset Base: FAOSTAT Temperature change NOFLAG - from Kaggle
    

    2. **Data Merge options:**

        **First Dataset:**
            - FAOSTAT Temperature Change
            - FAOSTAT Temperature Change NO Flag
            - HARDCRUT Information about Surf Temperature (historical data till 2017)

        **Second Dataset:**
            - World Regions - According to Our World in Data
            - CO2 and Greenhouse Gas Emissions

    3. **Preparation:**

        First & Second Dataset:

        - Standardize column names and checking for common entries between datasets
        - checking for missing values and duplicates
        - understanding the datatypes and dataframe

        **First Dataset:**

        - merging all dataframes based on country and year
        - choosing a timespan (1850+)
        - choosing a second timespan (1960+)
        - drop of unrealated columns with industrie categorisation and high NaN values
        - missing values: numerical variables, no categorical
        - 1. using KNN imputer for Nan values
        - 2. NaN values were dropped
    
        **Second Dataset:**

        - structure change of base dataset to create a "year" and "temp-change" column
        - merging all dataframes (adding only choosen columns from each dataset to the base)
        - drop of unused columns after merge
        - timespan 1961+ (Base timespan)
        - missing values: numerical variables, no categorical, nothing in temperature data
        - missing values: only in Greenhouse dataset for small island countries (no CO2 informations)
        - we drop the few NaN values of this countries

    4. **Final Datasets:**

        **Methods used to work with categorical values:**

        - factorize methode
        - get_dummies methode
        
        **First Dataset:**
        - df_project_dropped_dummies.csv
        - df_project_dropped_factorized.csv
        - df_project_KNN_dummies.csv
        - df_project_KNN_factorized.csv
        
        **Second Dataset:**
        - Data_World_temperature.csv

    This pre-processing and merging of datasets ensured that our data was ready for the modeling process.
    """)


# Modelling 
if page == pages[5] :

    st.markdown(
        """
        <style>
        .centered-title {
            font-size: 28px;
            text-align: center;
            border-top: 2px solid black;
            border-bottom: 2px solid black;
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
     )
    st.markdown('<h1 class="centered-title">Machine Learning Model Selection</h1>', unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    The objective of this analysis was to identify the best-performing machine learning model for predicting surface temperature anomalies using different regression techniques.
    We evaluated and compared the performance of six  models:

    - **Linear**
    - **Decision Tree**
    - **Random Forest**
    - **Lasso**
    - **LassoCV**
    - **Ridge**

    ### Evaluation Metrics

    - **R² Score (Coefficient of Determination)**:
        Measures the proportion of variance in the dependent variable that can be predicted from the independent variables. Higher R² values indicate better performance.
    - **Mean Absolute Error (MAE)**:
        Represents the average magnitude of the prediction errors, offering an easy-to-understand measure of accuracy. Lower MAE values indicate better model performance.
    - **Mean Squared Error (MSE)**:
        Represents the average of the squared prediction errors, placing more weight on larger errors. A smaller MSE means a better model.
    - **Root Mean Squared Error (RMSE)**:
        Represents the square root of MSE, providing an error measure in the same units as the original data. Lower RMSE values are better, and it's more sensitive to larger errors than MAE    
    """)
    
# Modelling - First Dataset
if page == pages[5] :
  with st.expander("**First Dataset**"):
    st.markdown("""
    Given the large number of models we tested (approximately 50), we decided to focus on the most noteworthy ones.
    Our primary criterion for selecting which models to analyze in greater detail was based on the highest R² score achieved on the test set.
    This helped us identify the models that best captured the underlying patterns and provided the most accurate predictions.

    Nonetheless, we chose to present the scores of all the models we created in a matrix table.
    This table provides a comprehensive overview of each model’s performance, allowing for easy comparison of the evaluation metrics across all models, even those not selected for in-depth analysis.
    """)

    data = {
        'Model': [
            'Random Forest (get dummies)', 'Random Forest (factorized)',
            'Random Forest (get dummies, knn)', 'Random Forest (factorized, knn)',
            'Decision Tree (factorized)', 'Decision Tree (get dummies)',
            'Decision Tree (get dummies, knn)', 'Decision Tree (factorized, knn)',
            'Linear (get dummies)', 'Ridge (get dummies)',
            'Linear (factorized)', 'Ridge (factorized)',
            'Ridge (get dummies, knn)', 'Linear (get dummies, knn)',
            'Linear (factorized, knn)', 'Ridge (factorized, knn)',
            'Lasso (get dummies)', 'Lasso (factorized)',
            'Lasso (factorized, knn)', 'Lasso (get dummies, knn)',
            'LassoCV (factorized, knn)', 'LassoCV (get dummies, knn)',
            'LassoCV (factorized)', 'LassoCV (get dummies)'
        ],

    'R2 Train': [
            0.891, 0.879, 0.884, 0.883,
            0.542, 0.575, 0.670, 0.555,
            0.378, 0.377,
            0.331, 0.331,
            0.385, 0.386,
            0.338, 0.337,
            0.323, 0.323,
            0.327, 0.327,
            0.002, 0.002,
            0.003, 0.003
        ],
        'R2 Test': [
            0.630, 0.615, 0.592, 0.579,
            0.515, 0.513, 0.491, 0.483,
            0.367, 0.364,
            0.343, 0.342,
            0.336, 0.335,
            0.335, 0.334,
            0.331, 0.331,
            0.326, 0.326,
            0.003, 0.003,
            0.000, 0.000
        ],
        'MAE Test': [
            0.256, 0.262, 0.259, 0.267,
            0.302, 0.298, 0.294, 0.301,
            0.339, 0.340,
            0.345, 0.345,
            0.346, 0.346,
            0.349, 0.349,
            0.346, 0.346,
            0.351, 0.351,
            0.432, 0.432,
            0.437, 0.437
        ],
        'MSE Test': [
            0.123, 0.128, 0.128, 0.132,
            0.162, 0.162, 0.160, 0.162,
            0.211, 0.212,
            0.219, 0.219,
            0.209, 0.209,
            0.209, 0.209,
            0.223, 0.223,
            0.212, 0.212,
            0.313, 0.313,
            0.333, 0.333
        ],
        'RMSE Test': [
            0.351, 0.358, 0.358, 0.363,
            0.402, 0.403, 0.400, 0.403,
            0.460, 0.461,
            0.468, 0.468,
            0.457, 0.457,
            0.457, 0.457,
            0.472, 0.472,
            0.460, 0.460,
            0.560, 0.560,
            0.577, 0.577
        ]
    }

    # Create a DataFrame
    metrics_df = pd.DataFrame(data)

    # Streamlit app
    st.title("Models with timespan 1960-2017")

    # Display
    st.dataframe(metrics_df.style.highlight_max(axis=0), use_container_width=True)


    data2 = {
        'Model': [
            'Random Forest (get dummies)', 'Random Forest (factorized)',
            'Random Forest (get dummies, knn)', 'Random Forest (factorized, knn)',
            'Decision Tree (factorized)', 'Decision Tree (get dummies)',
            'Decision Tree (factorized, knn)', 'Linear (get dummies)',
            'Ridge (get dummies)', 'Decision Tree (get dummies, knn)',
            'Linear (factorized)', 'Ridge (factorized)',
            'Lasso (get dummies)', 'Lasso (factorized)',
            'Ridge (get dummies, knn)', 'Linear (get dummies, knn)',
            'Linear (factorized, knn)', 'Lasso (factorized, knn)',
            'Lasso (get dummies, knn)', 'Ridge (factorized, knn)',
            'LassoCV (factorized, knn)', 'LassoCV (get dummies, knn)',
            'LassoCV (factorized)', 'LassoCV (get dummies)'
        ],
        'R2 Train': [
            0.810, 0.801, 0.714, 0.740,
            0.371, 0.340, 0.293, 0.279,
            0.278, 0.269,
            0.246, 0.245,
            0.242, 0.242,
            0.214, 0.214,
            0.185, 0.183,
            0.183, 0.184,
            0.004, 0.004,
            0.003, 0.003
        ],
        'R2 Test': [
            0.500, 0.481, 0.473, 0.458,
            0.346, 0.334, 0.293, 0.273,
            0.273, 0.269,
            0.247, 0.247,
            0.244, 0.244,
            0.192, 0.191,
            0.180, 0.179,
            0.179, 0.179,
            0.004, 0.004,
            0.003, 0.003
        ],
        'MAE Test': [
            0.328, 0.335, 0.334, 0.340,
            0.382, 0.387, 0.396, 0.410,
            0.410, 0.402,
            0.417, 0.417,
            0.419, 0.419,
            0.429, 0.429,
            0.432, 0.432,
            0.432, 0.432,
            0.483, 0.483,
            0.486, 0.486
        ],
        'MSE Test': [
            0.218, 0.226, 0.228, 0.235,
            0.284, 0.289, 0.306, 0.316,
            0.316, 0.316,
            0.327, 0.327,
            0.329, 0.329,
            0.350, 0.350,
            0.355, 0.355,
            0.355, 0.355,
            0.431, 0.431,
            0.433, 0.433
        ],
        'RMSE Test': [
            0.466, 0.475, 0.478, 0.484,
            0.533, 0.538, 0.553, 0.562,
            0.562, 0.563,
            0.572, 0.572,
            0.573, 0.573,
            0.592, 0.592,
            0.596, 0.596,
            0.596, 0.596,
            0.657, 0.657,
            0.658, 0.658
        ]
    }

    metrics_df2 = pd.DataFrame(data2)

    st.title("Models with timespan 1850-2017")

    st.dataframe(metrics_df2.style.highlight_max(axis=0), use_container_width=True)

    st.markdown("""
    **Key findings**: 
        - The R² score is generally lower in models using the larger dataset.
        - The Random Forest Regressor consistently performs the best across all tested models.
        - The Lasso Regressor consistently performs the worst among all models.
    """)

# Modelling - Second Dataset
if page == pages[5] :
  with st.expander("**Second Dataset**"):
    st.markdown("""
    In the first dataset we tested the different Models and variables, so here we concentrate on the best dataset for each temperature change.

    After training the Different models using the get_dummies method and dropping missing values, the following metrics were obtained:
    """)
    data3 = {
        'Model': ['Forest', 'Ridge', 'Decision Tree', 'Linear', 'Lasso', 'LassoCV'],
        'Temp Change - MSE': [0.134, 0.199, 0.217, 0.219, 0.232, 0.428],
        'Temp Change - MAE': [0.261, 0.337, 0.341, 0.353, 0.364, 0.526],
        'Temp Change - R2': [0.692, 0.544, 0.503, 0.497, 0.467, 0.018],
        'Temp Change - MSE**(1/2)': [0.366, 0.446, 0.465, 0.468, 0.482, 0.654],

        'Temp Change GHG - MSE': [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        'Temp Change GHG - MAE': [0.000, 0.001, 0.000, 0.002, 0.009, 0.004],
        'Temp Change GHG - R2': [0.999, 0.983, 0.997, 0.924, 0.000, 0.736],
        'Temp Change GHG - MSE**(1/2)': [0.001, 0.003, 0.001, 0.006, 0.022, 0.011],

        'Temp Change CO2 - MSE': [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        'Temp Change CO2 - MAE': [0.000, 0.001, 0.000, 0.002, 0.006, 0.003],
        'Temp Change CO2 - R2': [0.998, 0.978, 0.998, 0.976, 0.000, 0.642],
        'Temp Change CO2 - MSE**(1/2)': [0.001, 0.002, 0.001, 0.002, 0.016, 0.009],
    }

    metrics_df3= pd.DataFrame(data3)

    st.title("Model Performance Metrics for Temperature Change")

    # Display
    st.dataframe(metrics_df3.style.highlight_max(axis=0), use_container_width=True)

# Modelling - Final Comments
if page == pages[5] :
    st.markdown(
                """
                <style>
                .centered-title {
                    font-size: 28px;
                    text-align: center;
                    border-top: 2px solid black;
                    border-bottom: 2px solid black;
                    padding: 10px;
                }
                </style>
                """,
                unsafe_allow_html=True,
             )
    st.markdown('<h1 class="centered-title">Dataset Evaluation</h1>', unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""The Large number of models allowed a nice overview of all possible options and a good Performance Evaluation.
    In both datasets we see a similar outcome with the **get_dummies** method as best performing and the **Random Forest Regressor** as best model.
    In the second dataset we concentrate more on the best dataset for each temperature change.
    """)
    #GRAPH 01
    st.markdown("""
    ### Target Variable - temp_change
    - This dataset is from Kaggle and the base is the NASA-GISS Dataset.
    - This temperature change contains all measured data around the world and not only static values, since it will react to natural phenomena as well.
    - Linear Regression  (R² = 0.497), Decision Tree (R² = 0.503) and Random Forest  (R² = 0.692) work for this dataset best, with Random Forest best overall.
    - Since this data is not constant and will react for all temperature changes, even natural phenomena, it has a wide range of data.
    """)
    # image from file 01 
    st.markdown(
        """
        <div style='text-align: center;'>
            <img src='https://github.com/Carmine137/AUG24_world_temperature/blob/main/temp_change-predict.png?raw=true' width='800'/>
            <p style='font-size: 18px; color: gray;'>Figure 1: Scatter Plot Predicted vs Actual Values-temp_change-predict</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # GRAPH 02 
    st.markdown("""
    ### Target Variable - temperature_change_from_ghg
    - This Database uses the tempchange_ghg column as the target variable.
    - This dataset is from github and focuses on temperature changes from Greenhouse emissions.
    - Most Greenhouse emissions are more constant and change slowly but steady over time.
    
    Decision Tree (R² = 0.997) and Random Forest (R² = 0.999) work for this dataset best, with close followup on Ridge and Linear Regression.
    """)
    # image from file 02
    st.markdown(
        """
        <div style='text-align: center;'>
            <img src='https://github.com/Carmine137/AUG24_world_temperature/blob/main/temperature_change_from_ghg-predict.png?raw=true' width='800'/>
            <p style='font-size: 18px; color: gray;'>Figure 2: Scatter Plot Predicted vs Actual Values-temperature_change_from_ghg-predict</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    #GRAPH 03
    st.markdown("""
    ### Target Variable - temperature_change_from_co2
    
    - This Database uses the tempchange_co2 column as the target variable.
    - This dataset is from github and focuses on temperature changes from carbon dioxide emissions.
    - The temperature change is like the greenhouse emissions more constant and changes only slowly but steadily over time.

    Decision Tree (R² = 0.998) and Random Forest  (R² = 0.998) work for this dataset best, with close followup on Ridge and Linear Regression.
    """)
    # image from file 03
    st.markdown(
        """
        <div style='text-align: center;'>
            <img src='https://github.com/Carmine137/AUG24_world_temperature/blob/main/temperature_change_from_co2-predict.png?raw=true' width='800'/>
            <p style='font-size: 18px; color: gray;'>Figure 3: Scatter Plot Predicted vs Actual Values-temperature_change_from_co2-predict</p>
        </div>
        """, 
        unsafe_allow_html=True
    )


    st.markdown("""
    - For each dataset we can see now, how the same model and method works on diffrent datasets with the same structure.
    - To find the best model and methode for a dataset, it is realy important to test and check for each dataset.
    - The Random Forest workes best here, since it is good for complex data and performs with a nice accuracy.
    
    **Important:**
     For performance reason and since both datasets come to the same conclusion, we will keep the grafic analysis part only for the second dataset,
    since it shows best the diffrences in values of each dataset.
    """)

    
if page == pages[6] :
    #add ur predicition here
    st.markdown(
        """
        <style>
        .centered-title {
            font-size: 28px;
            text-align: center;
            border-top: 2px solid black;
            border-bottom: 2px solid black;
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
     )
    st.markdown('<h1 class="centered-title">Predictions</h1>', unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([4,7,1])  # Adjust the ratio as needed
    with col1:
        # PREDICTION PART
        @st.cache_resource

        # Loading the model for the prediction
        def load_model(model_filename, zip_path=None):
            # Try loading the model directly if zip file is not provided
            if zip_path is None:
                # Load from the pickle file directly
                with open(model_filename, 'rb') as template_model:
                    model = pickle.load(template_model)
            else:
                # Load from the zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    with zip_ref.open(model_filename) as model_file:
                        model = pickle.load(model_file)
            return model
            
        model_filename = 'CS_RandomForestModel_get_dummies.pkl'
        zip_file_path = 'Aug24_WorldTemp/Zip_Model.zip'
        
        # Check if the .pkl file exists and load directly, otherwise try loading from zip
        if os.path.exists(model_filename):
            RandomForest_Model = load_model(model_filename)
        else:
            RandomForest_Model = load_model(model_filename, zip_file_path)

        @st.cache_resource

        def load_min_max():
            with open('Aug24_WorldTemp/CS_feature_min_max_dummies.json', 'r') as json_file:
                min_max_dict = json.load(json_file)
            return min_max_dict

        min_max_dict = load_min_max()

        #Creating the Exploratory Variables
        model_exploratory_variables = []


        #Creating the sliders
        for feature, limits in min_max_dict.items():
            exploratory_variable = st.slider(
                f"{feature}", 
                float(limits['min']), 
                float(limits['max']), 
                float((limits['min'] + limits['max']) / 2)
            )
            model_exploratory_variables.append(exploratory_variable)


        # Loading the model for the prediction
        exploratory_variables = np.array([model_exploratory_variables])

        num_values = [
            'Year',
            'Population',
            'temperature_change_from_co2',
            'temperature_change_from_ch4',
            'temperature_change_from_n2o',
            'temperature_change_from_ghg',
            'co2',
            'co2_including_luc',
            'land_use_change_co2']

        model_exploratory_variables_dataframe = pd.DataFrame(exploratory_variables, columns=num_values)


        # Loading the Encoding for Continents and Countries
        with open('Aug24_WorldTemp/CS_target_encoding_dummies_Continent.json') as f:
            continent_encodings = json.load(f)

        with open("Aug24_WorldTemp/CS_target_encoding_dummies_Entity.json") as f:
            entity_encodings = json.load(f)

        # Load the continent-entity match dataframe
        mapping_df = pd.read_csv("Aug24_WorldTemp/continent_Entity_Match.csv")

    with col2:
        # Select continent and filter corresponding countries
        continent = st.selectbox("Select a Continent", list(continent_encodings.keys()))
        filtered_countries = mapping_df[mapping_df['Continent'] == continent]['Entity']
        country = st.selectbox("Select a Country", filtered_countries)

        # Get the one-hot encoded columns for the selected continent and country
        encoded_continent_columns = continent_encodings[continent]
        encoded_entity_columns = entity_encodings[country]

        encoded_columns = [
            'Entity_Afghanistan',
            'Entity_Albania',
            'Entity_Algeria',
            'Entity_Andorra',
            'Entity_Angola',
            'Entity_Antigua and Barbuda',
            'Entity_Argentina',
            'Entity_Armenia',
            'Entity_Australia',
            'Entity_Austria',
            'Entity_Azerbaijan',
            'Entity_Bahamas',
            'Entity_Bahrain',
            'Entity_Bangladesh',
            'Entity_Barbados',
            'Entity_Belarus',
            'Entity_Belgium',
            'Entity_Belize',
            'Entity_Benin',
            'Entity_Bhutan',
            'Entity_Bolivia',
            'Entity_Bosnia and Herzegovina',
            'Entity_Botswana',
            'Entity_Brazil',
            'Entity_Brunei',
            'Entity_Bulgaria',
            'Entity_Burkina Faso',
            'Entity_Burundi',
            'Entity_Cambodia',
            'Entity_Cameroon',
            'Entity_Canada',
            'Entity_Cape Verde',
            'Entity_Central African Republic',
            'Entity_Chad',
            'Entity_Chile',
            'Entity_China',
            'Entity_Colombia',
            'Entity_Comoros',
            'Entity_Congo',
            'Entity_Costa Rica',
            "Entity_Cote d'Ivoire",
            'Entity_Croatia',
            'Entity_Cuba',
            'Entity_Cyprus',
            'Entity_Czechia',
            'Entity_Democratic Republic of Congo',
            'Entity_Denmark',
            'Entity_Djibouti',
            'Entity_Dominica',
            'Entity_Dominican Republic',
            'Entity_East Timor',
            'Entity_Ecuador',
            'Entity_Egypt',
            'Entity_El Salvador',
            'Entity_Equatorial Guinea',
            'Entity_Eritrea',
            'Entity_Estonia',
            'Entity_Eswatini',
            'Entity_Ethiopia',
            'Entity_Fiji',
            'Entity_Finland',
            'Entity_France',
            'Entity_Gabon',
            'Entity_Gambia',
            'Entity_Georgia',
            'Entity_Germany',
            'Entity_Ghana',
            'Entity_Greece',
            'Entity_Grenada',
            'Entity_Guatemala',
            'Entity_Guinea',
            'Entity_Guinea-Bissau',
            'Entity_Guyana',
            'Entity_Haiti',
            'Entity_Honduras',
            'Entity_Hungary',
            'Entity_Iceland',
            'Entity_India',
            'Entity_Indonesia',
            'Entity_Iran',
            'Entity_Iraq',
            'Entity_Ireland',
            'Entity_Israel',
            'Entity_Italy',
            'Entity_Jamaica',
            'Entity_Japan',
            'Entity_Jordan',
            'Entity_Kazakhstan',
            'Entity_Kenya',
            'Entity_Kiribati',
            'Entity_Kuwait',
            'Entity_Kyrgyzstan',
            'Entity_Laos',
            'Entity_Latvia',
            'Entity_Lebanon',
            'Entity_Lesotho',
            'Entity_Liberia',
            'Entity_Libya',
            'Entity_Liechtenstein',
            'Entity_Lithuania',
            'Entity_Luxembourg',
            'Entity_Madagascar',
            'Entity_Malawi',
            'Entity_Malaysia',
            'Entity_Maldives',
            'Entity_Mali',
            'Entity_Malta',
            'Entity_Marshall Islands',
            'Entity_Mauritania',
            'Entity_Mauritius',
            'Entity_Mexico',
            'Entity_Micronesia (country)',
            'Entity_Moldova',
            'Entity_Mongolia',
            'Entity_Montenegro',
            'Entity_Morocco',
            'Entity_Mozambique',
            'Entity_Myanmar',
            'Entity_Namibia',
            'Entity_Nauru',
            'Entity_Nepal',
            'Entity_Netherlands',
            'Entity_New Zealand',
            'Entity_Nicaragua',
            'Entity_Niger',
            'Entity_Nigeria',
            'Entity_North Korea',
            'Entity_North Macedonia',
            'Entity_Norway',
            'Entity_Oman',
            'Entity_Pakistan',
            'Entity_Palau',
            'Entity_Panama',
            'Entity_Papua New Guinea',
            'Entity_Paraguay',
            'Entity_Peru',
            'Entity_Philippines',
            'Entity_Poland',
            'Entity_Portugal',
            'Entity_Qatar',
            'Entity_Romania',
            'Entity_Russia',
            'Entity_Rwanda',
            'Entity_Saint Kitts and Nevis',
            'Entity_Saint Lucia',
            'Entity_Saint Vincent and the Grenadines',
            'Entity_Samoa',
            'Entity_Sao Tome and Principe',
            'Entity_Saudi Arabia',
            'Entity_Senegal',
            'Entity_Serbia',
            'Entity_Seychelles',
            'Entity_Sierra Leone',
            'Entity_Singapore',
            'Entity_Slovakia',
            'Entity_Slovenia',
            'Entity_Solomon Islands',
            'Entity_Somalia',
            'Entity_South Africa',
            'Entity_South Korea',
            'Entity_South Sudan',
            'Entity_Spain',
            'Entity_Sri Lanka',
            'Entity_Sudan',
            'Entity_Suriname',
            'Entity_Sweden',
            'Entity_Switzerland',
            'Entity_Syria',
            'Entity_Tajikistan',
            'Entity_Tanzania',
            'Entity_Thailand',
            'Entity_Togo',
            'Entity_Tonga',
            'Entity_Trinidad and Tobago',
            'Entity_Tunisia',
            'Entity_Turkey',
            'Entity_Turkmenistan',
            'Entity_Tuvalu',
            'Entity_Uganda',
            'Entity_Ukraine',
            'Entity_United Arab Emirates',
            'Entity_United Kingdom',
            'Entity_United States',
            'Entity_Uruguay',
            'Entity_Uzbekistan',
            'Entity_Vanuatu',
            'Entity_Venezuela',
            'Entity_Vietnam',
            'Entity_Yemen',
            'Entity_Zambia',
            'Entity_Zimbabwe',
            'Continent_Africa',
            'Continent_Asia',
            'Continent_Europe',
            'Continent_North America',
            'Continent_Oceania',
            'Continent_South America'
        ]

        continent_coutry_vector = np.zeros(len(encoded_columns))

        # Set the selected continent and country to 1 in the feature vector
        for col in encoded_continent_columns + encoded_entity_columns:
            if col in encoded_columns:
                continent_coutry_vector[encoded_columns.index(col)] = 1
        continent_coutry_vector = continent_coutry_vector.reshape(1, -1)

        continent_coutry_dataframe = pd.DataFrame(continent_coutry_vector, columns=encoded_columns)

        # Concatenate the DataFrames along the columns (axis=1)
        merged_dataframe = pd.concat([model_exploratory_variables_dataframe, continent_coutry_dataframe], axis=1)

        predicted_value = RandomForest_Model.predict(merged_dataframe)[0] # Assuming a single prediction
        st.markdown(
        f"<p style='font-size:24px; font-weight:bold;'>The Predicted temperature is : {predicted_value}</p>", 
        unsafe_allow_html=True
        )

if page == pages[7] :
    st.markdown(
    """
    <style>
    .centered-title {
        font-size: 28px;
        text-align: center;
        border-top: 2px solid black;
        border-bottom: 2px solid black;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.markdown('<h1 class="centered-title">Conclusions</h1>', unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
    ## Conclusion
    **Random Forest Regressor** outperformed both the Decision Tree and Linear Regression models, demonstrating its ability to better generalize and provide more accurate predictions for this dataset.
    The Random Forest's superior performance suggests that it is well-suited for capturing the underlying patterns in the data, benefiting from its ensemble nature and the use of multiple decision trees to improve predictive accuracy.
    Moving forward, further tuning of the Random Forest model's hyperparameters or exploring other ensemble methods could potentially yield even better results.
    """)


