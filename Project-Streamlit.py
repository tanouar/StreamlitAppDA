
#Intro
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def wide_space_default():
    st.set_page_config(layout="wide")
wide_space_default()

# Main Code
github_co2_data = pd.read_csv("owid-co2-data.csv")
github_explanation = pd.read_csv("owid-co2-codebook.csv")
owid_surf_temp_anom = pd.read_csv("OWID_01_Surface_Temp_anomaly_historical_data.csv")
kaggle_temp_change_NOFLAG = pd.read_csv("Environment_Temperature_change_E_All_Data_NOFLAG.csv", encoding='cp1252')
owid_country_infos = pd.read_csv("OWID_02_CountryInfos.csv")
kaggle_temp_change_1 = pd.read_csv("FAOSTAT_data_1-10-2022_part1.csv")
kaggle_temp_change_2 = pd.read_csv("FAOSTAT_data_1-10-2022_part1.csv")
kaggle_temp_change = pd.concat([kaggle_temp_change_1, kaggle_temp_change_2], ignore_index=True)



#Creating Main Structure
st.sidebar.title("Summary")
pages=["Introduction", "Data Exploration", "Data Vizualization","Target Variable Choice", "Modelling", "Prediction", "Conclusion"]
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
    st.title("DataScientest - Wolrd Temperature Project - AUG24")
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
    st.header("Data Exploration")
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

# Page 2 - Data Visualization
if page == pages[2] :
    st.header("Data Visualization")
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


# Page 2 - Data Visualization
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

# Page 2 - Data Visualization
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

# Page 2 - Data Visualization
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

if page == pages[3] :
    Kaggle_mean_surf_temp_2022 = pd.read_csv("CS_Kaggle_mean_surf_temp_2022_03.csv")
    Kaggle_mean_surf_temp_NoFlag = pd.read_csv("CS_Kaggle_mean_surf_temp_NoFlag_04.csv")
    owid_surf_temp_anom  = pd.read_csv("CS_owid_surface_temp_anom_countries_02.csv")
    st.header("Target Variable Choice")
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
# Continuare aggiustando il grafico delle temperature, e poi resta solo la parte del modello e delle prediction
