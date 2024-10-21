import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
import math
import io
import plotly.express as px
import plotly.graph_objects as go

st.markdown(
    """
    <style>
    .css-1d391kg {
        margin-top: -30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Main Menu")

# Sidebar navigation
pages = [
    "🏠 Home",
    "📖 Introduction",
    "🔬 Exploration Analysis - NASA",
    "🌍 Exploration Analysis - OWID",
    "📉 Exploration Analysis - STA",
    "📊 Exploration Analysis - FAO",
    "🛠️ Modelling Preparation",
    "🤖 Machine Learning Models",
    "📈 Time-series modeling with SARIMA",
    "🔮 Prediction",
    "📌 Conclusion",
    "👥 Credits"
]
page = st.sidebar.radio("", pages)  

st.sidebar.markdown(
    """
    - **Course**: Data Analyst
    - **Instructor**: Tarik Anouar
    - **Date**: 11 June 2024
    - **Team Members**:
        - Desireé Jörke
        - Manasi Deshpande
        - Fiona Murphy
    """
)

#########################################################################################################################################################################################################################
if page == '🏠 Home':
  
  
# Your Streamlit app content
  st.title("World Temperature Analysis")

  # Add the image to the home page
  st.image("world.png",  use_column_width=True)


# Inject custom CSS
  st.markdown(
    """
    <style>
    .reportview-container {
        background: black;
        color: white;
    }
    .sidebar .sidebar-content {
        background: black;
    }
    </style>
    """,
    unsafe_allow_html=True
 )
   
#########################################################################################################################################################################################################################
if page == '📖 Introduction':
    st.write("## World Temperature: Effects of Greenhouse Gases on Global Temperatures")
    
    st.markdown("""
    <div style="text-align: justify;">
    <p><strong>Understanding what impacts our planet's temperature changes over time is vital for understanding the dynamics of climate change.</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="border: 1px solid #d6d6d6; padding: 10px; border-radius: 5px; background-color: #e0f7fa; margin-bottom: 20px;">
    <p>The goal of our project is to analyze the relationship between rising greenhouse gas emissions and their effect on global temperatures. 
    This project dives into historical temperature records to uncover trends and patterns, using data from 
    <a href="https://www.fao.org/faostat/en/#data/ET/metadata" target="_blank">FAO</a>, 
    <a href="https://data.giss.nasa.gov/gistemp/a" target="_blank">NASA</a>, and 
    <a href="https://ourworldindata.org/explorers/co2?facet=none&Gas+or+Warming=CO%E2%82%82&Accounting=Territorial&Fuel+or+Land+Use+Change=All+fossil+emissions&Count=Per+country&country=CHN~USA~IND~GBR~OWID_WRL" target="_blank">'Our World in Data'</a>.
    </p>
    <p>Through careful analysis, we want to understand how global warming has evolved over centuries and decades. We'll start by carefully looking at temperature data, going from the past to the present to reveal how temperatures have changed across different parts of the world over time.</p>
    <p>We will highlight this data exploration in further detail in the next steps.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: justify; margin-top: 20px;">
    <p><strong>Steps taken throughout this project:</strong></p>
    <p>- Analysis of various datasets to identify global patterns</p>
    <p>- Investigate relationships between temperature and factors like GDP, population, and CO2</p>
    <p>- Development of predictive models for forecasting temperature changes</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center;">
        <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExY29hNGFibnZuZzhqOWZvcTFhbnJlemZha2k1OGZhcTBubTVldGRmcCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/OBWPNNz0xLm3daysZS/giphy-downsized-large.gif" alt="Global Temperature Anomaly">
    </div>
    """, unsafe_allow_html=True)


####################################################################################################################################################################################################################

import pickle
import gzip
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io
import math
from scipy.stats import linregress

# NASA Exploration
if page == "🔬 Exploration Analysis - NASA":
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("nasa_x2.png", width=120)
    
    with col2:
        st.markdown("""
        <div style='display: flex; align-items: center; height: 120px;'>
            <h3 style='margin: 0;'>Exploration Analysis - NASA </h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Introduction Section 
    st.markdown(
        """
        <style>
        .intro-box {
            border: 2px solid #0B3D91; 
            border-radius: 10px;
            background-color: #f9f9f9;
            padding: 15px;
            margin: 15px 0;
            position: relative;
            display: flex; 
            align-items: center;
            justify-content: space-between;
        }
        .intro-header {
            color: #0B3D91; 
            font-size: 22px;
            font-weight: bold;
        }
        .intro-text {
            font-size: 16px;
            line-height: 1.6;
            flex: 1;
        }
        </style>
        <div class="intro-box">
            <div class="intro-text">
                <div class="intro-header">Background, History and Updates</div>
                The GISS Surface Temperature Analysis version 4 - the GISTEMP v4 - is an estimate of global surface temperature change. Graphs and tables are updated around the middle of every month using current data files from NOAA GHCN v4 (meteorological stations) and also from ERSST v5 (ocean areas), combined as described in our publications Hansen et al. (2010) and Lenssen et al. (2019). These updated files incorporate reports for the previous month and also late reports and corrections for earlier months. Temperature change indicates deviations from the typical or expected temperature for a specific location and time. Tables of Global and Hemispheric Monthly Means and also Zonal Annual Means are available. We want to show a brief overview of the NASA temperature dataset, including descriptive statistics and basic properties.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load data
    @st.cache_data
    def load_data():
        nasa = pd.read_csv("NASA_zonal.csv", encoding='latin1')
        nasa['Year'] = nasa['Year'].apply(lambda x: int(x.replace(',', '')) if isinstance(x, str) else x)
        return nasa
    
    nasa = load_data()

    # Show the data
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(nasa)
                    
    # Basic data info expander
    with st.expander("Properties and Descriptive statistics of the NASA dataset"):
        st.write("**Size of the DataFrame:**", nasa.shape)
        buffer = io.StringIO()
        nasa.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        st.markdown("**Variables:**")
        st.markdown("- The year of the measures\n"
                    "- 'Glob': This column represents the global average temperature anomaly. It provides the average temperature anomaly across the entire Earth, combining data from all latitudes and longitudes.\n"
                    "- 'NHem': This column represents the temperature anomaly for the Northern Hemisphere. It includes data from all latitudes in the Northern Hemisphere, from the equator (0 degrees latitude) to the North Pole (90 degrees latitude).\n"
                    "- 'SHem': This column represents the temperature anomaly for the Southern Hemisphere. It includes data from all latitudes in the Southern Hemisphere, from the equator (0 degrees latitude) to the South Pole (-90 degrees latitude).\n"
                    "- The rest of the variables represent latitude bands, indicating different regions of the Earth. E.g. the column '24N-90N' represents the latitude band from 24 degrees North to 90 degrees North, covering the Arctic region.")
        st.write("**Missing values per column:**", nasa.isna().sum())
        st.write("**Number of duplicates:**", nasa.duplicated().sum())
        st.write("**Data Description:**")
        st.write(nasa.describe())
    #Boxplot for distribution of variables
    st.markdown("#### Temperature anomalies in the different regions - Box-and-Whisker Plot")
    st.write('The following graph shows box-whisker plots for temperature anomalies in the different regions of the NASA data set. The analysis is essential to gain a better understanding of the distribution and dispersion of these data. Box-whisker plots provide a compact and informative representation of the statistical distribution and allow important aspects of the data to be grasped at a glance.')
    columns = ['Glob', 'NHem', 'SHem', "24N-90N", "24S-24N", "90S-24S", "64N-90N", "44N-64N", "24N-44N","EQU-24N", "24S-EQU", "44S-24S", "64S-44S", "90S-64S"]
    fig, ax = plt.subplots(figsize=(10, 6))
    nasa[columns].boxplot(ax=ax)
    ax.set_title('Temperature Anomalies - Box-and-Whisker Plot')
    ax.set_xlabel('Scope')
    ax.set_ylabel('Temperature Anomaly')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.write('As a brief summary of the Box-and-Whisker Plot Observations and descriptive statistics in general it can be stated that the range of data (min-max) for most variables falls between -1 and 1. Only two variables (64N-90N, 90S-64S) exceed these boundaries. For 10 variables, the standard deviation is less than 0.5,  less than 1.0 for 3 variables, and less than 1.5 °C for 1 variable. In most variables, the mean and median values are closely aligned, indicating a low impact of extreme values on the mean. This is consistent with the box and whisker plot, where outliers are visible for Glob, NHem, 24-90N, 64N-90N, 24N-44N, 24S-EQU, and 90S-64S. For all other variables, the box-whisker plot does not show extreme values. For most variables (except 90S-64S), the spread from maximum to Q2 is larger than from minimum to Q2. Since Q2 is very close to 0 °C, this suggests that there are more temperature changes above average for that specific year. In the North Pole region (64N-90N), a greater variability in temperature change is observed. Anomaly values extend upwards to over +2 degrees Celsius (and sometimes even over +3 degrees Celsius), while the box extends downward to about -1.8 degrees Celsius')
  
    # Line plot for global temperature anomalies
    st.markdown("#### Global and Hemispheric Temperature Anomalies (1880-2023) - Lineplot")
    st.write('Visualising time series data with a line chart provide a clear view of the historical development of temperature anomalies on a global level and also in different geographic regions (here the north and the south Hemisphere). By visualising these data, we can see changes over time, identify seasonal variations, and analyse potential long-term trends.')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=nasa, x='Year', y='Glob', label='Global', ax=ax)
    sns.lineplot(data=nasa, x='Year', y='NHem', label='North Hemisphere', ax=ax)
    sns.lineplot(data=nasa, x='Year', y='SHem', label='South Hemisphere', ax=ax)
    ax.set_title('Hemispheric Temperature Anomalies (1880-2023)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature Anomaly (°C)')
    st.pyplot(fig)
    st.write("""The Lineplot shows an increasing negative temperature change until 1910 (approx.) and increasing positive temperature change from approx. 1910 onwards until present. The graph indicates that temperature changes have been steadily increasing on average in recent years. This suggests that it is getting warmer on a global scale. Comparing temperature anomalies between the Northern and the Southern hemisphere shows that, especially since the year 2000, the temperature anomalies have been more positive in the Northern Hemisphere than in the Southern one. So the Temperature anomalies have been more positive in the Northern Hemisphere. This observation aligns with the overall understanding of climate change, as the Northern Hemisphere has been shown to experience more pronounced warming trends compared to the Southern Hemisphere. It could be attributed to various factors, including differences in land distribution, ocean currents, atmospheric circulation patterns, and human activities concentrated in the Northern Hemisphere. Overall, it is important to note that both hemispheres are experiencing significant warming trends, and the overall global temperature anomaly is increasing over time. This information underscores the importance of addressing climate change and its impacts on a global scale.""")
  
    # Striped plot for climatic development with segmented color map
    from matplotlib.colors import LinearSegmentedColormap
    st.markdown("#### Climatic Development Over Years - Data Stripes")
    st.write("""The next data visualisation contains data stripes, which provide an intuitive way to visualise climate change and temperature trends. They offer a quick and clear representation of Earth's warming, making it easy to identify long-term temperature trends and point out differences between the earth zones/ latitudes.""")
    cmap = LinearSegmentedColormap.from_list('climate_stripes', ['turquoise', 'white', 'red'], N=256)
    years = nasa['Year']
  
    for column in columns:
        data = nasa[column]
        fig, ax = plt.subplots(figsize=(12, 1))
        norm = plt.Normalize(data.min(), data.max())
        colors = [cmap(norm(val)) for val in data]
        ax.bar(years, [1] * len(years), color=colors, width=1)
        ax.set_xlim(min(years), max(years))
        x_ticks = range(min(years), max(years), 20)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(year) for year in x_ticks])
        ax.set_yticklabels([])
        ax.set_xlabel('Year')
        ax.set_ylabel('Value')
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation='horizontal', pad=0.60)
        cbar.set_label(f'Color for {column}')
        ax.set_title(f'Climatic development {column}')
        st.pyplot(fig)
    st.write('It appears that in recent decades, the majority of regions around the world have experienced predominantly positive temperature anomalies, especially since the 1980s / 1990s. However, there is an exception for the ‘90S-64S’ (South Pole) regions, where greater temperature variance is evident. Also the temperate zone in the south (64S-44S) shows earlier larger temperature changes since around 1970.')
  
    # Scatter plots with linear regression
    st.markdown("#### Relationship between years and temperature anomalies for different latitudes - Scatterplots")
    st.write('Another way of visualising the data is to use a scatter plot. These Scatter plots with linear regression lines provide visual insight into the relationship between years and temperature anomalies for different latitudes.')
    num_cols = len(nasa.columns) - 1
    num_rows = math.ceil(num_cols / 3)
    num_cols_subplot = min(num_cols, 3)
    fig, axes = plt.subplots(num_rows, num_cols_subplot, figsize=(15, 4 * num_rows))
  
    for i, column in enumerate(nasa.columns[1:]):
        row_idx = i // 3
        col_idx = i % 3
        ax = axes[row_idx, col_idx] if num_rows > 1 else axes[col_idx]
        ax.scatter(nasa['Year'], nasa[column], color='turquoise')
        ax.set_xlabel('Year')
        ax.set_ylabel(column)
        ax.set_title(f'Scatter Plot: Year vs. {column}')
        slope, intercept, r_value, p_value, std_err = linregress(nasa['Year'], nasa[column])
        line = slope * nasa['Year'] + intercept
        ax.plot(nasa['Year'], line, color='red')
        ax.text(0.8, 0.1, f'r = {r_value:.2f}\n p = {p_value:.4f}', transform=ax.transAxes)
  
    if num_cols_subplot < 3:
        for i in range(num_cols_subplot, 3):
            fig.delaxes(axes[row_idx, i])
  
    st.pyplot(fig)
    st.write('These graphs make it possible to identify trends, patterns, and correlations in temperature data. The linear regressions help quantify the direction and strength of these relationships and provide important insights into how climate has evolved in different regions over the years. To obtain a more profound comprehension of temperature variations over time across various latitudinal bands, a Pearson correlation analysis was performed for each latitude. Additionally, scatter plots were created to visualise the relationship between temperature change and the corresponding year. The plots reveal a consistent temperature increase over time, as all linear regression trends are positive. For the latitude 90S-64S (South Pole) plot, there is significant scatter, showing a more weak correlation. This suggests that while the South Pole has seen varied temperatures, using just the year is not enough to predict these anomalies. An also noticeable temperature dip occurred in approx. 1910 / 1920s and from the 1950s to 1980s (except in regions like SHem, 24S-24N, and 90S-64S). This non-uniform decrease hints at regional influences on temperature shifts, warranting further study.')


  ####

################################################################################################################################################################################################

if page ==  "🌍 Exploration Analysis - OWID":

   
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("CO2.png", width=120)
    
    with col2:
        st.markdown("""
        <div style='display: flex; align-items: center; height: 120px;'>
            <h3 style='margin: 0;'>Exploration Analysis - Our World In Data (OWID) </h3>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### The OWID Dataset")

    st.markdown("""
     <div style="border: 1px solid #d6d6d6; padding: 10px; border-radius: 5px; background-color: #e0f7fa; margin-bottom: 20px;">
     <p>The CO2 and Greenhouse Gas Emissions dataset is a collection of key metrics maintained by Our World in Data.
     It is updated regularly and includes data on CO2 emissions (annual, per capita, cumulative and consumption-based), other greenhouse gasses, energy mix, and other relevant metrics.</p>
     </div>
     """, unsafe_allow_html=True)
    
    st.write('Overview of the OWID dataset, including statistics and basic properties: This step provides a first insight into the dataset, including the available variables and the general structure.')

    # Load Data
    @st.cache
    def load_data():
        Co2 = pd.read_csv("owid-co2-data.csv", encoding='latin1')
        return Co2

    Co2 = load_data()

      # Show the data
    if st.checkbox('Show raw data'):
       st.subheader('Raw data')
       st.write(Co2)

     # Expandable section for descriptive statistics
    with st.expander("Descriptive statistics of the OWID dataset"):
         st.dataframe(Co2.describe())
         st.markdown('**Looking at the OWID data set, the summary statistics indicate various things:**')
         st.markdown('*  In various variables, the mean and median value differ substantially (e.g. co2: 379.98 mean vs 3.8 median. This mismatch could indicate the presence of outliers skewing the value distribution')
         st.markdown('*  A high number of missing values denoted as "0", skewing the distribution')
         st.markdown('*  The large difference between the Q3 and Q4 and the max value in various variables (e.g. co2, total_ghg) indicates the existence of very high outlier values')


    with st.expander("Properties of the OWID dataset"):
         st.markdown("###### Dimensions")
         st.markdown(f"- Number of Rows: {Co2.shape[0]}\n"
                     f"- Number of Columns: {Co2.shape[1]}\n")
         st.markdown("")
         st.markdown("###### Data types")
         st.markdown("- 76 variables are of data type float\n"
                    "- 1 variable is of dtype integer\n"
                    "- 2 variables are of dtype object\n"
                    "- This dataset is basically from the year 1880-2022 and shows the year wise values of CO2 emissions across different countries for every year\n")

         st.markdown("")
         st.markdown("###### Missing values")
         st.markdown("- There are missing values in almost every variable of the dataset\n"
                     "-  There are almost 31 columns with more than 50% of missing values\n")
         st.markdown("")
         st.markdown("###### Variables")
         st.markdown("- The dataset consists mainly of numerical variables on CO2 emissions with different scopes like emissions per emission source and the scope of aggregation (like total, shared, cumulative, per capita)\n"
                    "- Other context metrics like year, population, country and GDP\n"
                    "- Total countries in the dataset is around 231\n"
                    "- The values of carbon dioxide emissions are calculated in million tonnes\n"
                    "- For an in-detail description see [OWID CO2 Data GitHub](https://github.com/owid/co2-data)\n")

         st.markdown("***")

    with st.expander("CO2 Dataset Missing Values Analysis"):
         st.markdown("Below is the table showing the count and percentage of missing values for each column in the CO2 dataset")
    # Total missing values
         mis_val = Co2.isnull().sum()
    
    # Percentage of missing values
         mis_val_percent = 100 * mis_val / len(Co2)
    
    # Make a table with the results
         mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    # Rename the columns
         mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'Missing Values', 1: '% of Total Values'})
    
    # Sort the table by percentage of missing descending and filter out columns with no missing values
         mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values('% of Total Values', ascending=False).round(1)

    # Display the missing values table using Streamlit
    
         st.dataframe(mis_val_table_ren_columns)
    
         st.write('**Having a more in detail look at the amount of missing values in the data set shows that:**')
         st.markdown('  * There is a large amount of missing values in the data set, accumulating to 56,62% of all values in the data set.')
         st.markdown('  * The amount of missing values varies a great deal across variables,')
         st.markdown('  * Some variables have a comparably low percentage of missing values and are below 1/3 of all entries (e.g. share_global_luc_co2, co2),')
         st.markdown('  * While others with amount of missing values exceed 90% of entries (e.g. consumption_co2, other_industry_co2)')
    
         st.markdown('  * The high share of missing values across a large part of the variables in the OWID data set (ranging from 15.3% to 94.9% across variables) poses some challenges to data selection and data preprocessing that might influence interpretability of the results further down the road.')

         st.markdown("***")
if page ==  "🌍 Exploration Analysis - OWID":
#Plots
#Barplot of different categories of C02 emissions
 with st.expander("Barplot Representing the Distribution of CO2 Emissions Across Different Categories"):
      st.write(" This barplot provides a graphical representation of the percentage contribution of each category to the total CO2 emissions")
# CO2 categories
      categories = [
      'CO2',
      'Flaring CO2',
      'Other Industry CO2',
      'Methane',
      'Nitrous Oxide',
      'Oil CO2',
      'Gas CO2',
      'Coal CO2',
      'Cement CO2',
      'Total GHG',
      'Land Use Change CO2'
       ]

# Corresponding sum values for the selected categories
      co2_values = [
       11858676.64,
       90882.13,
       45375.86,
       956622.59,
       342210.54,
       2835551.22,
       1286208.57,
       3935870.71,
       216475.7,
       5022398.45,
       4609805.57
        ]
  
# Calculate percentages
      total_co2 = sum(co2_values)
      percentages = [(value / total_co2) * 100 for value in co2_values]

# Create a DataFrame
      df_bar = pd.DataFrame({
                  'Category': categories,
                  'Percentage': percentages
                  })
 

# Create bar plot with Plotly
      fig = px.bar(df_bar, x='Category', y='Percentage', title='CO2 Emissions by Category',
              labels={'Percentage': 'Percentage of Total CO2 Emissions'},
              color='Percentage',
              color_continuous_scale='Viridis')

# Update layout for better visualization
      fig.update_layout(
             xaxis_title='Category',
             yaxis_title='Percentage of Total CO2 Emissions',
             title={'text': 'CO2 Emissions by Category', 'x':0.5},
             xaxis_tickangle=-45
            )
# Display the plot in Streamlit
      st.plotly_chart(fig)
      st.markdown("***")

if page ==  "🌍 Exploration Analysis - OWID":
# Description of the plot
  with st.expander("### Description of the CO2 Emissions Distribution", expanded=True):
       
       st.write("""
  - **CO2 emissions** constitute the largest portion, representing **38%** of the total emissions.
  - **Land Use Change CO2** follows closely, accounting for **14.8%** of the total emissions, indicating the significant impact of land use practices on CO2 levels.
  - **Total GHG (Total Greenhouse Gases)** contribute **16.1%** to the emissions, emphasizing the collective impact of all greenhouse gases.
  - **Coal CO2** is a significant contributor at **12.6%**, indicating the role of coal in CO2 emissions from energy production.
  - **Oil CO2** accounts for **9.1%** of emissions, highlighting the contribution of oil-based activities.
  - **Gas CO2** represents **4.1%** of emissions, indicating the contribution of gas-related activities.
  - While categories like **Flaring CO2** and **Other Industry CO2** individually contribute smaller percentages, they still contribute to the overall emissions profile.
  - The bar chart underscores the diverse sources of CO2 emissions and the importance of addressing each category in mitigation strategies.
  """)

#Line plot for Global Co2 emissions by emission sources 
if page ==  "🌍 Exploration Analysis - OWID":
   with st.expander("Line plot representing Global CO2 Emissions by Emission Sources"):
        st.write("The line plot illustrates global CO2 emissions over time, categorized by various emission sources. Each line in the plot represents the trend of CO2 emissions from a specific source, such as flaring, industrial processes, methane, nitrous oxide, oil, gas, coal, cement production, land use changes, and the total greenhouse gas emissions.")

# Select relevant columns for CO2 emissions by different sources
        emission_sources = ['flaring_co2', 'other_industry_co2', 'methane', 'nitrous_oxide',
                    'oil_co2', 'gas_co2', 'coal_co2', 'cement_co2', 'total_ghg', 'land_use_change_co2']
        
# Aggregate data by summing over years
        emission_data = Co2.groupby('year')[emission_sources].sum().reset_index()

# Mapping variable names to custom legend labels
        legend_labels = {
       'flaring_co2': 'Flaring CO2',
       'other_industry_co2': 'Other Industry CO2',
       'methane': 'Methane',
       'nitrous_oxide': 'Nitrous Oxide',
       'oil_co2': 'Oil CO2',
       'gas_co2': 'Gas CO2',
       'coal_co2': 'Coal CO2',
       'cement_co2': 'Cement CO2',
       'total_ghg': 'Total GHG',
       'land_use_change_co2': 'Land Use Change CO2'
         }

# Plotting'
    
        emission_data_melted = emission_data.melt(id_vars=['year'], 
                                                  value_vars=emission_sources, 
                                                  var_name='source', 
                                                  value_name='emissions')
        
        emission_data_melted['source'] = emission_data_melted['source'].map(legend_labels)
        
        emission_data_melted.dropna(subset=['source'], inplace=True)
        # Create the line plot with Plotly
        fig = px.line(emission_data_melted, 
                      x='year', 
                      y='emissions', 
                      color='source', 
                      title='Global CO2 Emissions by Emission Sources',
                      labels={'emissions': 'CO2 Emissions (million tonnes)', 'year': 'Year'},
                      template='plotly_white')
        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='CO2 Emissions (million tonnes)',
            title={'text': 'Global CO2 Emissions by Emission Sources', 'x':0.5,'xanchor': 'center'},
            legend_title_text='Emission Source',
            xaxis=dict(tickmode='linear', tick0=1880, dtick=10)
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)
       
if page ==  "🌍 Exploration Analysis - OWID":
# Description of the plot
 with st.expander("Description of the Global CO2 Emissions by Emission Sources"):
       st.write("""
  - The contributions of different emission sources to the total CO2 emissions vary over time.
  - Some sources might show increasing trends, while others may exhibit fluctuations or decreasing patterns.
  - Certain emission sources, such as coal, oil, and gas, might stand out as major contributors to CO2 emissions due to their relatively higher emission levels.
  - The plot might unveil temporal patterns or anomalies in CO2 emissions from specific sources over time, indicating potential shifts in energy usage, industrial activities, or environmental policies.
  - Land use change CO2 emissions show distinct patterns, reflecting alterations in land use practices like deforestation, afforestation, or changes in agricultural land management.
  - The plot includes a line for total greenhouse gas emissions, which provides an overview of the cumulative impact of all emission sources considered, with an initial steady rise and then a sudden rise from around the 1990s till the 2020s, followed by a sudden decline.
  - The sudden decline can also be attributed to missing values that have not yet been addressed.
  - Observing the lines collectively can help in understanding the interrelation between different emission sources and their combined effect on global CO2 levels.
  """)
       st.markdown("***")
  
if page ==  "🌍 Exploration Analysis - OWID":
  with  st.expander("Top 15 Countries with Highest CO2 Emissions"):
        st.write("The line plot illustrates the trend of methane emissions over time for the top 15 countries with the highest total methane emissions. Each line represents the methane emissions trajectory for one of the top 15 countries")

        # Filter top 15 countries and recent years
        top15_countries = ['United States', 'China', 'Russia', 'Germany', 'Japan', 'India', 'United Kingdom', 'Canada', 'France', 'Italy', 'Poland', 'South Africa', 'Mexico', 'South Korea', 'Ukraine', 'World']
        df_top15 = Co2[Co2['country'].isin(top15_countries)].copy()
        df_top_recent = df_top15.loc[(df_top15["year"] >= 1950) & (df_top15.country != 'World')]
      
        # Create Plotly Express line plots for CO2, CH4, and N2O emissions
        fig_co2 = px.line(df_top_recent, x='year', y='co2', color='country', width=800, height=600)
        fig_ch4 = px.line(df_top_recent[df_top_recent['year'] >= 1990], x='year', y='methane', color='country', width=800, height=600)
        fig_n2o = px.line(df_top_recent[df_top_recent['year'] >= 1990], x='year', y='nitrous_oxide', color='country', width=800, height=600)

        # Add 'Other sources' column and rename columns
        df_top_recent['Other sources'] = df_top_recent['cumulative_flaring_co2'] + df_top_recent['cumulative_other_co2'].copy()
        df_top_recent = df_top_recent.rename(columns={'cumulative_gas_co2': 'Natural Gas Combustion', 'cumulative_oil_co2': 'Petroleum Combustion', 'cumulative_coal_co2': 'Coal Combustion'})

        # Create a histogram plot
        fig_emissions = px.histogram(df_top_recent, x='country', y=['Natural Gas Combustion', 'Petroleum Combustion', 'Coal Combustion', 'Other sources']).update_xaxes(categoryorder='total descending')
        fig_emissions.update_layout(xaxis_title='Country', yaxis_title='Cumulative CO2 Emissions by Fuel', width=800, height=600, legend_title_text='', legend=dict(y=1, x=0.68, bgcolor='rgba(255,255,255,0)'))

        # Streamlit select box for choosing the graph
        select_graph = st.selectbox('Select a figure to visualize', ['CO2 Emissions', 'CH4 Emissions', 'N2O Emissions','Distribution of CO2 Emissions by Fuel'])
        if select_graph == 'CO2 Emissions':
            st.plotly_chart(fig_co2)
        elif select_graph == 'CH4 Emissions':
            st.plotly_chart(fig_ch4)
        elif select_graph == 'N2O Emissions':
            st.plotly_chart(fig_n2o)
        else:
            st.plotly_chart(fig_emissions)
  
# Description of the plot
  with st.expander("Description of Global Co2 Emissions Distribution in the Top 15 countries"):
        st.write("""
      - The analysis is based on recent data focusing on the top 15 countries with significant CO2 emissions, excluding the 'World' entry and considering years from 1950 onwards.
      - China emerges as the largest CO2 emitter, with a noticeable increase from around 2,000 tonnes in the 1970s to approximately 11,500 tonnes in recent years.
      - The United States ranks second in CO2 emissions, showing fluctuations over the years. After a peak of around 8,200 tonnes in 2004, emissions have seen a slight reduction, indicating possible mitigation measures.
      - India exhibits a significant increase in CO2 emissions, reaching around 2,500 tonnes presently, suggesting a rising trend.
      - Russia experienced a surge in CO2 emissions around 1990, peaking at approximately 2,200 tonnes, but has since decreased to around 1,800 tonnes as of 2022, indicating efforts towards emission reduction.
      - Other countries like Japan, the UK, and Ukraine show varying levels of CO2 emissions, with values generally below 1,500 tonnes.
      - In terms of methane emissions, China leads with around 1,100 tonnes, followed by the United States, India, Russia, and Mexico, with emissions ranging from 750 to 170 tonnes.
      - A similar pattern is observed for nitrous oxide emissions, with China, the United States, India, and Russia being the top emitters.
      - The distribution of CO2 emissions by fuel type reveals that the United States has the highest emissions from coal combustion, followed by petroleum and natural gas, totaling around 17 million tonnes.
      - China follows with approximately 4.2 million tonnes, predominantly from coal combustion. Germany and the UK also exhibit significant emissions from coal, petroleum, and natural gas.
      - Russia's emissions, around 4.1 million tonnes, are evenly distributed among coal, petroleum, and natural gas.
      - Japan and France follow with notable emissions, while South Korea has the lowest emissions across all categories.
      - China consistently exhibits high levels of methane emissions over the years, likely due to its extensive agricultural activities, coal mining, and rapidly growing industrial sector.
      - The United States also shows a notable presence in emissions, attributed to its diverse economy, including agriculture, oil and gas production, and waste management practices.
      - India's methane emissions exhibit an upward trend, reflecting its growing population, agricultural practices, and expanding industrial base, which heavily relies on coal for energy production.
      - Russia's methane emissions may stem from various sources such as natural gas production, agricultural activities, and landfills, reflecting the country's vast territory and resource-intensive industries.
      - The European Union, representing a collective of countries, demonstrates efforts to curb methane emissions over time, possibly driven by regulatory measures, technological advancements, and increased awareness of environmental issues.
      """)
#####################################################################################################################################################################
 
if page ==  "📉 Exploration Analysis - STA":

    
     col1, col2 = st.columns([1, 3])
     with col1:
        st.image("Surface temperature anomaly.png", width=120)
    
     with col2:
        st.markdown("""
        <div style='display: flex; align-items: center; height: 120px;'>
            <h3 style='margin: 0;'>Exploration Analysis - Surface Temperature Anomaly (STA) </h3>
        </div>
        """, unsafe_allow_html=True)

     st.markdown("#### The Surface Temperature Anomaly Dataset")

     st.markdown("""
     <div style="border: 1px solid #d6d6d6; padding: 10px; border-radius: 5px; background-color: #e0f7fa; margin-bottom: 20px;">
     <p>Surface temperature anomaly, measured in degrees Celsius The temperature anomaly is relative to the 1951-1980 global average temperature.
     Data is based on the HadCRUT analysis from the Climatic Research Unit (University of East Anglia) in conjunction with the Hadley Centre (UK Met Office).</p>
     </div>
     """, unsafe_allow_html=True)

     
     st.write('Overview of the Surface Temperature Anomaly dataset, including statistics and basic properties: This step provides a first insight into the dataset, including the available variables and the general structure.')

    # Load Data
     @st.cache
     def load_data():
       sta = pd.read_csv("hadcrut-surface-temperature-anomaly.csv", encoding='latin1')
       return sta

     sta = load_data()

      # Show the data
     if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(sta)

     # Expandable section for descriptive statistics
     with st.expander("Descriptive statistics of the Surface Temperature Anomaly dataset"):
          st.dataframe(sta.describe())
          st.markdown('* The Entity is the country variable and the code is the country codes')
          st.markdown('* The Year variables is from 1850-2017 and the surface temperature anomaly is measured for every country every year')
          st.markdown('* The Surface temperature anomaly dataframe in total has 4 columns')

     with st.expander("Properties of the Surface Temperature Anomaly dataset"):
          st.markdown("###### Dimensions")
          st.markdown(f"- Number of Rows: {sta.shape[0]}\n"
                      f"- Number of Columns: {sta.shape[1]}\n")
          st.markdown("")
          st.markdown("###### Data types")
          st.markdown("- 1 variables are of data type float\n"
                      "- 1 variable is of dtype integer\n"
                      "- 2 variables are of dtype object\n")
          st.markdown("")
          st.markdown("###### Missing values")
          st.markdown("- Exist only in the Code variable in the dataset\n")
                      
          st.markdown("")
          st.markdown("###### Variables")
          st.markdown("- The dataset consists only 4 columns: The Year from 1850-2017, the surface temeprature measured in different countries every year over the mentioned time period and the country codes\n"
                      "- For an in-detail description see [Surface Temeprature Anomaly Data](https://ourworldindata.org/grapher/hadcrut-surface-temperature-anomaly)\n")

          st.markdown("***")
         
          st.markdown("#### Missing values")

    # Total missing values
          mis_val = sta.isnull().sum()
    
    # Percentage of missing values
          mis_val_percent = 100 * mis_val / len(sta)
    
    # Make a table with the results
          mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    # Rename the columns
          mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'Missing Values', 1: '% of Total Values'})
    
    # Sort the table by percentage of missing descending and filter out columns with no missing values
          mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values('% of Total Values', ascending=False).round(1)

    # Display the missing values table using Streamlit
          st.write("Below is the table showing the count and percentage of missing values for each column in the Surface temperature anomaly dataset:")
          st.dataframe(mis_val_table_ren_columns)
    
          st.write('**Having a more in detail look at the amount of missing values in the data set shows that:**')
          st.markdown('  * The column of Code had missing values of 164 about 0.55%')
          st.markdown('  * But when checked it was observed that there was an entry Micronesia in the code column')
          st.markdown('  * So we removed this entry and now the dataset has no missing values for the further computation and analysis')
         
          st.markdown("***")

if page ==  "📉 Exploration Analysis - STA":
 with st.expander("Surface Temperature Anomaly in Top 5 countries"):
      st.write("The plot illustrates the surface temperature anomaly trends in the top 5 countries (Afghanistan, Chad, Uganda, Romania, and Belarus) from the years 1880 to 2017.")
      top_countries = ['Afghanistan', 'Chad', 'Uganda', 'Romania', 'Belarus']
      surface_temp_top_countries = sta[sta['Entity'].isin(top_countries)]
    # Plotting
      fig = px.line(surface_temp_top_countries, x='Year', y='Surface temperature anomaly', color='Entity', 
              title='Surface Temperature Anomaly in Top 5 Countries (1880-2017)',
              labels={'Year': 'Year', 'Surface temperature anomaly': 'Surface Temperature Anomaly'})

      fig.update_layout(xaxis_title='Year', yaxis_title='Surface Temperature Anomaly', legend_title='Country')
    # Display the plot in Streamlit
      st.plotly_chart(fig)

    # Description of the plot 
if page ==  "📉 Exploration Analysis - STA":
 with st.expander("Description of Surface Temperature Anomaly Trends"):
      st.write("""
   - The plot allows for a visual comparison of surface temperature anomalies across the top 5 countries over the available time period.
   - Each country's data spans a different time period: Afghanistan from 1947 to 2017, Chad from 1946 to 2017, Uganda from 1901 to 2017, and Romania and Belarus from 1850 to 2017.
   - There have been significant fluctuations in temperature anomaly trends over the centuries.
   - Until around 1975, the trends remained relatively constant across most countries.
   - Belarus shows a sudden drop in surface temperature anomaly around 1945, possibly due to natural variations caused by increased industrial activities during World War II.
   - All the countries have exhibited an increasing trend in surface temperature anomalies after 1975 until 2017.
   - Uganda shows the highest surface temperature anomaly, reaching around 3.8 degrees Celsius around 2015.
   - An interesting observation is the falling trend in surface temperature anomaly for Afghanistan around 2017, indicating a deviation from the overall increasing trend observed in other countries.
   - This anomaly might warrant further investigation into the factors influencing temperature patterns in Afghanistan.
    """)     
      st.markdown("***") 

if page ==  "📉 Exploration Analysis - STA":
   sns.set_style("whitegrid")
   @st.cache
   def load_data():
       merged_data = pd.read_csv("merged_data.csv", encoding='latin1')
       return merged_data
   merged_data = load_data()
        
if page ==  "📉 Exploration Analysis - STA":  
  with st.expander("CO2 Emissions and Surface Temperature Anomalies Over Years"):
       st.write("The Line plot represents two line plots on the same graph. The first line plot depicts the trend of surface temperature anomaly over the years from 1850 to 2017. The second line plot illustrates the trend of CO2 emissions over the years from 1880 to 2022.")

      # Create a figure and axis object
       fig, ax1 = plt.subplots(figsize=(12, 6))
      # Plot CO2 emissions on the primary y-axis
       sns.lineplot(data=merged_data, x='Year', y='co2', color='red', ax=ax1, label='CO2 Emissions')
      # Set the y-label for CO2 emissions
       ax1.set_ylabel('CO2 Emissions (Tonnes)', color='red')
      # Create a secondary y-axis for Surface Temperature Anomaly
       ax2 = ax1.twinx()
       sns.lineplot(data=merged_data, x='Year', y='Surface temperature anomaly', color='blue', ax=ax2, label='Surface Temperature Anomaly')
      # Set the y-label for Surface Temperature Anomaly
       ax2.set_ylabel('Surface Temperature Anomaly (°C)', color='blue')
      # Set labels and title
       ax1.set_xlabel('Year')
       plt.title('CO2 Emissions and Surface Temperature Anomaly Over Years')
      # Show legend
       lines1, labels1 = ax1.get_legend_handles_labels()
       lines2, labels2 = ax2.get_legend_handles_labels()
       ax1.legend(lines1, ['CO2 Emissions'], loc='upper left')
       ax2.legend(lines2, ['Surface Temperature Anomaly'], loc='upper right') 
      # Rotate x-axis labels for better readability
       plt.xticks(rotation=45)
      # Show the plot
       st.pyplot(fig)
# Description of the plot
if page ==  "📉 Exploration Analysis - STA":
 with st.expander("Description of CO2 Emissions and Surface Temperature Anomalies Trends"):
      st.write("""                                                                                                                                  
    - Both line plots show an overall increasing trend over the respective time periods.
    - The surface temperature anomaly exhibits a steady increase from 1850 to 2017, while CO2 emissions show a rising trend from 1880 to 2022.
    - Despite the general upward trajectory, both plots also exhibit periods of fluctuations and variability.
    - These fluctuations may result from various factors such as natural climate variability, human activities, and external events.
    - The simultaneous increase in both surface temperature anomaly and CO2 emissions suggests a potential relationship between the two variables.
    - This observation aligns with the scientific understanding that increasing CO2 emissions contribute to global warming, leading to rising surface temperatures.
    - In recent years, there appears to be a steeper increase in both surface temperature anomaly and CO2 emissions.
    - This observation suggests a potential acceleration in global warming and underscores the urgency of addressing climate change mitigation efforts. 
    """)
      st.markdown("***")

if page ==  "📉 Exploration Analysis - STA":
    import streamlit as st
    import plotly.express as px
    import pandas as pd

if page ==  "📉 Exploration Analysis - STA":
 with st.expander("Surface Temperature Anomalies Over Years in different countries"):
      st.write("The plot shows surface temperature anomaly over the years from 1850 to 2017 across different countries")
    # Sort the values of Year Column
      sta = sta.sort_values(by='Year')
    # Plotly Choropleth Map with a different color scale
      fig = px.choropleth(
          sta,
          locations='Code',
          color='Surface temperature anomaly',
          hover_name='Entity',
          animation_frame='Year',
          projection='natural earth', 
          title='Surface Temperature Anomaly Over Time',
          color_continuous_scale='Viridis'  # Change the color scale to Viridis
          )

    # Customize the layout
      fig.update_layout(
              coloraxis_colorbar=dict(
              title='Surface Temperature Anomaly (°C)'
              ),
              coloraxis_colorbar_thickness=25,
              coloraxis_colorbar_len=0.5,
              autosize=False,
              width=1000,
              height=600,
              xaxis=dict(range=[1850, 2017])
              )

# Display the map in Streamlit
      st.plotly_chart(fig)
########################################################################################################################################################################################################################

if page == "🛠️ Modelling Preparation":
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
    st.markdown('<h1 class="centered-title">Modelling Preparation</h1>', unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="blue-box">
        <h3>Objective</h3>
        <p>Clean and prepare data for exploration and modeling, ensuring quality and consistency.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Datasets"):
        st.markdown("""
        - CO2 emissions data (OWID)
        - Surface temperature anomaly data (HadCRUT)
        """)

    st.markdown("""
    ### Pre-processing Steps
    1. **Renaming Columns:**
        - Standardized column names
        - Converted to lowercase and renamed for clarity

    2. **Handling Missing Values & Duplicates:**
        - Identified and removed rows with missing values.
        - Deleted 46,181 rows from the OWID dataset.
        - Deleted 346 rows from the Surface Temperature Anomaly dataset.
        - The final dataset has no missing values.
        - Ensured no duplicate records exist.

    3. **Outlier Detection and Removal:**
        - Identified outliers using a boxplot for the surface temperature anomaly column.
        - Removed outliers using the Z-score method. Data points with Z-scores greater than a threshold (e.g., 3) were considered outliers and removed.

    4. **Merging Datasets:**
        - Merged datasets based on country, iso_code, and year
        - Integrated features related to CO2 emissions, greenhouse gases, GDP, and population

    6. **Feature Selection:**
        - Dropped irrelevant columnss
        - Identified the target variable amd renamed it from surface temperature anomaly to sta for readibility.
        - Normalized features using Min-Max normalization to scale the data to a range [0, 1].

    7. **Further Cleaning and Formatting:**
        - Ensured appropriate data types
        - Converted float64 columns to int64 for standardization

    8. **Final Data Checks:**
        - Ensured no remaining missing values
        - Verified data types to ensure they were appropriate for further analysis and modeling.

    This meticulous pre-processing and merging of datasets ensured that our data was clean, well-structured, and ready for the next steps in our analysis and modeling process.
    """)

    @st.cache
    def load_data():
        datas_pre_processed = pd.read_csv("datas_pre_processed.csv", encoding='latin1')
        return datas_pre_processed

    datas_pre_processed = load_data()
    
    st.dataframe(datas_pre_processed)



###
########################################################################################################################################################################################################################

if page ==  "🤖 Machine Learning Models":
  # Title of the app
     st.title('Machine Learning Models')
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
     st.markdown('<h1 class="centered-title">Machine Learning Models</h1>', unsafe_allow_html=True)
     st.markdown("<br><br>", unsafe_allow_html=True)
     st.markdown("""
### Dataset Preparation and Initial Splitting for Machine Learning Models

In the initial stage of our machine learning models, we began by preparing our dataset for analysis. The dataset comprised several variables:

- **country_id**: Unique identifier for each country
- **year**: Year of the record
- **gdp**: Gross Domestic Product
- **population**: Population count
- **co2**: Total CO2 emissions
- **coal_co2**: CO2 emissions from coal
- **flaring_co2**: CO2 emissions from gas flaring
- **gas_co2**: CO2 emissions from gas
- **methane**: Methane emissions
- **nitrous_oxide**: Nitrous oxide emissions
- **oil_co2**: CO2 emissions from oil
- **sta**: Surface temperature anomaly (target variable)

The target variable for our models was the surface temperature anomaly (sta), while the remaining variables served as the features. We employed the `train_test_split` method to divide the dataset into training and testing subsets, allocating 80% of the data to training and 20% to testing, with a random state of 42 to ensure reproducibility. This foundational step was crucial in setting up the dataset for effective training and evaluation of our machine learning models.
""")
if page ==  "🤖 Machine Learning Models":
  with st.expander("**Linear Regression and Decision Tree Models**"):
    # Initial paragraph
    st.markdown("""
    Initially, the linear regression and decision tree models were used to compare and test the predictive performance. 
    The linear regression model showed limited predictive performance, as indicated by its relatively low R² scores on both the training and test sets, suggesting it struggles to capture the underlying relationship in the data. 
    On the other hand, the decision tree model can capture non-linear relationships and interactions between variables, potentially offering improved predictive accuracy and better handling of complex data structures. 
    However, the observed R² scores on the training and test sets for the decision tree model (Training R2 Score: 1.0, Test R2 Score: 0.061) suggested that it also struggles to capture the relationship effectively. Therefore, further analysis with different max depth values was conducted.

    The max_depth analysis of 5, 10, 15, and 20 was decided to investigate the impact of tree complexity on model performance and to mitigate overfitting observed. 
    By limiting the depth, we aimed to find a balance where the model generalizes better to unseen data, potentially improving the R² score on the test set while avoiding perfect but misleading performance on the training set.
    """)
    data = {
    "Model": ["Linear Regression", "Decision Tree", "Max Depth = 5", "Max Depth = 10", "Max Depth = 15", "Max Depth = 20"],
    "R² Train": [0.17, 1.0, 0.45, 0.85, 0.99, 0.99],
    "R² Test": [0.19, 0.06, 0.36, 0.19, 0.09, 0.071],
    "MAE": [0.40, 0.42, 0.36, 0.39, 0.41, 0.42],
    "MSE": [0.26, 0.30, 0.20, 0.25, 0.29, 0.30],
    "RMSE": [0.51, 0.55, 0.45, 0.50, 0.54, 0.54]
    }

# Create a DataFrame
    df = pd.DataFrame(data)

# Display the table
    st.table(df)
  # Title Description
    st.markdown("<h2 style='text-align: center;'>Comparison of the Max Depth Values for the Decision Tree Models</h2>", unsafe_allow_html=True)
  
    st.markdown("Line Plot illustrates a better understanding of the R² values represented at different max depth levels of 5, 10, 15, and 20 respectively.")

    max_depth_values = [5, 10, 15, 20]

# Define the R2 scores for training and test sets
    training_r2_scores = [0.46, 0.86, 0.98, 0.99]
    test_r2_scores = [0.38, 0.24, 0.10, 0.16]

# Plotting the R2 scores
    plt.figure(figsize=(10, 6))
    plt.plot(max_depth_values, training_r2_scores, marker='o', label='Training R2 Score')
    plt.plot(max_depth_values, test_r2_scores, marker='o', label='Test R2 Score')
    plt.title('Comparison of the Max Depth Values for the Decision Tree Models')
    plt.xlabel('max_depth')
    plt.ylabel('R2 Score')
    plt.xticks(max_depth_values)
    plt.legend()
    plt.grid(True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


# Plot description
    st.markdown("""
The plot illustrates the relationship between the maximum depth (max_depth) of a Decision Tree regressor and the R² scores for both the training and test sets. 
As the max_depth increases from 5 to 20, the training R² score improves significantly, reaching almost perfect values (0.99) at depths 15 and 20. 
This indicates that the model increasingly captures the patterns in the training data, eventually overfitting it. 
Conversely, the test R² score is highest at a max_depth of 5 (0.36) and decreases with deeper trees, falling to 0.07 at a max_depth of 20. 
This trend suggests that as the model complexity increases, its performance on the test data diminishes due to overfitting.
    """)

# Last st.markdown
    st.markdown("""
The Decision Tree with a max depth of 5 achieves a Test R² score of 0.36, closely aligning with the Random Forest's Test R² score of 0.39. 
When comparing the performance metrics of the Decision Tree Regressor with various maximum depths and a Random Forest Regressor, 
the max depth of 5 for the Decision Tree emerges as the best choice.
    """)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

if page == "🤖 Machine Learning Models":
    with st.expander("**Random Forest Model**"):
        st.write("In the field of predictive analytics and data science, Random Forest modelling stands out as a powerful and versatile machine learning technique, where multiple decision trees are trained and aggregated to improve the overall predictive performance and robustness of the model. Those models are particularly well-suited for handling complex datasets with numerous features and intricate relationships.")

        @st.cache_data
        def load_data():
            return pd.read_csv("datas_pre_processed.csv", index_col=0)

        df = load_data()
        st.write("### Data Preview")
        st.write(df.head())

        # Show correlation matrix
        if st.checkbox("Show Correlation Matrix"):
            correlation_matrix = df.corr()
            plt.figure(figsize=(10, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            st.pyplot(plt)

        # Feature Selection based on Correlation
        st.write("### Feature Selection")
        correlation_with_target = df.corr()['sta'].abs().sort_values(ascending=False)
        selected_features = ['year', 'gdp', 'population', 'coal_co2', 'co2']
        st.write("Selected Features:", selected_features)

        # Split the data
        X = df[selected_features]
        y = df['sta']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Training
        st.write("### Model Training")
        n_estimators = st.slider("Number of Estimators", min_value=10, max_value=200, value=100, step=10)
        random_forest = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        random_forest.fit(X_train, y_train)

        # Predictions and Evaluation
        y_train_pred = random_forest.predict(X_train)
        y_test_pred = random_forest.predict(X_test)
        mse_rf = mean_squared_error(y_test, y_test_pred)
        mae_rf = mean_absolute_error(y_test, y_test_pred)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        st.write("### Model Evaluation")
        metrics_df_rf = pd.DataFrame({
            'Metric': ['Mean Squared Error', 'Mean Absolute Error', 'R² Score Train', 'R² Score Test'],
            'Value': [mse_rf, mae_rf, r2_train, r2_test]
        })
        st.write(metrics_df_rf)

        # Feature Importance
        st.write("### Feature Importance")
        feature_importance_rf = random_forest.feature_importances_
        feature_importance_df_rf = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance_rf})
        feature_importance_df_rf = feature_importance_df_rf.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(8, 6))
        sns.barplot(x="Importance", y="Feature", data=feature_importance_df_rf)
        plt.title("Feature Importance")
        st.pyplot(plt)

        # Prediction Plot
        st.write("### Prediction Plot")
        min_val_rf = min(y_test.min(), y_test_pred.min())
        max_val_rf = max(y_test.max(), y_test_pred.max())
        line_rf = np.linspace(min_val_rf, max_val_rf, 100)

        distances_rf = np.abs(y_test - y_test_pred)
        threshold_rf = np.percentile(distances_rf, 75)
        far_from_line_rf = distances_rf > threshold_rf
        near_to_line_rf = ~far_from_line_rf

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test[near_to_line_rf], y_test_pred[near_to_line_rf], alpha=0.5, label='Ideal Predictions', color='blue')
        plt.scatter(y_test[far_from_line_rf], y_test_pred[far_from_line_rf], alpha=0.5, color='red', label='Inaccurate Predictions')
        plt.plot(line_rf, line_rf, color='green', linestyle='--', label='Correlation Line')

        plt.xlabel('Actual Temperature Anomaly')
        plt.ylabel('Predicted Temperature Anomaly')
        plt.title('Actual vs. Predicted Temperature Anomaly (Random Forest)')
        plt.legend()
        st.pyplot(plt)



if page ==  "🤖 Machine Learning Models":
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
  st.markdown('<h1 class="centered-title">Gradient Boosting</h1>', unsafe_allow_html=True)
  st.markdown("<br><br>", unsafe_allow_html=True)
  st.markdown("""Gradient Boosting is also a powerful machine learning technique used for regression and classification tasks. It builds models sequentially, with each new model attempting to correct the errors made by the previous models. The method combines the predictions of multiple weak learners, typically decision trees, to produce a strong learner that delivers accurate predictions. Our goal is to initialise Gradient Boosting to improve predictive accuracy in comparison to the previous models. First, we initialise the Gradient Boosting Regressor with n_estimators=200. This parameter specifies the number of boosting stages (or weak learners) to be used. In this case, 200 decision trees will be built sequentially, each one correcting the errors of the previous ones.""")

  import streamlit as st
  import pandas as pd
  import numpy as np
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import GradientBoostingRegressor
  from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  # Function to calculate RMSE
  def rmse(y_true, y_pred):
      return np.sqrt(mean_squared_error(y_true, y_pred))
  
  # Load the dataset
  df = pd.read_csv("datas_pre_processed.csv", index_col=0)

  # Display correlation matrix in a foldable section
  correlation_matrix = df.corr()
  with st.expander("Correlation Matrix"):
      st.write(correlation_matrix)
  
  # Select relevant features
  selected_features = ['year', 'gdp', 'population', 'coal_co2', 'co2']
  X = df[selected_features]
  y = df['sta']
  
  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  # Initialize the Gradient Boosting regressor
  gradient_boosting = GradientBoostingRegressor(n_estimators=200, random_state=30)
  
  # Train the model
  gradient_boosting.fit(X_train, y_train)
  
  # Make predictions on the test set
  y_pred_gb = gradient_boosting.predict(X_test)
  
  # Calculate evaluation metrics
  r2_train_gb = r2_score(y_train, gradient_boosting.predict(X_train))
  r2_test_gb = r2_score(y_test, y_pred_gb)
  mae_gb = mean_absolute_error(y_test, y_pred_gb)
  mse_gb = mean_squared_error(y_test, y_pred_gb)
  rmse_gb = rmse(y_test, y_pred_gb)
  
  # Display evaluation metrics in a table
  st.subheader("Evaluation Metrics")
  metrics_data = {
      'Metric': ['R² (Train)', 'R² (Test)', 'MAE', 'MSE', 'RMSE'],
      'Value': [r2_train_gb, r2_test_gb, mae_gb, mse_gb, rmse_gb]
  }
  metrics_df = pd.DataFrame(metrics_data)
  st.table(metrics_df)
  
  # Calculate the absolute errors
  absolute_errors_gb = np.abs(y_test - y_pred_gb)
  
  # Define a threshold for highlighting large errors
  error_threshold = 0.4
  
  # Define colors based on the absolute errors
  colors_gb = np.where(absolute_errors_gb <= error_threshold, 'blue', 'red')
  
  # Create a scatter plot with colors
  plt.figure(figsize=(8, 6))
  plt.scatter(y_test, y_pred_gb, c=colors_gb, alpha=0.5)
  plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='gray', linestyle='--')
  plt.xlabel('Actual Temperature Anomaly')
  plt.ylabel('Predicted Temperature Anomaly')
  plt.title('Gradient Boosting - Actual vs. Predicted Temperature Anomaly')
  
  # Add legend for colors
  plt.legend(['Ideal Prediction', f'Absolute Error <= {error_threshold}', f'Absolute Error > {error_threshold}'], loc='upper left')
  
  # Display plot
  st.pyplot(plt)

  # Initialize Gradient Boosting regressors with different n_estimators
  gradient_boosting_50 = GradientBoostingRegressor(n_estimators=50, random_state=30)
  gradient_boosting_200 = GradientBoostingRegressor(n_estimators=200, random_state=30)
  gradient_boosting_400 = GradientBoostingRegressor(n_estimators=400, random_state=30)
  
  # Train the models
  gradient_boosting_50.fit(X_train, y_train)
  gradient_boosting_200.fit(X_train, y_train)
  gradient_boosting_400.fit(X_train, y_train)
  
  # Make predictions on the test set
  y_pred_gb_50 = gradient_boosting_50.predict(X_test)
  y_pred_gb_200 = gradient_boosting_200.predict(X_test)
  y_pred_gb_400 = gradient_boosting_400.predict(X_test)
  
  # Calculate evaluation metrics for all models
  r2_test_gb_50 = r2_score(y_test, y_pred_gb_50)
  r2_test_gb_200 = r2_score(y_test, y_pred_gb_200)
  r2_test_gb_400 = r2_score(y_test, y_pred_gb_400)
  
  mae_gb_50 = mean_absolute_error(y_test, y_pred_gb_50)
  mae_gb_200 = mean_absolute_error(y_test, y_pred_gb_200)
  mae_gb_400 = mean_absolute_error(y_test, y_pred_gb_400)
  
  mse_gb_50 = mean_squared_error(y_test, y_pred_gb_50)
  mse_gb_200 = mean_squared_error(y_test, y_pred_gb_200)
  mse_gb_400 = mean_squared_error(y_test, y_pred_gb_400)
  
  rmse_gb_50 = np.sqrt(mse_gb_50)
  rmse_gb_200 = np.sqrt(mse_gb_200)
  rmse_gb_400 = np.sqrt(mse_gb_400)
  
  # Create a DataFrame for comparison
  metrics_data = {
      'Metric': ['R²', 'MAE', 'MSE', 'RMSE'],
      'Gradient Boosting (50 estimators)': [r2_test_gb_50, mae_gb_50, mse_gb_50, rmse_gb_50],
      'Gradient Boosting (200 estimators)': [r2_test_gb_200, mae_gb_200, mse_gb_200, rmse_gb_200],
      'Gradient Boosting (400 estimators)': [r2_test_gb_400, mae_gb_400, mse_gb_400, rmse_gb_400]
  }
  comparison_df = pd.DataFrame(metrics_data)
  comparison_df.set_index('Metric', inplace=True)
  
  st.markdown("<h3>Comparison of Gradient Boosting Models with different Estimators</h3>", unsafe_allow_html=True)

  #Create a bar chart using Matplotlib with custom colors
  fig, ax = plt.subplots(figsize=(10, 6))
    
  models = comparison_df.index
  metrics = comparison_df.columns
    
  x = np.arange(len(models))
  bar_width = 0.2
    
  colors = ['#4682B4', '#808080', '#A9A9A9', '#C0C0C0']  
    
  for i, metric in enumerate(metrics):
      ax.bar(x + i * bar_width, comparison_df[metric], width=bar_width, label=metric, color=colors[i])
    
  ax.set_xticks(x + (len(metrics) - 1) * bar_width / 2)
  ax.set_xticklabels(models, rotation=45, ha="right")
  ax.legend()
    
  # Add annotations to each bar
  for i in range(len(models)):
      for j, metric in enumerate(metrics):
          value = comparison_df.iloc[i][metric]
          ax.text(i + j * bar_width, value + 0.05, f'{value:.2f}', ha='center', va='bottom')
      ax.set_ylabel('Metrics')
      ax.set_title('Comparison of Gradient Boosting Models')
      ax.set_ylim(0, max(comparison_df.values.max(axis=1)) + 0.2)  
  # Display the plot
  st.pyplot(fig)  

###################################################################################################################

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import pickle

if page == '📈 Time-series modeling with SARIMA':
    st.title('Time-series modeling with SARIMA')

    # Load and prepare the data
    ts = pd.read_csv('ts_final.csv', index_col='Date', parse_dates=True)

    st.subheader('Introduction')
    st.write("Time series data is all around us, like in this case as weather patterns to demand forecasting and seasonal trends. To predict future values, we turn to powerful models like the Seasonal Autoregressive Integrated Moving Average (SARIMA). This is a versatile and widely used time series forecasting model as an extension of the non-seasonal ARIMA model, designed to handle data with seasonal patterns. SARIMA captures both short-term and long-term dependencies within the data, making it a robust tool for forecasting. It combines the concepts of autoregressive (AR), integrated (I), and moving average (MA) models with seasonal components.")

    # Display the first few rows of the dataset
    st.subheader('Dataset Preview')
    st.write(ts.head())

    # Visualize the data
    st.subheader('Monthly Temperature Anomaly Over Time')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts, label='Temperature Anomaly')
    ax.set_title('Monthly Temperature Anomaly')
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature Anomaly')
    ax.legend()
    st.pyplot(fig)

    # Check for stationarity
    st.subheader('Check for Stationarity')
    adf_result = adfuller(ts)
    st.write('ADF Statistic:', adf_result[0])
    st.write('p-value:', adf_result[1])
    for key, value in adf_result[4].items():
        st.write(f'Critical Value ({key}): {value}')

    if adf_result[1] > 0.05:
        st.write("The series is not stationary. Applying differencing...")
        ts_diff = ts.diff().dropna()

        # Check stationarity again
        adf_result_diff = adfuller(ts_diff)
        st.write('ADF Statistic (differenced):', adf_result_diff[0])
        st.write('p-value (differenced):', adf_result_diff[1])
        for key, value in adf_result_diff[4].items():
            st.write(f'Critical Value ({key} - differenced): {value}')
        
        # Plot differenced series
        st.subheader('Differenced Series')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(ts_diff, label='Differenced Temperature Anomaly')
        ax.set_title('Differenced Monthly Temperature Anomaly')
        ax.set_xlabel('Year')
        ax.set_ylabel('Temperature Anomaly')
        ax.legend()
        st.pyplot(fig)
    else:
        st.write("The series is stationary. Proceeding with modeling...")
        ts_diff = ts

    # ACF and PACF plots
    st.subheader('ACF and PACF Plots')
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plot_acf(ts_diff, ax=ax[0], lags=40)
    plot_pacf(ts_diff, ax=ax[1], lags=40)
    st.pyplot(fig)

    # Split the data into training and testing sets
    st.subheader('Train-Test Split')
    train_size = st.slider('Select training set size (as a percentage of total data):', 50, 95, 80)
    train_size = int(len(ts) * train_size / 100)
    train, test = ts[:train_size], ts[train_size:]

    # Fit SARIMA model on the training set (example parameters, adjust as needed)
    st.subheader('Fit SARIMA Model')
    st.write('Fitting SARIMA model with parameters (1, 1, 1)x(1, 1, 1, 12)')
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    
    # Display summary in a scrollable text box
    summary_html = results.summary().as_html()
    st.markdown(f'<div style="overflow: scroll; height: 200px; border: 1px solid black;">{summary_html}</div>', unsafe_allow_html=True)

    # Model diagnostics
    st.subheader('Model Diagnostics')
    fig = results.plot_diagnostics(figsize=(12, 8))
    st.pyplot(fig)

    # Forecasting on the test set
    st.subheader('Forecasting')
    forecast_steps = len(test)
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_ci = forecast.conf_int()

    # Plot forecast against actual data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train, label='Train')
    ax.plot(test, label='Test', color='orange')
    ax.plot(forecast.predicted_mean, label='Forecast', color='red')
    ax.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='red', alpha=0.3)
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature Anomaly')
    ax.legend()
    st.pyplot(fig)

    # Calculate RMSE and MAPE
    st.subheader('Model Evaluation Metrics')
    actual = test
    predicted = forecast.predicted_mean

    st.write('Actual values:', actual)
    st.write('Predicted values:', predicted)

    if len(actual) == len(predicted):
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = mean_absolute_percentage_error(actual, predicted)

        st.write(f'RMSE: {rmse:.4f}')
        st.write(f'MAPE: {mape:.4f}')
    else:
        st.write("Error: The actual and predicted series do not have the same length.")
        st.write(f'Length of actual values: {len(actual)}')
        st.write(f'Length of predicted values: {len(predicted)}')

    # Save the model
    st.subheader('Save the Model')
    if st.button('Save Model'):
        with open('sarima_model.pkl', 'wb') as pkl:
            pickle.dump(results, pkl)
        st.write("Model saved as 'sarima_model.pkl'.")

    # Load the model (for demonstration)
    st.subheader('Load the Model')
    if st.button('Load Model'):
        with open('sarima_model.pkl', 'rb') as pkl:
            loaded_model = pickle.load(pkl)
        st.write("Model loaded successfully.")
        st.write(loaded_model.summary())

    st.subheader('Conclusion')
    st.markdown("The analysis of the temperature anomaly data using the Seasonal Autoregressive Integrated Moving Average (SARIMA) model has yielded promising results. After fitting the SARIMA model to the historical temperature anomaly data since 1880, we achieved impressive accuracy metrics. These metrics indicate that the SARIMA model accurately captures the patterns and trends in the historical temperature anomaly data, significantly outperforming the baseline model. The low error values suggest that the model is reliable in predicting future temperature anomalies. An analysis of the residuals confirmed that they resemble white noise, which validates the model's assumptions. The residuals exhibited no significant autocorrelation, indicating that the SARIMA model has effectively captured the underlying structure of the temperature anomaly data. Using the fitted SARIMA model, we forecasted the temperature anomalies for the next three years. The forecast indicates a continuing increase in temperature anomalies, suggesting that the trend of rising temperatures observed in the historical data is expected to persist. The increasing temperature anomalies forecasted by the SARIMA model align with broader climate change trends observed globally. These findings underscore the importance of continued monitoring and proactive measures to mitigate the impacts of climate change. The model's accuracy provides confidence in its predictions, making it a valuable tool for researchers and policymakers in planning and implementing climate-related strategies. In conclusion, the SARIMA model has proven to be a robust and reliable method for analysing and forecasting temperature anomalies.")

###########################################################################################
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor

if page == "🔮 Prediction":
    st.title('Prediction Simulation with Gradient Boosting')
    st.write(' ')
  
    def prediction():
        def load_model():
            try:
                with open("gradient2.pkl", "rb") as f:
                    model = pickle.load(f)
                st.write("Model loaded successfully.")
                return model
            except FileNotFoundError:
                st.error("Model file not found. Please check the file path.")
                return None
            except pickle.UnpicklingError:
                st.error("Error unpickling the model file. The file might be corrupted.")
                return None
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                return None
    
        def get_features(year, coal_co2, population, gdp, co2):
            selected_features = np.array([year, coal_co2, population, gdp, co2])
            return selected_features.reshape(1, -1)
    
        def sta(features):
            model = load_model()
            if model is not None and hasattr(model, 'predict'):
                try:
                    prediction_result = model.predict(features)
                    return np.round(prediction_result, 3)
                except AttributeError as e:
                    st.error(f"AttributeError: {e}. There might be an issue with the model or input features.")
                    return None
                except Exception as e:
                    st.error(f"An unexpected error occurred during prediction: {e}")
                    return None
            else:
                st.error("Loaded model does not have a 'predict' method. It might be the wrong type.")
                return None
    
    
        data = pd.read_csv("datas_pre_processed.csv")
        df3 = data.copy()
    
        # min/max values
        year_min, year_max = df3['year'].min(), df3['year'].max()
        population_min, population_max = df3['population'].min(), df3['population'].max()
        gdp_min, gdp_max = df3['gdp'].min(), df3['gdp'].max()
        coal_co2_min, coal_co2_max = df3['coal_co2'].min(), df3['coal_co2'].max()
        co2_min, co2_max = df3['co2'].min(), df3['co2'].max()
    
        year_value = df3['year'].max()
        population_value = df3['population'].mean()
        gdp_value = df3['gdp'].mean()
        coal_co2_value = df3['coal_co2'].mean()
        co2_value = df3['co2'].mean()
    
    
        # Year slider
        year = st.slider("Year", min_value=int(year_min), max_value=int(year_max), step=1, value=int(year_value))
        
        # Population slider
        population = st.slider("Population", min_value=float(population_min), max_value=float(population_max), value=float(population_value))
        
        # GDP slider
        gdp = st.slider("GDP", min_value=float(gdp_min), max_value=float(gdp_max), value=float(gdp_value))

        # Coal CO2 slider
        coal_co2 = st.slider("Coal CO2", min_value=float(coal_co2_min), max_value=float(coal_co2_max), value=float(coal_co2_value))
        
        # CO2 slider
        co2 = st.slider("CO2", min_value=float(co2_min), max_value=float(co2_max), value=float(co2_value))
    
        # prediction button
        if st.button("Predict"):
            selected_features = get_features(year, coal_co2, population, gdp, co2)
            st.write("Selected features:", selected_features)
            prediction_result = sta(selected_features)
            if prediction_result is not None:
                st.write("Predicted Surface Temperature Anomaly:", prediction_result)
            else:
                st.write("Prediction could not be made due to an error in loading the model.")

    if __name__ == "__main__":
        prediction()


########################################################################################################################################################################################################################
if page ==  "📌 Conclusion":
  # Title of the app
     st.title('Conclusion')
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

if page == "📌 Conclusion":
    st.markdown("""
    ## Comparison of all Models & Conclusion
    
    The following table presents the performance metrics of various machine learning models tested on the datas_preprocessed dataset, alongside the SARIMA time series model applied to the NASA dataset. These metrics include R² scores for both training and test sets, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
    """)
    
    # Define the data for the table
    model_metrics = {
       "Model/Metric": ["Machine Learning models on the datas_preprocessed dataset", "Linear Regression", "Decision Tree", "Lasso", "Ridge", "Random Forest", "Gradient Boost", "Time Series Model on the NASA dataset", "SARIMA"],
        "R² Score Train": ["-", 0.17, 0.45, 0.17, 0.18, 0.92, 0.75, "-", "-"],
        "R² Score Test": ["-", 0.19, 0.36, 0.17, 0.18, 0.40, 0.45, "-", 0.09],
        "MAE (Mean Absolute Error)": ["-", 0.40, 0.36, 0.41, 0.41, 0.36, 0.34, "-", 0.09],
        "MSE (Mean Squared Error)": ["-", 0.26, 0.20, 0.26, 0.26, 0.20, 0.18, "-", 0.01],
        "RMSE (Root Mean Squared Error)": ["-", 0.51, 0.45, 0.51, 0.51, 0.44, 0.42, "-", 0.12]
    }
    
    
    # Create the table
    st.table(model_metrics)
    
    st.markdown("""
    The results from the machine learning models on the datas_preprocessed dataset and the time series analysis using the SARIMA model on the NASA dataset provide insightful conclusions regarding the surface temperature anomaly. 
    The Gradient Boosting model stands out as the best performer among the machine learning models with an R² score of 0.45 on the test set, indicating its superior ability to explain the variance in the data. 
    The SARIMA model for time series forecasting on the NASA dataset shows excellent performance with low MAE (0.09), MSE (0.01), and RMSE (0.12), indicating its robustness in predicting temperature anomalies.
    
    These results align with the project's aim of assessing the surface temperature anomaly, which is increasing over the years. By integrating datasets from GISTEMP v4, CO2 and Greenhouse Gas Emissions, and FAOSTAT, a comprehensive analysis was conducted. 
    This integration helps in understanding the multi-faceted aspects of temperature anomalies, influenced by global surface temperature changes, CO2 emissions, and greenhouse gas concentrations.
    """)
  ###

if page ==   "👥 Credits" :
   st.title('Credits')

   col1, col2, col3 = st.columns(3)
if page ==  "👥 Credits" :   
  with col1:
     st.write("**Members of the project team:**")
     st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)
     st.write("**Resources:**")
     st.markdown("<br><br><br><br><br><br><br>", unsafe_allow_html=True)
     #st.write("**Project report: uploaden?**") # upload report
if page ==  "👥 Credits" :   
  with col2:
     st.write("Manasi Deshpande")
     st.write("Desireé Jörke")
     st.write("Fiona Murphy")
     st.markdown("<br>", unsafe_allow_html=True)
     st.write("Tarik Anour (Tutor)")
     st.markdown("<br>", unsafe_allow_html=True)
     st.write("[NASA GISTEMP Data](https://data.giss.nasa.gov/gistemp/)")
     st.write("[OWID CO2 Data](https://github.com/owid/co2-data)")
     st.write("[Surface Temperature Anomaly Data](https://ourworldindata.org/grapher/hadcrut-surface-temperature-anomaly)")
     st.write("[FAO Annual Surface Temperature Change dataset](https://www.fao.org/faostat/en/#data/ET)")
    
  with col3:     
     linkedin_icon = "https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg"

     st.markdown(
            f'<a href="https://www.linkedin.com/in/manasi-deshpande-b68730191/" target="_blank">'
            f'<img class="linkedin-logo" src="{linkedin_icon}" alt="LinkedIn" width="20" height="20" />'
            f'</a>', 
            unsafe_allow_html=True
            )


     st.markdown(
            f'<a href="https://www.linkedin.com/in/desireé-jörke-7ba6321a3/" target="_blank">'
            f'<img class="linkedin-logo" src="{linkedin_icon}" alt="LinkedIn" width="20" height="20" />'
            f'</a>', 
            unsafe_allow_html=True
        )

     st.markdown(
            f'<a href="https://https://www.linkedin.com/in/fionamurphy90/" target="_blank">'
            f'<img class="linkedin-logo" src="{linkedin_icon}" alt="LinkedIn" width="20" height="20" />'
            f'</a>', 
            unsafe_allow_html=True
        )

if page ==  "👥 Credits" :  
     st.markdown("<br><br>", unsafe_allow_html=True)
     st.markdown("<span style='font-size: 12px;'>\*For each member of the group, specify the level of expertise around the problem addressed:</span>  \n<span style='font-size: 12px;'>   None of the members have prior knowledge with respect to in-depth climate data analysis.</span>", unsafe_allow_html=True)


# linkedIn logo 1 https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Logo.svg.original.svg
# linkedIn logo 2 https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg

########################################################################################################################################################################################################################

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

if page == "📊 Exploration Analysis - FAO":
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.image("fao.png", width=120)
    
    with col2:
        st.markdown("""
        <div style='display: flex; align-items: center; height: 120px;'>
            <h3 style='margin: 0;'>Exploration Analysis - Food and Agriculture Organization of the United Nations (FAO) </h3>
        </div>
        """, unsafe_allow_html=True)

    # Function to load data
    @st.cache
    def load_data():
        ETC_all = pd.read_csv('Environment_Temperature_change_E_All_Data.csv', encoding='latin1')
        ETC_all_noflag = pd.read_csv('Environment_Temperature_change_E_All_Data_NOFLAG.csv', encoding='latin1')
        ETC_all_area_codes = pd.read_csv('Environment_Temperature_change_E_AreaCodes.csv', encoding='latin1')
        ETC_all_area_flags = pd.read_csv('Environment_Temperature_change_E_Flags.csv', encoding='latin1')
        ETC_cleaned = pd.read_csv('ETC.csv', encoding='latin1')
        return ETC_all, ETC_all_noflag, ETC_all_area_codes, ETC_all_area_flags, ETC_cleaned

    # Load the data
    ETC_all, ETC_all_noflag, ETC_all_area_codes, ETC_all_area_flags, ETC_cleaned = load_data()


    st.markdown("""
<div style="border: 1px solid #d6d6d6; padding: 10px; border-radius: 5px; background-color: #e0f7fa; margin-bottom: 20px;">
    <p>The FAOSTAT Temperature Change data provides updates on how land temperatures have shifted from 1961 to 2023. 
    It includes monthly, seasonal, and yearly temperature changes compared to the 1951-1980 baseline. 
    This means that current temperature changes are measured against the average temperatures recorded during the period from 1951 to 1980, 
    which serves as a reference point for understanding long-term trends.</p>
</div>
""", unsafe_allow_html=True)
    
    with st.expander("Full description of data"):
         st.markdown("""
**Data description:**\n
The FAOSTAT Temperature change on land domain disseminates statistics of mean surface temperature change by country, with annual updates. The current dissemination covers the period 1961–2023. Statistics are available for monthly, seasonal and annual mean temperature anomalies, i.e., temperature change with respect to a baseline climatology, corresponding to the period 1951–1980. The standard deviation of the temperature change of the baseline methodology is also available. Data are based on the publicly available GISTEMP data, the Global Surface Temperature Change data distributed by the National Aeronautics and Space Administration Goddard Institute for Space Studies (NASA-GISS)..\n
**Statistical concepts and definitions:**\n

Statistical standards: Data in the Temperature Change on land domain are not an explicit SEEA variable. Nonetheless, country and regional calculations employ a definition of “Land area” consistent with SEEA Land Use definitions, specifically SEEA CF Table 5.11 “Land Use Classification” and SEEA AFF Table 4.8, “Physical asset account for land use.” The Temperature Change domain of the FAOSTAT Agri-Environmental Indicators section is compliant with the Framework for the Development of Environmental Statistics FDES 2013), contributing to FDES Component 1: Environmental Conditions and Quality, Sub-component 1.1: Physical Conditions, Topic 1.1.1: Atmosphere, climate and weather, Core set/ Tier 1 statistics a.1.\n
**Reference area:**\n
Area of all the Countries and Territories of the world. In 2023: 198 countries and 39 territories.&nbsp; | Code - reference area: FAOSTAT, M49, ISO2 and ISO3 (https://www.fao.org/faostat/en/#definitions).CHAR(13)CHAR(10)CHAR(13)CHAR(10)FAO Global Administrative Unit Layer (GAUL National level – reference year 2014. FAO Geospatial data repository GeoNetwork. Permanent address: https://www.fao.org:80/geonetwork?uuid=f7e7adb0-88fd-11da-a88f-000d939bc5d8
\n**Time coverage:**\n
1961-2023 | Periodicity: Monthly, Seasonal, Yearly\n\n
**Base period:**\n
1951-1980
""")

    # Toggle button to show/hide checkboxes and dataframes
    if 'show_data' not in st.session_state:
        st.session_state.show_data = False

    if st.button("View Original Datasets"):
        st.session_state.show_data = not st.session_state.show_data

    if st.session_state.show_data:
        # Checkboxes for dataframes
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            view_ETC_all = st.checkbox("View: ETC_all")
        with col2:
            view_ETC_all_noflag = st.checkbox("View: ETC_all_noflag")
        with col3:
            view_ETC_all_area_codes = st.checkbox("View: ETC_all_area_codes")
        with col4:
            view_ETC_all_area_flags = st.checkbox("View: ETC_all_area_flags")

        # Display dataframes based on checkboxes
        if view_ETC_all:
            st.write("### ETC_all")
            st.write(ETC_all)
        if view_ETC_all_noflag:
            st.write("### ETC_all_noflag")
            st.write(ETC_all_noflag)
        if view_ETC_all_area_codes:
            st.write("### ETC_all_area_codes")
            st.write(ETC_all_area_codes)
        if view_ETC_all_area_flags:
            st.write("### ETC_all_area_flags")
            st.write(ETC_all_area_flags)

    # Button for final dataset "Environmental Temperature Change"
    session_state = st.session_state
    if "show_etc_checkboxes" not in session_state:
        session_state.show_etc_checkboxes = False
    if "show_etc_df" not in session_state:
        session_state.show_etc_df = False
    if "show_etc_stats" not in session_state:
        session_state.show_etc_stats = False

    show_etc_checkboxes = st.button("Final Dataset: Environmental Temperature Change")
    if show_etc_checkboxes:
        session_state.show_etc_checkboxes = not session_state.show_etc_checkboxes

    if session_state.show_etc_checkboxes:
        # Checkbox to view DataFrame
        view_df_etc = st.checkbox("View DataFrame")
        session_state.show_etc_df = view_df_etc

        if view_df_etc:
            st.write("### Final Dataset on Environmental Temperature Change")
            st.write(ETC_cleaned)

        # Checkbox to view statistics
        view_stats_etc = st.checkbox("1961 - 2023 Overall Change")
        session_state.show_etc_stats = view_stats_etc

        if view_stats_etc:
            # List of regions
            regions = ["Americas", "Asia", "Europe", "Africa", "Oceania", "World"]

            # List to store results
            results = []

            for region in regions:
                region_data = ETC_cleaned[ETC_cleaned['Area'] == region].copy()
                temp_1961 = region_data[region_data['Year'] == 1961]['Temp Change'].values[0]
                temp_2023 = region_data[region_data['Year'] == 2023]['Temp Change'].values[0]

                # Calculate total difference of change between 2023 and 1961
                total_diff = temp_2023 - temp_1961

                # Calculate year-on-year difference
                region_data['Yearly Change'] = region_data['Temp Change'].diff()

                # Calculate the average difference of change between two years
                avg_diff = region_data['Yearly Change'].mean()

                # Calculate the median difference of change between two years
                median_diff = region_data['Yearly Change'].median()

                # Calculate the standard deviation for year-on-year difference
                std_dev_diff = region_data['Yearly Change'].std()

                # Append results to the list
                results.append({
                    'Region': region,
                    'Temp Change 1961 (°C)': temp_1961,
                    'Temp Change 2023 (°C)': temp_2023,
                    'Total Difference (2023 - 1961)': total_diff,
                    'Average Difference': avg_diff,
                    'Median Difference': median_diff,
                    'Standard Deviation': std_dev_diff
                })

            results_df = pd.DataFrame(results)

            # Display statistics table
            st.write("### Overall change in global temperature by continent")
            st.write(results_df)

    # Define list of regions
    regions = ["Americas", "Asia", "Europe", "Africa", "Oceania", "World"]

    # Function to filter data based on selected region and year range
    def filter_data(region, start_year, end_year):
        if region == "World":
            # Calculate overall temperature change by year
            region_data = ETC_cleaned.groupby('Year')['Temp Change'].mean().reset_index()
        else:
            # Filter data for the selected region and year range
            region_data = ETC_cleaned[(ETC_cleaned['Area'] == region) & 
                                      (ETC_cleaned['Year'] >= start_year) & 
                                      (ETC_cleaned['Year'] <= end_year)]
        return region_data

    # UI
    st.write("##### Regional temperature change over time")

    # Slider for selecting range of years
    start_year, end_year = st.slider("Select range of years", min_value=1961, max_value=2023, value=(1961, 2023))

    # Dropdown for selecting continent or world view
    selected_continent = st.selectbox("Select continent or world view", regions, index=len(regions)-1)

    # Filter data based on selected continent and year range
    filtered_data = filter_data(selected_continent, start_year, end_year)

    # Create an interactive line chart with hover tooltips
    fig = px.line(filtered_data, x='Year', y='Temp Change', labels={'Temp Change': 'Temperature Change (°C)'}, 
                  title=f'Temperature Change Over Time - {selected_continent}',
                  hover_data={'Year': True, 'Temp Change': True})
    fig.update_traces(mode='lines+markers', hovertemplate='%{x}<br>Year: %{customdata[0]}<br>Temp Change: %{y:.2f}°C')
    fig.update_xaxes(title_text='Year')
    fig.update_yaxes(title_text='Temperature Change (°C)')
    st.plotly_chart(fig)

    # Prepare data for the boxplot
    boxplot_data = []
    for region in regions:
        if region == "World":
            boxplot_data.append(ETC_cleaned)
        else:
            boxplot_data.append(ETC_cleaned[ETC_cleaned['Area'] == region])

    # Create Box plot
    fig_boxplot = go.Figure()
    for i, region_data in enumerate(boxplot_data):
        fig_boxplot.add_trace(go.Box(
            y=region_data['Temp Change'],
            name=regions[i],
            boxmean='sd',
            marker=dict(color=px.colors.qualitative.Plotly[i]),
            boxpoints='all',
            jitter=0.5,  # Spread out data points
            width=0.4  # Adjust box width
        ))
    
    # Update layout for better visualization
    fig_boxplot.update_layout(
        title="Surface Temperature Anomalies by Continent",
        xaxis=dict(title='Continent'),
        yaxis=dict(title='Temperature Change (°C)'),
        boxmode='group',  # group boxes of different traces
        showlegend=True,
        legend=dict(title='Continent')
    )
    
    # Show the plot
    st.plotly_chart(fig_boxplot)

    # Top 10 Regions Bar Chart
    ETC_2023 = ETC_cleaned[ETC_cleaned['Year'] == 2023]
    
    # Sort data by temperature change in descending order
    ETC_2023_sorted = ETC_2023.sort_values(by='Temp Change', ascending=False)
    
    # Select top 10 regions with the highest temperature anomaly
    top_10_regions = ETC_2023_sorted.head(10)
    
    # Plot bar chart
    fig_bar = px.bar(top_10_regions, x='Temp Change', y='Area', orientation='h', 
                     title='Top 10 Regions with Highest Temperature Anomaly in 2023',
                     labels={'Temp Change': 'Temperature Anomaly (°C)', 'Area': 'Region'})
    
    # Update layout for better visualization
    fig_bar.update_layout(
        xaxis=dict(title='Temperature Anomaly (°C)'),
        yaxis=dict(title='Region')
    )
    
    # Show the plot
    st.plotly_chart(fig_bar)

    # Function to plot hot/cold temperature categories
    def plot_temperature_categories(ETC_cleaned):
        st.write("### Temperature Categories Over Years")

        # Function to calculate temperature categories
        def calculate_temperature_categories(df, year):
            data_year = df[df['Year'] == year]
            lower_than_2_sd = (data_year['Temp Change'] < -2 * data_year['Std Dev']).sum()
            within_2_sd = ((data_year['Temp Change'] >= -2 * data_year['Std Dev']) & (data_year['Temp Change'] <= 2 * data_year['Std Dev'])).sum()
            greater_than_2_sd = (data_year['Temp Change'] > 2 * data_year['Std Dev']).sum()
            return lower_than_2_sd, within_2_sd, greater_than_2_sd

        # List of years to analyze
        years = [1961, 1991, 2021]
        colors = ['blue', 'green', 'red']

        fig = go.Figure()

        for year, color in zip(years, colors):
            cold, normal, warm = calculate_temperature_categories(ETC_cleaned, year)
            categories = ['Cold', 'Normal', 'Warm']
            counts = [cold, normal, warm]
            
            fig.add_trace(go.Bar(
                x=categories,
                y=counts,
                name=str(year),
                marker_color=color
            ))

        fig.update_layout(
            title="Number of Countries by Temperature Category",
            xaxis_title="Temperature Category",
            yaxis_title="Number of Countries",
            barmode='group'
        )

        st.plotly_chart(fig)

    # Call the function to plot the temperature categories
    plot_temperature_categories(ETC_cleaned)



       

###



