import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
import io
import pickle
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import json


df_2021 = pd.read_csv("Jun24da_WorldHapiness/world-happiness-report-2021.csv")
df_pre2021 = pd.read_csv("Jun24da_WorldHapiness/world-happiness-report.csv")
df_all_notencoded = pd.read_csv("Jun24da_WorldHapiness/df_all_notencoded.csv")


### SIDEBAR CONFIGURATION ###

st.sidebar.image("Jun24da_WorldHapiness/happiness_pic_1.jpg", width=300)

st.sidebar.title("Table of contents")
pages=["Introduction", "Data Exploration and Preprocessing", "Data Visualization", "Data Modelling", "Prediction", "Conclusion"]
page=st.sidebar.radio("", pages)

st.sidebar.markdown("<br><br><br><br>", unsafe_allow_html=True)

#Path to logo file
logo_url = 'https://assets-datascientest.s3-eu-west-1.amazonaws.com/notebooks/looker_studio/logo_datascientest.png'

st.sidebar.markdown(
  f"""
  <style>
    .sidebar-content {{
        text-align: center;}}

    .sidebar-content img {{
        width: 50px; 
        margin: 10px 0;}}

    </style>
    <div class="sidebar-content">
      <p><i><b>A Datascientest project carried out by Delphine Parmentier and </br> Evangelos Ziogas</b></i></p>
      <p><i><b>Project mentor: Tarik Anouar</b></i></p>
      <img src="{logo_url}" alt="Logo">
      <p>Data Analyst Training - June 2024 Cohort</p>
    </div>
    """,
    unsafe_allow_html=True)


### PAGE 1 - INTRODUCTION ###

if page == pages[0] : 
  title = '<h1 style="color: #008B8B;">Unveiling Global Happiness: A Data-Driven Analysis</h1>'
  st.markdown(title, unsafe_allow_html=True)
  header_html = "<h2 style='color: black;'>Welcome aboard the Happiness Study! </h2>"
  st.markdown(header_html, unsafe_allow_html=True)
  
  st.write ("Are you curious about what makes people happy around the world? Our data analytics project dives into global happiness, "
  "revealing what contributes to joy and contentment across different countries.  By analyzing key data, "
  "we uncover trends and insights that paint a clearer picture of happiness worldwide. Join us as we explore the factors" 
  " that drive happiness and see what makes the world smile!")

   

#### PAGE 2 - DATA EXPLORATION AND CLEANING #####

if page == pages[1] : 
  header_html = "<h1 style='color: #b30086;'>Data Exploration and Preprocessing </h1>"
  st.markdown(header_html, unsafe_allow_html=True)
  st.subheader("Presentation of the datasets used")
  st.markdown ("- **Dataset n.1: data for years 2005 to 2020 ('df_pre2021')**")
  st.dataframe(df_pre2021.head())
  st.write(df_pre2021.shape)

#Capturing df_pre2021.info() output
  with st.expander("**Dataset n.1 specifications**"):
    buffer = io.StringIO()
    df_pre2021.info(buf=buffer)
    info_string = buffer.getvalue()
    st.text(info_string)
 
  st.markdown("<br>", unsafe_allow_html=True)

  st.markdown ("- **Dataset n.2: data for the year 2021 ('df_2021')**")
  st.dataframe(df_2021.head())
  st.write(df_2021.shape)

 #Capturing df_2021.info() output
  with st.expander("**Dataset n.2 specifications**"):
    buffer = io.StringIO()
    df_2021.info(buf=buffer)
    info_string = buffer.getvalue()
    st.text(info_string)

  st.markdown("<br>", unsafe_allow_html=True)

  #Renaming columns and merging datasets
  df_2021.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)
  df_pre2021.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)
  df_pre2021.rename(columns={'Life_Ladder': 'Ladder_score', 'Log_GDP_per_capita': 'Logged_GDP_per_capita',
                             'Healthy_life_expectancy_at_birth':'Healthy_life_expectancy'}, inplace=True)
  
  df_all = df_pre2021.merge(right = df_2021, on=['Country_name', 'Ladder_score', 'Logged_GDP_per_capita', 'Social_support', 'Healthy_life_expectancy', 
                                                'Freedom_to_make_life_choices', 'Generosity', 'Perceptions_of_corruption'], how = 'outer')
  
  st.markdown ("- **Dataset n.3: merged data to get years 2005 to 2021 included ('df_all')**")
  st.dataframe(df_all.head())
  st.write(df_all.shape)

  
  #Capturing df_all.info() output
  with st.expander("**Dataset n.3 specifications**"):
    buffer = io.StringIO()
    df_all.info(buf=buffer)
    info_string = buffer.getvalue()
    #st.text(info_string)

  #Split the info_string into lines for easier processing
    info_lines = info_string.splitlines()

  #Define the specific line or condition you want to highlight
    keywords_to_highlight = ['Positive_affect', 'Negative_affect', 'Standard_error_of_ladder_score', 'upperwhisker', 'lowerwhisker', 'Ladder_score_in_Dystopia',
                             'Explained_by:_Log_GDP_per_capita','Explained_by:_Social_support', 'Explained_by:_Healthy_life_expectancy',
                              'Explained_by:_Freedom_to_make_life_choices', 'Explained_by:_Generosity', 'Explained_by:_Perceptions_of_corruption','Dystopia_+_residual' ]
    
    highlighted_lines = []
    for line in info_lines:
       if any (keyword in line for keyword in keywords_to_highlight):
          highlighted_lines.append(f"{line} ★")
       else:
          highlighted_lines.append(line)

  #Join the lines back together
    highlighted_info_string = "\n".join(highlighted_lines)

  #Display the info in Streamlit using markdown with HTML support
    st.markdown(f"```text\n{highlighted_info_string}\n```")


  #Description of the issues
  st.markdown('★ <i>variables removed during data cleaning.</i>', unsafe_allow_html=True)
  st.write("❗Several issues are observed in this merged dataset, that require a data cleaning step: irrelevant columns, missing values"
           " and varying data scales.")

  st.markdown("<br>", unsafe_allow_html=True)

  st.subheader("Data cleaning")


  #Cleaning

  #Creating a separate dataset for Dystopia only

  columns_extract = ["Ladder_score_in_Dystopia",
           "Explained_by:_Log_GDP_per_capita",
           "Explained_by:_Social_support",
           "Explained_by:_Healthy_life_expectancy",
           "Explained_by:_Freedom_to_make_life_choices",
           "Explained_by:_Generosity",
           "Explained_by:_Perceptions_of_corruption",
           "Dystopia_+_residual"]

  dystopia = df_all[columns_extract]
 
  #Removing Dystopia from df_all 

  columns_remove = ["Ladder_score_in_Dystopia",
                    "Explained_by:_Log_GDP_per_capita",
                    "Explained_by:_Social_support",
                    "Explained_by:_Healthy_life_expectancy",
                    "Explained_by:_Freedom_to_make_life_choices", 
                    "Explained_by:_Generosity", 
                    "Explained_by:_Perceptions_of_corruption",
                    "Dystopia_+_residual", 
                    "Standard_error_of_ladder_score",
                    "upperwhisker","lowerwhisker",
                    "Positive_affect","Negative_affect"]

  df_all = df_all.drop(columns=columns_remove)
  
  #Reordering the columns:
  new_order = ['Regional_indicator','Country_name', 'year', 'Ladder_score', 'Logged_GDP_per_capita',
             'Social_support', 'Healthy_life_expectancy', 'Freedom_to_make_life_choices',
             'Generosity', 'Perceptions_of_corruption']

  df_all = df_all.reindex(columns=new_order)
  st.write("**1. Removing irrelevant columns from df_all and reordering the columns logically:**")
  
  #Capturing df_all.info() output
  buffer = io.StringIO()
  df_all.info(buf=buffer)
  info_string = buffer.getvalue()
  
  #Split the info_string into lines for easier processing
  info_lines = info_string.splitlines()

  #Define the specific line or condition you want to highlight
  keywords_to_highlight = ['Country_name', 'Ladder_score']
    
  highlighted_lines = []
  for line in info_lines:
       if any (keyword in line for keyword in keywords_to_highlight):
          highlighted_lines.append(f"{line} ✔")
       else:
          highlighted_lines.append(line)

 
  #Join the lines back together
  highlighted_info_string = "\n".join(highlighted_lines)

  #Display the info in Streamlit using markdown with HTML support
  st.markdown(f"```text\n{highlighted_info_string}\n```")


  st.markdown("<br><br>", unsafe_allow_html=True)
  

  #Handling the missing values

  st.write("**2. Handling missing values:**")
  st.write("**→ Filling the missing values for the year with '2021' :**")
  df_all['year'].fillna(2021, inplace=True)

  #Capture df_pre2021.info() output
  with st.expander("**df_pre2021 specifications**"):
    buffer = io.StringIO()
    df_pre2021.info(buf=buffer)
    info_string = buffer.getvalue()
    st.text(info_string)


  #Capture df_all.info() output
  with st.expander("**df_all after filling missing year values**"):
    buffer = io.StringIO()
    df_all.info(buf=buffer)
    info_string = buffer.getvalue()
    
 #Split the info_string into lines for easier processing
    info_lines = info_string.splitlines()

 #Define the specific line or condition you want to highlight
    keywords_to_highlight = ['Country_name', 'year', 'Ladder_score']
    
    highlighted_lines = []
    for line in info_lines:
       if any (keyword in line for keyword in keywords_to_highlight):
          highlighted_lines.append(f"{line} ✔")
       else:
          highlighted_lines.append(line)

  #Join the lines back together
    highlighted_info_string = "\n".join(highlighted_lines)

  #Display the info in Streamlit using markdown with HTML support
    st.markdown(f"```text\n{highlighted_info_string}\n```")

  st.markdown("<br>", unsafe_allow_html=True)

  #Looking at the distributions of quantitative variables to define how the missing values should be replaced:
  
  st.write("**→ Filling missing values for the quantitative variables:**")
  st.write("In order to determine the optimal method for replacing missing values - either by the mean or the median - "
           "we assessed the distribution of each variable:")

  
  #Plotting distributions of quantitative variables

  fig = make_subplots(rows=3, cols=2, subplot_titles=(""))
  color = '#66ccff'

  #First row of histograms
  fig1 = px.histogram(df_all, x='Logged_GDP_per_capita', nbins=30, title='Distribution of Logged_GDP_per_capita')
  fig1.update_traces(marker_color=color)
  fig1.update_layout(bargap=0.1)

  fig2 = px.histogram(df_all, x='Social_support', nbins=30, title='Distribution of Social_support')
  fig2.update_traces(marker_color=color)
  fig2.update_layout(bargap=0.1)

  col1, col2 = st.columns(2)

  with col1:
    st.plotly_chart(fig1, use_container_width=True)

  with col2:
    st.plotly_chart(fig2, use_container_width=True)

  #Second row of histograms
  fig3 = px.histogram(df_all, x='Healthy_life_expectancy', nbins=30, title='Distribution of Healthy_life_expectancy')
  fig3.update_traces(marker_color=color)
  fig3.update_layout(bargap=0.1)

  fig4 = px.histogram(df_all, x='Freedom_to_make_life_choices', nbins=30, title='Distribution of Freedom_to_make_life_choices')
  fig4.update_traces(marker_color=color)
  fig4.update_layout(bargap=0.1)

  col1, col2 = st.columns(2)

  with col1:
    st.plotly_chart(fig3, use_container_width=True)

  with col2:
    st.plotly_chart(fig4, use_container_width=True)

  #Third row of histograms
  fig5 = px.histogram(df_all, x='Generosity', nbins=30, title='Distribution of Generosity')
  fig5.update_traces(marker_color=color)
  fig5.update_layout(bargap=0.1)

  fig6 = px.histogram(df_all, x='Perceptions_of_corruption', nbins=30, title='Distribution of Perceptions_of_corruption')
  fig6.update_traces(marker_color=color)
  fig6.update_layout(bargap=0.1)

  col1, col2 = st.columns(2)

  with col1:
    st.plotly_chart(fig5, use_container_width=True)

  with col2:
    st.plotly_chart(fig6, use_container_width=True)


  
  st.write("✦ We can see in all cases that distributions are not symetrical. For this reason, we choose to impute missing values " 
           "with the median of each variable, which is more robust to skewed distributions and less sensitive to outliers than the mean.")
  
  #Filling the missing values with the median:

  #List of columns where missing values are to be replaced with median
  columns_to_fill = ['Logged_GDP_per_capita', 'Social_support', 'Healthy_life_expectancy', 'Freedom_to_make_life_choices', 
                   'Generosity', 'Perceptions_of_corruption']

  #Replacing missing values with the median in specified columns
  for col in columns_to_fill:
    median_value = df_all[col].median()
    df_all[col].fillna(median_value, inplace=True)

  with st.expander("**df_all after filling missing values for quantitative variables**"):
    buffer = io.StringIO()
    df_all.info(buf=buffer)
    info_string = buffer.getvalue()

  #Split the info_string into lines for easier processing
    info_lines = info_string.splitlines()

  #Define the specific line or condition you want to highlight
    keywords_to_highlight = ['Country_name', 'year', 'Ladder_score', 'Logged_GDP_per_capita', 'Social_support', 'Healthy_life_expectancy', 
                             'Freedom_to_make_life_choices', 'Generosity',' Perceptions_of_corruption']
    
    highlighted_lines = []
    for line in info_lines:
       if any (keyword in line for keyword in keywords_to_highlight):
          highlighted_lines.append(f"{line} ✔")
       else:
          highlighted_lines.append(line)

  #Join the lines back together
    highlighted_info_string = "\n".join(highlighted_lines)


  
  #Display the info in Streamlit using markdown with HTML support
    st.markdown(f"```text\n{highlighted_info_string}\n```")


  st.markdown("<br>", unsafe_allow_html=True)

  #Filling missing continent values based on the existing country-region pairs in the dataset

  st.write("**→ Filling missing values for the regional indicator:**")
  st.write("Finally, for the regional indicator the missing values were completed in 2 steps thanks to AI:<br>"
    "**1-** Filling the missing values based on existing country-region pairs.<br>"
    "**2-** Any remaining country without a corresponding continent after the first step was mapped by the AI system to the "
    "most appropriate region available in the dataset.",
    unsafe_allow_html=True)
  
  df_all['Regional_indicator'] = df_all.groupby('Country_name')['Regional_indicator'].transform(lambda x: x.ffill().bfill())

  country_to_continent = {
    'Angola': 'Sub-Saharan Africa',
    'Belize': 'Latin America and Caribbean',
    'Bhutan': 'South Asia',
    'Central African Republic': 'Sub-Saharan Africa',
    'Congo (Kinshasa)': 'Sub-Saharan Africa',
    'Cuba': 'Latin America and Caribbean',
    'Djibouti': 'Sub-Saharan Africa',
    'Guyana': 'Latin America and Caribbean',
    'Oman': 'Middle East and North Africa',
    'Qatar': 'Middle East and North Africa',
    'Somalia': 'Sub-Saharan Africa',
    'Somaliland region': 'Sub-Saharan Africa',
    'South Sudan': 'Sub-Saharan Africa',
    'Sudan': 'Sub-Saharan Africa',
    'Suriname': 'Latin America and Caribbean',
    'Syria': 'Middle East and North Africa',
    'Trinidad and Tobago': 'Latin America and Caribbean'
}

  def fill_missing_continents(row):
    if pd.isnull(row['Regional_indicator']):
       return country_to_continent.get(row['Country_name'], None)
    return row['Regional_indicator']

  df_all['Regional_indicator'] = df_all.apply(fill_missing_continents, axis=1)

  with st.expander("**df_all after filling missing regional indicator values**"):
     buffer = io.StringIO()
     df_all.info(buf=buffer)
     info_string = buffer.getvalue()
     st.text(info_string)

  st.markdown("<br><br>", unsafe_allow_html=True)
  

  #Data Normalization

  st.markdown("**3. Data normalization:**")
    
  with st.expander("**df_all pre-normalization**"):
     st.markdown ("")
     st.dataframe(df_all.head())
     st.write(df_all.shape)

  with st.expander("**df_all post-normalization**"):
     from sklearn.preprocessing import MinMaxScaler
     to_normalize = ['Logged_GDP_per_capita', 'Healthy_life_expectancy', 'Generosity'] 
     scaler = MinMaxScaler()
     df_all[to_normalize] = scaler.fit_transform(df_all[to_normalize])
     st.dataframe(df_all.head())

  st.write(" ✔ The dataset in now clean and complete and can be used for further analysis.")



#### PAGE 3 - DATA VISUALIZATION #####

if page == pages[2] :
  header_html = "<h1 style='color: #b30086;'>Data Visualization </h1>"
  st.markdown(header_html, unsafe_allow_html=True)
  st.subheader("Analysis of the happiness score across regions and countries")
  

  #Plotting global distribution of the ladder score (2005-2021)

  fig = px.box(df_all_notencoded, y='Ladder_score', title='Global distribution of the ladder score')
  fig.update_traces(line=dict(width=1.5, color='#3333cc'), fillcolor='#85a3e0')
  fig.update_layout(
    height=400,  
    width=600,
    title_font_size=20)

  st.plotly_chart(fig)
 
  st.write("✦ The worldwide distribution of the ladder score appears symmetric, indicating a balanced dataset"
          " that fairly represents different levels of well-being or life satisfaction.")
  st.write("Analyzing rankings accross continents and countries, we were able to identify the top 10 happiest countries in the world in 2021:")

  #Sorting df_all_notencoded descending
  df_all_notencoded_sorted = df_all_notencoded.sort_values(by='Ladder_score', ascending = False)

  #Plotting the distribution of the ladder score per continent
  fig = px.box(df_all_notencoded_sorted, x='Regional_indicator', y='Ladder_score',
               color='Regional_indicator',
               color_discrete_sequence=px.colors.qualitative.Plotly,
               title='Ladder score distribution accross continents')
  
  fig.update_traces(marker=dict(size=10, line=dict(width=2, color='#000000')), selector=dict(type='box'))
  
  fig.update_layout(
    height=600,  
    width=800,
    title_font_size=20,
    xaxis_title='Continent',
    yaxis_title='Ladder Score',
    xaxis_tickangle=-45)

  st.plotly_chart(fig)

 
  #Sorting ladder scores descending and calculating mean
  df_2021_sorted = df_2021.sort_values(by='Ladder score', ascending = False)
  continent_avg = df_2021_sorted.groupby('Regional indicator', as_index=False)['Ladder score'].mean()
  continent_avg = continent_avg.sort_values(by='Ladder score', ascending=False)
  
  #Creating two columns to get continent and country graphs side by side:
  col1, col2 = st.columns([3,1])

  #Plotting average ladder scores in 2021 per continent

  with col1:
    fig_continent = px.bar(continent_avg, x='Ladder score', y='Regional indicator', 
                           color='Ladder score', color_continuous_scale='viridis',
                           title='Average ladder score per continent in 2021')
   
    fig_continent.update_layout(showlegend=False, 
                                coloraxis_showscale=False, 
                                height=450, 
                                width=600, 
                                title_font_size=20,
                                xaxis_title='Average Ladder Score', 
                                yaxis_title='Continent',
                                yaxis=dict(autorange="reversed"))

    st.plotly_chart(fig_continent)
    
    
  #Plotting ladder scores for the top 10 happiest countries in 2021

  df_2021_sorted = df_2021.sort_values(by='Ladder score', ascending = False)
  df_2021_top10 = df_2021_sorted.head(10)

  with col2:
    fig_country = px.bar(df_2021_top10, x='Ladder score', y='Country name', orientation='h',
                         color='Ladder score', color_continuous_scale='viridis',
                         title='Top 10 Happiest Countries by Ladder Score in 2021')
  
    fig_country.update_layout(showlegend=False, 
                              coloraxis_showscale=False, 
                              height=450, 
                              width=600, 
                              title_font_size=20, 
                              xaxis_title='Ladder Score', 
                              yaxis_title='Country', 
                              yaxis=dict(autorange="reversed"))

    st.plotly_chart(fig_country)

 
  st.write("✦ Western Europe, North America and New Zealand are ranked among the happiest regions in 2021," 
           " with Finland and Denmark being the leading countries.")
  
  #Plotting historical trend (2005-2021)

  st.write("To gain a comprehensive global perspective on the evolution of the scores, "
           "we also examined the historical trend from 2005 to 2021.")
  
  df_2021 = df_all_notencoded[df_all_notencoded["year"] == 2021]
  top_10 = df_2021.sort_values(by='Ladder_score', ascending=False).head(10)
  top_countries = top_10["Country_name"].tolist()
  df_top_10 = df_all_notencoded[df_all_notencoded['Country_name'].isin(top_countries)]
  df_top_10_sorted = df_top_10.sort_values(by=['Country_name', 'year'])

  fig = px.line(
    df_top_10_sorted, 
    x='year', 
    y='Ladder_score', 
    color='Country_name',  
    markers=True, 
    title='Evolution of Ladder Score Values Over Years for the Top 10 Countries',
    color_discrete_sequence=px.colors.qualitative.Plotly)
  
  fig.update_layout(
    height=600,
    width=900,
    title_font_size=20,
    xaxis_title='Year',
    yaxis_title='Ladder Score',
    xaxis=dict(
      tickmode='linear',  
      showgrid=True,  
    ),
    yaxis=dict(
      showgrid=True,
    ),
    legend_title='Country',
    legend=dict(
      title='Country',
      x=1.05,  
      y=1,
      traceorder='normal'
    ),
    xaxis_tickangle=-45,  
  )

  st.plotly_chart(fig, use_container_width=True)
 
  st.write("✦ While Denmark and Finland still appear to be among the happiest countries over time, "
         "we can also observe from this timeframe that a global decrease of the ladder score was observed for all countries in 2014."
          " One might attribute this decline to socio-economic factors, leading to instability, economic hardship, and insecurity worldwide.")


  st.write("✦ We have also compared the 10 least happy countries with the hypothetical nation of Dystopia, and noticed that "
           "nations like Afghanistan, Zimbabwe and Rwanda are extremely close to the Ladder Score of Dystopia. Indeed, "
           "most of the countries depicted here suffer from political turmoil and severe economic conditions," 
           " meaning they are very close to the hypothetical worst nation.")

  #Comparison of ladder scores with Dystopia

  worst_10 = df_2021.sort_values(by='Ladder_score', ascending=True).head(10)
  dystopia_score = 2.43

  fig_dys = px.bar(worst_10, x='Country_name', y='Ladder_score', 
                           color='Ladder_score', color_continuous_scale='viridis',
                           title='Comparison of the 10 least happy countries to Dystopia')
   
  #Adding horizontal line for the dystopia score
  fig.add_shape(
    type='line',
    x0=worst_10['Country_name'].iloc[0],
    y0=dystopia_score,
    x1=worst_10['Country_name'].iloc[-1],
    y1=dystopia_score,
    line=dict(color='red', dash='dash'),
    xref='x',
    yref='y')

  #Adding annotation for the dystopia score
  fig_dys.add_annotation(
    x=worst_10['Country_name'].iloc[len(worst_10) // 2],
    y=dystopia_score,
    text="Dystopia's Score",
    showarrow=False,
    yshift=10,
    font=dict(color='red'))

  fig_dys.update_layout(showlegend=False, 
                        coloraxis_showscale=False, 
                        height=450, 
                        width=600, 
                        title_font_size=20,
                        xaxis_title='Country', 
                        yaxis_title='Ladder Score',
                        legend_title_text='',
                        shapes=[dict(
                          type='line',
                          x0=worst_10['Country_name'].iloc[0],
                          y0=dystopia_score,
                          x1=worst_10['Country_name'].iloc[-1],
                          y1=dystopia_score,
                          line=dict(color='red', dash='dash')
                        )]
  )

  st.plotly_chart(fig_dys)

  st.write(" ")
  st.subheader("Relationship of the happiness score to external factors")

  #Plotting the heatmap 

  factors = ['Ladder_score', 'Logged_GDP_per_capita', 'Social_support', 'Healthy_life_expectancy', 'Freedom_to_make_life_choices','Generosity', 'Perceptions_of_corruption']
  df_fact = df_all_notencoded[factors]
  corr = df_fact.corr()
  corr_rounded = corr.round(2)

  fig = px.imshow(
    corr_rounded,
    color_continuous_scale='viridis',
    width =800, height = 600,
    title="Correlation between the 6 factors known to influence life evaluation, and the ladder score",
    text_auto=True)

  fig.update_layout(
    xaxis_title='Factors',
    yaxis_title='Factors',
    title_font_size=20,
    coloraxis_colorbar=dict(title='Correlation'))
  
  st.plotly_chart(fig)

 
  st.write("✦ Logged GDP per capita and healthy life expectancy are showing the highest values on the correlation matrix" 
         " in relation to the ladder score, meaning that those are the ones influencing the most the life score.")

  
  #Plotting the ladder score for the three happiest vs three least happy countries

  list_2021 = df_all_notencoded[df_all_notencoded["year"] == 2021]
  top_3 = list_2021.sort_values(by='Ladder_score', ascending=False).head(3)
  least_3 = list_2021.sort_values(by='Ladder_score', ascending=True).head(3)
  
  fig = make_subplots(rows=1, cols=2, subplot_titles=("Comparison of GDP", "Comparison of Healthy Life Expectancy"))
  
 #Plotting logged GDP per capita and healthy life comparison for the three least vs three happiest countries:

 #First subplot: GDP comparison
  fig.add_trace(go.Bar(
    x=top_3["Country_name"],
    y=top_3["Logged_GDP_per_capita"],
    name='GDP (Top 3)',
    marker_color='#98d83e'), row=1, col=1)

  fig.add_trace(go.Bar(
    x=least_3["Country_name"],
    y=least_3["Logged_GDP_per_capita"],
    name='GDP (Least 3)',
    marker_color='#440154'), row=1, col=1)

  #Second subplot: Healthy Life Expectancy comparison
  fig.add_trace(go.Bar(
    x=top_3["Country_name"],
    y=top_3["Healthy_life_expectancy"],
    name='Life Expectancy (Top 3)',
    marker_color= '#98d83e'), row=1, col=2)

  fig.add_trace(go.Bar(
    x=least_3["Country_name"],
    y=least_3["Healthy_life_expectancy"],
    name='Life Expectancy (Least 3)',
    marker_color="#440154"), row=1, col=2)

# Update layout
  fig.update_layout(
    title='Comparison of GDP and Healthy Life Expectancy: Happiest vs Least Happy Countries (2021)',
    xaxis_title='Countries',
    yaxis_title='Normalized number of years',
    barmode='group',
    width=800,
    height=500,
    showlegend=False)

# Update x and y axis labels for both subplots
  fig.update_xaxes(title_text="Countries", row=1, col=1)
  fig.update_yaxes(title_text="Logged GDP per Capita", row=1, col=1)

  fig.update_xaxes(title_text="Countries", row=1, col=2)
  fig.update_yaxes(title_text="Normalized number of years", row=1, col=2)

# Display the plot in Streamlit
  st.plotly_chart(fig)

  st.write("✦ Finally, upon comparing the healthy life expectancy across the three happiest countries and the three least happy countries,"
         " we could see striking disparity emerging, highlighting strong economic inequalities and difficulties in accessing healthcare.")


#### PAGE 4 - DATA MODELLING #####

if page == pages[3] :
  header_html = "<h1 style='color: #b30086;'>Data Modelling </h1>"
  st.markdown(header_html, unsafe_allow_html=True)
  st.subheader("Chosen Models: Linear Regression, Decision Tree, Random Forest Tree and Gradient Boosting ")
  
  st.write("")
   
  if st.button("Scores and Metrics") :
      data = {
        'Models': ['Linear Regression', 'DecisionTree Regressor', 'Random Forest', 'Gradient Boosting'],
        'R² train': [0.75, 0.76, 0.77, 0.9],
        'R² test': [0.73, 0.69, 0.73, 0.84],
        'MAE train': [0.43, 0.44, 0.41, 0.27],
        'MAE test': [0.45, 0.47, 0.45, 0.34],
        'MSE train': [0.31, 0.31, 0.28, 0.12],
        'MSE test': [0.56, 0.37, 0.33, 0.2],
        'RMSE train': [0.33, 0.56, 0.53, 0.35],
        'RMSE test': [0.58, 0.60, 0.57, 0.45]
                } 
      table = pd.DataFrame(data)
      GD_index = table[table["Models"] == "Gradient Boosting"].index
      style = table.style.apply(lambda x:['background: #27dce0' if x.name in GD_index else '' for i in x], axis=1)
      
      st.table(style)
      
  if st.button("Gradient Boosting visualization") :
    st.subheader("Residuals of the Gradient Boosting model")
    st.write("")
    image = Image.open("residuals.png")
    st.image(image, width=800)

    st.write("")

  if st.button("Feature Importance") :
    st.subheader("Feature importance of the Gradient Boosting model")
    image2 = Image.open("feature_importance.png")
    st.image(image2, width=800)        
    

#### PAGE 5 - PREDICTION #####

if page == pages[4] :
  header_html = "<h1 style='color: #b30086;'>Prediction </h1>"
  st.markdown(header_html, unsafe_allow_html=True)

  def load_model():
        with open('gb_reg.pkl', 'rb') as file:
            gb_reg = pickle.load(file)
        return gb_reg
      
  def load_dict():
        with open("feature_min_max.json", "r") as json_file:
            min_max_dict = json.load(json_file)
        return min_max_dict    

  def load_target_mapping():
        with open('target_variable.json', 'r') as json_file:
            target = json.load(json_file)
        return target
      
  gb_reg = load_model()
  target = load_target_mapping()
  min_max_dict = load_dict()
    
  # Only include the desired features and exclude the target variable
  include_features = ["Logged_GDP_per_capita", "Social_support", "Healthy_life_expectancy", "Freedom_to_make_life_choices"]

  characteristics_list = []
  for feature, limits in min_max_dict.items():
        if feature in include_features:
            characteristics = st.slider(
                f"{feature}", 
                float(limits['min']), 
                float(limits['max']), 
                float((limits['min'] + limits['max']) / 2)
            )
            characteristics_list.append(characteristics)
        elif feature != "Ladder_score":
            # Set default value for excluded features
            characteristics_list.append(float((limits['min'] + limits['max']) / 2))
   
  characteristics = np.array([characteristics_list])
  prediction = gb_reg.predict(characteristics)
  st.markdown(""" <style> .big-font {font-size:24px !important;}.green-font {font-size:24px !important; color: green;}  </style>""",
  unsafe_allow_html=True) 
  st.markdown('<p class="big-font">Predicted Ladder Score: &nbsp;&nbsp; <span class="green-font">{}</span></p>'.format(prediction[0]), unsafe_allow_html=True)

  
#### PAGE 6 - CONCLUSION #####

if page == pages[5] :
  header_html = "<h1 style='color: #b30086;'>Conclusion</h1>"
  st.markdown(header_html, unsafe_allow_html=True)

  st.write("Happiness is influenced by a lot of variables, including economic indicators, social support, life expectancy," 
           " freedom to make life choices, generosity in a society and perceptions of corruption, among others.")
  
  st.write("")
  st.write("We can observe that GDP per capita, life expectancy and the social support are the most important variables to consider"
           " for happiness. With money (i.e Log GDP per Capita) being the most essential variable to account for happiness, according to our model.")
  st.write("However happiness is very complex to measure and more attention needs to be brought into how it can properly be measured.")

  st.write("")
  image3 = Image.open("Jun24da_WorldHapiness/money_everything.jpg")
  st.image(image3, width=700)

  if st.button("Special Thanks") :
      st.write("Thank you for your attention!")
      
      st.write("")
      st.write("- Special thanks to the team at Datascientest for giving us the opportunity to work on this project.")
      st.write("")
      
      st.write("- A thank you to Tarik, our mentor in this project.")
      st.write("")
