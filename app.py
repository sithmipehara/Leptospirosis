import streamlit as st
import pandas as pd
import pymongo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import folium 
from streamlit_folium import folium_static
import json
from matplotlib import cm, colors
from streamlit_option_menu import option_menu
from PIL import Image
import altair as alt
from datetime import datetime

# Set the theme to dark
st.set_page_config(page_title="Local Leptospirosis Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* Style for individual metric boxes */
.metric-box {
    padding: 20px;
    border-radius: 8px;
    color: white;
    text-align: center;
    font-size: 16px;
    font-weight: bold;
}

.container {
    padding: 10px;  /* Reduced padding */
    border-radius: 0px;
    height: 160px; /* Adjust height to auto for flexibility */
    color: white; /* Text color */
    margin: 5px;  /* Reduced margin */
}
.donut-container {
    padding: 10px;  /* Reduced padding */
    border-radius: 0px;
    background-color: #0E1117; 
}
.chart-container {
    padding: 10px;  /* Reduced padding */
    border-radius: 0px;
}

.box1 { background-color: #5cd6d6;color: #000000; }
.box2 { background-color: #668cff;color: #000000; }
.box3 { background-color: #ffcc66; color:#000000;}
.box4 { background-color: #ffd480;color: #000000; }

.stSelectbox {
    transition: background-color 0.3s;
    line-height: 50px;
    height:140px;
    text-align: center;
    background-color: #3c3c44; /* Change this to your desired color */
    border-radius: 20px; /* Optional: rounded corners */
    padding: 10px; /* Optional: padding inside the selectbox */
}

.custom-title {
    font-size: 36px;  /* Change to your desired font size */
    color: #5cd6d6; 
    text-align: center ;/* Change to your desired font color */
}

.custom-content {
    font-size: 20px;   /* Change to your desired font size for content */
    color: white;   
    text-align: center ;/* Change to your desired font color for content */
}

</style>
""", unsafe_allow_html=True)

# MongoDB connection details
mongo_url = "mongodb+srv://sithmi_pehara:genius2000@cluster0.y5lkbfe.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(mongo_url)
db = client['leptospirosis_srilanka']
collection = db['Yearly_data']
weekly_data_collection = db['Weekly_data']

# Load data from MongoDB
@st.cache_data
def load_data():
    data = list(collection.find())
    df = pd.DataFrame(data)
    return df

df = load_data()

# Load weekly data from MongoDB
@st.cache_data
def load_weekly_data():
    data = list(weekly_data_collection.find())
    df = pd.DataFrame(data)
    return df

weekly_df = load_weekly_data()

# Prepare for Sri Lanka data
SriLanka_data_collection = db['Yearly_data']
SriLanka_data = pd.DataFrame(list(SriLanka_data_collection.find()))

# Change 'Year' and 'PDF_ID' columns to integer type
SriLanka_data['Year'] = SriLanka_data['Year'].astype(int)
SriLanka_data['PDF_ID'] = SriLanka_data['PDF_ID'].astype(int)
SriLanka_data['Cases'] = SriLanka_data['Cases'].fillna(0).astype(int)

# Header metrics
current_year = datetime.now().year
total_cases_current_year = SriLanka_data[SriLanka_data['Year'] == current_year]['Cases'].sum()

# Display the metrics
col1, col2, col3 = st.columns(3)
col1.markdown(f"<div class='metric-box box1'>Country<br><span style='font-size: 24px;'>Sri Lanka</span></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-box box2'>Present Year<br><span style='font-size: 24px;'>{current_year}</span></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-box box3'>Total Number of Cases<br><span style='font-size: 24px;'>{total_cases_current_year}</span></div>", unsafe_allow_html=True)

st.write(" ")
st.write(" ")

# District coordinates (assuming these are accurate)
district_coordinates = {
    'Colombo': [6.9271, 79.9553],
    'Gampaha': [6.9929, 80.2225],
    'Kalutara': [6.5852, 79.9560],
    'Ampara': [7.2975, 81.6820],
    'Anuradhapura': [8.3122, 80.4131],
    'Badulla': [6.9802, 81.0577],
    'Batticaloa': [7.7102, 81.6924],
    'Galle': [6.0535, 80.2209],
    'Hambantota': [6.1246, 81.1011],
    'Jaffna': [9.6684, 80.0074],
    'Kandy': [7.2906, 80.6336],
    'Kurunegala': [7.4839, 80.3683],
    'Matale': [7.4698, 80.6217],
    'Matara': [5.9485, 80.5353],
    'Nuwara Eliya': [6.9708, 80.7829],
    'Puttalam': [8.0362, 79.8283],
    'Trincomalee': [8.5778, 81.2289],
    'Vavuniya': [8.7514, 80.4971],
    'Kilinochchi': [9.4027, 80.3702],
    'Mannar': [8.9832, 79.9737],
    'Monaragala': [6.8425, 81.3075],
    'Polonnaruwa': [7.4820, 81.0000],
    'Ratnapura': [6.6936, 80.3940],
    'Kalmune': [7.4261, 81.7065]
}

# Function to download the GeoJSON file (optional)
def download_geojson():
  import requests
  url = "https://raw.githubusercontent.com/MalakaGu/Sri-lanka-maps/master/discrict_map/District_geo.json"
  response = requests.get(url)
  if response.status_code == 200:
    with open("District_geo.json", "wb") as f:
      f.write(response.content)
    st.success("GeoJSON file downloaded successfully!")
  else:
    st.error(f"Failed to download GeoJSON file. Status code: {response.status_code}")

# Function to calculate annual cases for each region
@st.cache_data
def calculate_annual_cases(df):
    df['Year'] = df['Year'].fillna(0).astype(int)  
    df['Cases'] = df['Cases'].fillna(0).astype(int)
    annual_cases = df.groupby(['Year', 'Region'])['Cases'].sum().reset_index()
    return annual_cases

# Get the annual cases
annual_cases_df = calculate_annual_cases(weekly_df)

# Function to create a color scale based on the number of cases
def get_color(cases, max_cases):
    if cases == 0:
        return 'gray'  # No data
    elif cases < max_cases * 0.25:
        return 'green'  # Low cases
    elif cases < max_cases * 0.5:
        return 'yellow'  # Moderate cases
    elif cases < max_cases * 0.75:
        return 'orange'  # High cases
    else:
        return 'red'  # Very high cases

def create_sri_lanka_map(filtered_data):
    # Create a base map centered on Sri Lanka
    sri_lanka_map = folium.Map(location=[7.8731, 80.7718], zoom_start=7)

    # Get the maximum number of cases for scaling the color
    max_cases = filtered_data['Cases'].max()

    # Add district markers with color based on cases
    for district, coordinates in district_coordinates.items():
        # Get cases for the district in the selected year
        district_data = filtered_data[filtered_data['Region'] == district]
        if not district_data.empty:
            cases = district_data['Cases'].values[0]
            tooltip_text = f"{district}: {cases} cases"
            # Get color based on the number of cases
            color = get_color(cases, max_cases)
        else:
            cases = 0
            tooltip_text = f"{district}: No Data"
            color = 'gray'  # Default color for no data
        
        # Add marker with a location icon and dynamic color
        folium.Marker(
            location=coordinates,
            icon=folium.Icon(color=color, icon_color='white', icon='info-sign', prefix='glyphicon'),
            popup=tooltip_text,
            tooltip=tooltip_text
        ).add_to(sri_lanka_map)
    
    # Load GeoJSON data for district boundaries
    with open('District_geo.json', 'r', encoding='utf-8') as f:
        geojson_data = f.read()
        
    # Create a GeoJSON layer with black and thin borders
    geojson_layer = folium.GeoJson(geojson_data, style_function=lambda x: {'color': 'black', 'weight': 1})
    geojson_layer.add_to(sri_lanka_map)

    return sri_lanka_map

def plot_top_districts(filtered_data):
    # Get the top 10 districts by cases
    top_districts = filtered_data.nlargest(10, 'Cases')

    # Create a DataFrame for displaying
    top_districts_df = pd.DataFrame({
        'District': top_districts['Region'],
        'Cases': top_districts['Cases']
    })

    # Calculate the maximum number of cases for scaling the progress bars
    max_cases = top_districts['Cases'].max()

    # Create a progress bar for each row
    top_districts_df['Progress Bar'] = top_districts_df.apply(
        lambda row: f'<div style="width: 100%; background-color: #e0e0e0; border-radius: 5px; margin: 5px 0;">'
                     f'<div style="width: {(row["Cases"] / max_cases) * 100}%; background-color:  #809fff; height: 15px; border-radius: 5px;"></div>'
                     f'</div>', axis=1)

    # Display the header for the table
    st.markdown("<h6>Top 10 Districts with Leptospirosis Cases</h6>", unsafe_allow_html=True)

    # Create a formatted string for the table
    table_html = "<table style='width: 90%;'><thead><tr><th>District</th><th>Progress Bar</th></tr></thead><tbody>"
    for index, row in top_districts_df.iterrows():
        table_html += f"<tr><td>{row['District']}</td><td>{row['Progress Bar']}</td></tr>"
    table_html += "</tbody></table>"

    # Render the table
    st.markdown(table_html, unsafe_allow_html=True)
        
# Create a time series plot for the selected region
def plot_time_series():
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='none')
    plt.style.use('dark_background')
    ax.plot(region_data['Year'], region_data['Cases'], marker='o')
    ax.set_xlabel('Year')
    ax.set_ylabel('Cases')
    ax.set_title(f'{selected_region} District', color='#99b3ff')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(region_data['Year'])
    plt.grid(True, color='gray')
    plt.gca().set_facecolor('black')
    st.pyplot(fig, transparent=True)

# Function to plot yearly cases
def plot_yearly_cases():
    SriLanka_cases = SriLanka_data.groupby('Year')['Cases'].sum().reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.style.use('dark_background')  # Set dark theme
    plt.plot(SriLanka_cases['Year'], SriLanka_cases['Cases'], marker='o', linestyle='-')
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.title('From 2007 - Present', color='#99b3ff')
    plt.xlabel('Year', color='white')
    plt.ylabel('Cases', color='white')
    plt.xticks(SriLanka_cases['Year'])
    plt.grid(True, color='gray')
    plt.gca().set_facecolor('black')  # Set background color to black
    st.pyplot(plt, transparent=True)
    

# Function to plot weekly cases
def plot_weekly_cases():
    SriLanka_data['Cases'] = SriLanka_data['Cases'].fillna(0)
    
    plt.figure(figsize=(12, 6))
    plt.style.use('dark_background')  # Set dark theme
    plt.plot(SriLanka_data['PDF_ID'], SriLanka_data['Cases'], marker='', linestyle='-', label='Actual Cases')
    
    plt.title('From 2007 - Present', color='#99b3ff')
    plt.xlabel('Week', color='white')
    plt.ylabel('No. of Leptospirosis Cases', color='white')
    plt.grid(True, color='gray')
    plt.gca().set_facecolor('black')  # Set background color to black
    st.pyplot(plt, transparent=True)

# Function to forecast using RNN for weekly cases
def forecast_rnn():
    # Set random seed for reproducibility
    random_seed = 1234
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    # Prepare the data
    data = SriLanka_data['Cases'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences for training
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 10
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split the data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build the RNN model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Evaluate model performance
    y_test_reshaped = y_test.reshape(-1, 1)
    y_test_inverse = scaler.inverse_transform(y_test_reshaped)

    mae = mean_absolute_error(y_test_inverse, predictions)
    rmse = mean_squared_error(y_test_inverse, predictions, squared=False)

    # Calculate accuracy percentage
    accuracy_percentage = (1 - (mae / np.mean(y_test_inverse))) * 100

    # Forecast for the next 12 weeks
    last_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)
    future_forecast = []
    for _ in range(12):
        forecast = model.predict(last_sequence)
        future_forecast.append(forecast[0, 0])
        last_sequence = np.append(last_sequence[:, 1:, :], forecast.reshape(1, 1, 1), axis=1)

    future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))
    future_forecast = np.round(future_forecast).astype(int)

    return predictions, future_forecast, mae, rmse, accuracy_percentage

# Function to forecast annual cases using RNN
def forecast_annual_rnn():
    # Set random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    # Prepare the data
    annual_data = SriLanka_data.groupby('Year')['Cases'].sum().values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(annual_data)

    # Create sequences for training
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 5  # Change this for annual data
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split the data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build the RNN model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Evaluate model performance
    y_test_reshaped = y_test.reshape(-1, 1)
    y_test_inverse = scaler.inverse_transform(y_test_reshaped)

    mae = mean_absolute_error(y_test_inverse, predictions)
    rmse = mean_squared_error(y_test_inverse, predictions, squared=False)

    # Calculate accuracy percentage
    accuracy_percentage = (1 - (mae / np.mean(y_test_inverse))) * 100

    # Forecast for the next 5 years
    last_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)
    future_forecast = []
    for _ in range(5):
        forecast = model.predict(last_sequence)
        future_forecast.append(forecast[0, 0])
        last_sequence = np.append(last_sequence[:, 1:, :], forecast.reshape(1, 1, 1), axis=1)

    future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))
    future_forecast = np.round(future_forecast).astype(int)

    return predictions, future_forecast, mae, rmse, accuracy_percentage

# Prepare annual data for each district
def prepare_annual_district_data(df):
    # Group by Year and Region, sum the Cases
    annual_district_data = df.groupby(['Year', 'Region'])['Cases'].sum().reset_index()
    return annual_district_data

def create_donut_chart(data, year_label, selected_region):
    # Filter data by the selected region
    region_data = data[data['Region'] == selected_region]
    
    # Calculate total cases for the region and cases for the specified year
    total_cases = region_data['Cases'].sum()
    year_cases = region_data[region_data['Year'] == year_label]['Cases'].sum()
    
    # Prepare data for the Altair chart
    response_data = pd.DataFrame({
        'Attrition': ['Year Cases', 'Other Cases'],
        'Count': [year_cases, total_cases - year_cases]
    })

    # Donut chart with transparent area
    donut_chart = alt.Chart(response_data).mark_arc(innerRadius=75, stroke='#ffcc66', strokeWidth=2).encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color('Attrition:N', scale=alt.Scale(domain=['Year Cases', 'Other Cases'], range=['#ffcc66', 'rgba(0, 0, 0, 0)']),
                        legend=None),
        tooltip=["Attrition", "Count"]
    ).properties(width=300, height=300)

    # Center text overlay showing the year label and percentage
    center_text = alt.Chart(pd.DataFrame({
        'text1': [year_label],
        'text2': [f"{(year_cases / total_cases * 100) if total_cases > 0 else 0:.1f}%"]
    })).mark_text(
        align='center',
        baseline='middle',
        size=25,
        dy=-10,
        color='white'
    ).encode(
        text='text1:N'
    )

    # Percentage text overlay in the center
    percentage_text = alt.Chart(pd.DataFrame({
        'text': [f"{(year_cases / total_cases * 100) if total_cases > 0 else 0:.1f}%"]
    })).mark_text(
        align='center',
        baseline='middle',
        size=32,
        dy=20,
        color='white',
        fontWeight='bold'
    ).encode(
        text='text:N'
    )

    # Combine the donut chart and center text
    return (donut_chart + center_text + percentage_text).configure_view(stroke=None)

    
# Create columns with different widths
col1, col2, col3 = st.columns([1, 2, 2]) 

with col1:
    st.markdown("<div class='donut-container'>", unsafe_allow_html=True)
    st.write(" ")
    st.write(" ")
    selected_year = st.selectbox("**Select a Year**", sorted(annual_cases_df['Year'].unique()))
    filtered_data = annual_cases_df[annual_cases_df['Year'] == selected_year]
    selected_region = st.selectbox("**Select a District**", sorted(annual_cases_df['Region'].unique()))
    region_data = annual_cases_df[annual_cases_df['Region'] == selected_region]
    st.markdown("</div>", unsafe_allow_html=True)
    
with col2:
    st.markdown(f"<div class='donut-container'><h5 style='text-align: center;'>Leptospirosis Cases Distribution in Year {selected_year}</h5>", unsafe_allow_html=True)
    sri_lanka_map = create_sri_lanka_map(filtered_data)
    folium_static(sri_lanka_map,width=500,height=500)
    st.markdown("</div>", unsafe_allow_html=True)
    
with col3:
    st.markdown("<div class='donut-container'>", unsafe_allow_html=True)
    st.write(" ")
    st.write(" ")
    plot_top_districts(annual_cases_df[annual_cases_df['Year'] == selected_year])
    st.markdown("</div>", unsafe_allow_html=True)

# Additional Row for New Graphs
col5, col6, col7 = st.columns([2,1,1])  
    
with col5:
    st.markdown("<div class='chart-container'><h5 style='text-align: center;'>Annual District-wise Leptospirosis Cases</h5>", unsafe_allow_html=True)
    plot_time_series()
    st.markdown("</div>", unsafe_allow_html=True)
    
with col6:
    highest_year = region_data.loc[region_data['Cases'].idxmax(), 'Year']
    st.write(" ")
    st.write(" ")
    st.markdown(f"<div class='chart-container'><h6 style='text-align: center;'>Percentage of Highest Cases in {highest_year} </h6>", unsafe_allow_html=True)
    donut_chart_highest_year = create_donut_chart(annual_cases_df, highest_year, selected_region)
    st.altair_chart(donut_chart_highest_year)
    st.markdown("</div>", unsafe_allow_html=True)
    
with col7:
    current_year = annual_cases_df['Year'].max() 
    st.write(" ")
    st.write(" ")
    st.markdown(f"<div class='chart-container'><h6 style='text-align: center;'>Percentage of Highest Cases in {current_year}</h6>", unsafe_allow_html=True)
    donut_chart_current_year = create_donut_chart(annual_cases_df, current_year, selected_region)
    st.altair_chart(donut_chart_current_year)
    st.markdown("</div>", unsafe_allow_html=True)
    
# Additional Row for New Graphs
col8, col9 = st.columns(2)  
    
with col8:
    st.markdown("<div class='chart-container'><h5 style='text-align: center;'>Annual Leptospirosis Cases</h5>", unsafe_allow_html=True)
    plot_yearly_cases()
    if st.button("Show Annual Forecast"):
        predictions, future_forecast, mae, rmse, accuracy_percentage = forecast_annual_rnn()
    
        # Plotting the annual forecast
        annual_cases = SriLanka_data.groupby('Year')['Cases'].sum().reset_index()
    
        plt.figure(figsize=(12, 6))
        plt.style.use('dark_background')  # Set dark theme
        plt.plot(annual_cases['Year'], annual_cases['Cases'], marker='o', linestyle='-', label='Actual Annual Cases')
    
        forecast_years = np.arange(annual_cases['Year'].iloc[-1] + 1, annual_cases['Year'].iloc[-1] + 6)
        plt.plot(forecast_years, future_forecast, marker='o', linestyle='-', label='Forecasted Annual Cases', color='orange')
    
        # Plot the predicted values
        predicted_years = annual_cases['Year'].iloc[-len(predictions):].values
        plt.plot(predicted_years, predictions, marker='o', linestyle='-', label='Predicted Annual Cases', color='yellow')
    
        plt.title('Annual Leptospirosis Cases with Forecast', color='#99b3ff')
        plt.xlabel('Year', color='white')
        plt.ylabel('No. of Leptospirosis Cases', color='white')
        plt.grid(True, color='gray')
    
        # Assuming 'annual_cases' is a DataFrame with a 'Year' column
        last_year = annual_cases['Year'].iloc[-1]
        forecast_years = np.arange(last_year + 1, last_year + 6)  # 5 years of forecast
        all_years = np.concatenate([annual_cases['Year'], forecast_years])
        plt.xticks(all_years)
        #plt.xticks(annual_cases['Year']+5)
    
        plt.legend()
        plt.gca().set_facecolor('black')  # Set background color to black
        st.pyplot(plt, transparent=True)
    
        # Display accuracy metrics
        st.subheader("LSTM Model Accuracy Metrics for Annual Cases")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        st.write(f"Test Accuracy Percentage: {accuracy_percentage:.2f}%")

        # Display forecast values
        st.subheader("Forecast for the Next 5 Years")
        forecast_years_df = pd.DataFrame({
           'Year': forecast_years,
           'Forecasted Cases': future_forecast.flatten()
        })
        st.table(forecast_years_df)
    st.markdown("</div>", unsafe_allow_html=True)
    
with col9:
    st.markdown("<div class='chart-container'><h5 style='text-align: center;'>Weekly Leptospirosis Cases</h5>", unsafe_allow_html=True)
    plot_weekly_cases()

    # Button to show weekly forecast
    if st.button("Show Weekly Forecast"):
        predictions, future_forecast, mae, rmse, accuracy_percentage = forecast_rnn()
    
        # Plotting the weekly forecast
        plt.figure(figsize=(12, 6))
        plt.style.use('dark_background')  # Set dark theme
        plt.plot(SriLanka_data['PDF_ID'], SriLanka_data['Cases'], marker='', linestyle='-', label='Actual Cases')
    
        forecast_weeks = np.arange(len(SriLanka_data) + 1, len(SriLanka_data) + 13)
        plt.plot(forecast_weeks, future_forecast, marker='', linestyle='-', label='Forecasted Cases', color='orange')
    
        # Plot the predicted values
        predicted_weeks = np.arange(len(SriLanka_data) - len(predictions), len(SriLanka_data))
        plt.plot(predicted_weeks, predictions, marker='', linestyle='-', label='Predicted Cases', color='yellow')
    
        plt.title('Weekly Leptospirosis Cases with Forecast', color='#99b3ff')
        plt.xlabel('Week', color='white')
        plt.ylabel('No. of Leptospirosis Cases', color='white')
        plt.grid(True, color='gray')
        plt.legend()
        plt.gca().set_facecolor('black')  # Set background color to black
        st.pyplot(plt, transparent=True)
    
        # Display accuracy metrics
        st.subheader("LSTM Model Accuracy Metrics for Weekly Cases")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        st.write(f"Test Accuracy Percentage: {accuracy_percentage:.2f}%")

        # Display forecast values
        st.subheader("Forecast for the Next 12 Weeks")
        forecast_weeks_df = pd.DataFrame({
            'Week': np.arange(len(SriLanka_data) + 1, len(SriLanka_data) + 13),
            'Forecasted Cases': future_forecast.flatten()
        })
        st.table(forecast_weeks_df)
    st.markdown("</div>", unsafe_allow_html=True)

# Custom title with specific class
st.markdown('<p class="custom-title">Thank you for visiting our dashboard !!!</p>', unsafe_allow_html=True)

# Custom main content with specific class
st.markdown('<p class="custom-content">We appreciate your time and interest.</p>', unsafe_allow_html=True)
