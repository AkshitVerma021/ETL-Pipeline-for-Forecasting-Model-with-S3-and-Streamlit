import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# File uploader for dataset
st.sidebar.title("Upload Car Data CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    car = pd.read_csv(uploaded_file)
else:
    car = pd.read_csv('Cleaned_Car_data.csv')

# Attempt to find key columns if missing
def find_column(possible_names, df):
    for col in possible_names:
        if col in df.columns:
            return col
    return None

company_column = find_column(['company', 'brand', 'maker', 'manufacturer'], car)
year_column = find_column(['year', 'model_year', 'production_year'], car)
fuel_type_column = find_column(['fuel_type', 'fuel', 'engine_type'], car)

if company_column:
    car.rename(columns={company_column: 'company'}, inplace=True)
else:
    st.error("No company-related column found in the dataset.")

if year_column:
    car.rename(columns={year_column: 'year'}, inplace=True)
else:
    st.error("No year-related column found in the dataset.")

if fuel_type_column:
    car.rename(columns={fuel_type_column: 'fuel_type'}, inplace=True)
else:
    st.error("No fuel type column found in the dataset.")

# Set Seaborn style for a professional look
sns.set_style("whitegrid")

# Sidebar for input selection
st.title("Car Price Prediction App")
st.write("Fill in the details below to predict the price of a car.")

companies = sorted(car['company'].unique())
car_models = sorted(car['name'].unique())
years = sorted(car['year'].unique(), reverse=True)
fuel_types = car['fuel_type'].unique()

# User inputs
company = st.selectbox("Select Company", ["Select Company"] + companies)
car_model = st.selectbox("Select Car Model", car_models)
year = st.selectbox("Select Year", years)
fuel_type = st.selectbox("Select Fuel Type", fuel_types)
driven = st.number_input("Enter Kilometers Driven", min_value=0, step=1000)

if st.button("Predict Price"):
    if company == "Select Company":
        st.error("Please select a valid company.")
    else:
        # Make prediction
        prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                                data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))
        
        st.success(f"Estimated Price: ₹{np.round(prediction[0], 2)}")

# Car Sales Analysis
st.title("Car Sales Analysis")
selected_year = st.selectbox("Select Year for Analysis", years)

sales_data = car[car['year'] == selected_year].groupby('company').size().reset_index(name='count')
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='company', y='count', data=sales_data, ax=ax, palette='coolwarm')
ax.set_xlabel("Company", fontsize=12)
ax.set_ylabel("Number of Cars Sold", fontsize=12)
ax.set_title(f"Car Sales in {selected_year} by Company", fontsize=14)
st.pyplot(fig)

# Line Chart Section with Prediction
st.title("Car Data Visualization")
st.write("Select a company and columns to visualize their relationship.")

selected_company = st.selectbox("Select Company for Visualization", ["All"] + companies)
filtered_car = car if selected_company == "All" else car[car['company'] == selected_company]

x_axis_option = st.selectbox("Select Column for X-Axis", filtered_car.columns[2:])
y_axis_option = st.selectbox("Select Column for Y-Axis", filtered_car.columns[2:])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(filtered_car[x_axis_option], filtered_car[y_axis_option], marker='o', linestyle='-', markersize=5, color='b', alpha=0.7)
ax.set_xlabel(x_axis_option, fontsize=12)
ax.set_ylabel(y_axis_option, fontsize=12)
ax.set_title(f"{y_axis_option} vs {x_axis_option} for {selected_company}", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig)

# Predict Car Price Over Time
st.title("Car Price Prediction Over Time")
#st.write("See how the predicted car price changes over the years.")

if company != "Select Company" and car_model:
    price_predictions = []
    for yr in years:
        pred = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                          data=np.array([car_model, company, yr, driven, fuel_type]).reshape(1, 5)))
        price_predictions.append(np.round(pred[0], 2))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years, price_predictions, marker='o', linestyle='-', color='g', alpha=0.7)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Predicted Price (₹)", fontsize=12)
    ax.set_title(f"Predicted Price Over Time for {car_model}", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
else:
    st.warning("Please select a valid company and car model to see predictions.")
