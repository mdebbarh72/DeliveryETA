import streamlit as st
import pandas as pd
import joblib
import numpy as np

def main():
    st.set_page_config(page_title="Delivery ETA Predictor", layout="wide")
    st.title("Delivery ETA Predictor")
    
    try:
        model = joblib.load('models/delivery_model.pkl')
        df = pd.read_csv('data/bronze/FoodDeliveryClean.csv')
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return

    st.sidebar.header("Input Features")
    
    distance = st.sidebar.slider("Distance (km)", 0.0, 20.0, 5.0)
    weather = st.sidebar.selectbox("Weather", df['Weather'].unique())
    traffic = st.sidebar.selectbox("Traffic Level", df['Traffic_Level'].unique())
    time_of_day = st.sidebar.selectbox("Time of Day", df['Time_of_Day'].unique())
    vehicle = st.sidebar.selectbox("Vehicle Type", df['Vehicle_Type'].unique())
    prep_time = st.sidebar.slider("Preparation Time (min)", 0, 60, 15)
    exp = st.sidebar.slider("Courier Experience (yrs)", 0.0, 10.0, 2.0)

    if st.sidebar.button("Predict ETA"):
        input_data = pd.DataFrame({
            'Distance_km': [distance],
            'Weather': [weather],
            'Traffic_Level': [traffic],
            'Time_of_Day': [time_of_day],
            'Vehicle_Type': [vehicle],
            'Preparation_Time_min': [prep_time],
            'Courier_Experience_yrs': [exp]
        })
        
        pred = model.predict(input_data)[0]
        st.success(f"Estimated Delivery Time: {pred:.0f} minutes")

    st.header("Visualizations")
    st.subheader("Data Overview")
    st.dataframe(df.head())
    
    st.subheader("Model Metrics")
    st.write("Model has been trained with XGBoost and optimized with RandomizedSearchCV.")
    st.write("Le modèle se trompe en moyenne de ~4 minutes (MAE) et a un R² > 0.81.")

if __name__ == "__main__":
    main()
