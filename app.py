import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import folium
from streamlit_folium import st_folium
import altair as alt

# Load model and encoders
model = joblib.load("models/best_model.pkl")
encoders = joblib.load("models/encoders.pkl")

st.set_page_config(layout="wide")
st.title("Flight Price Prediction 🛫")

# --- Define a mapping for source and destination cities to coordinates ---
city_coords = {
    "Delhi": [28.7041, 77.1025],
    "New Delhi": [28.6139, 77.2090],
    "Mumbai": [19.0760, 72.8777],
    "Cochin": [9.9312, 76.2673],
    "Kolkata": [22.5726, 88.3639],
    "Banglore": [12.9716, 77.5946],
    "Hyderabad": [17.3850, 78.4867],
    # Add more cities as needed
}

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Flight Inputs")
    # Multiple airline selection using multiselect
    airlines = st.multiselect("Airline(s)", encoders["Airline"].classes_, default=[encoders["Airline"].classes_[0]])
    source = st.selectbox("Source", encoders["Source"].classes_)
    destination = st.selectbox("Destination", encoders["Destination"].classes_)
    journey_date = st.date_input("Date of Journey", datetime.today())

    dep_time = st.time_input("Departure Time", datetime.now().time())
    arrival_time = st.time_input("Arrival Time", datetime.now().time())
    # Hide the duration input (commented out)
    # duration = st.text_input("Duration (e.g., 2h 30m)", "2h 30m")
    # Instead, set default duration values
    duration = ""  # duration not used; defaults below will be used
    stops = st.selectbox("Total Stops", [0, 1, 2, 3, 4])
    additional_info = st.selectbox("Additional Info", encoders["Additional_Info"].classes_)
    
    st.header("Calendar Predictions")
    start_date = st.date_input("Start Date", journey_date)
    end_date = st.date_input("End Date", journey_date)

# --- Main Panel ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Prediction")
    # Must match training column order
    COLUMN_ORDER = [
        "Airline",
        "Source",
        "Destination",
        "Additional_Info",
        "Date",
        "Month",
        "Total_Stops",
        "Dep_hour",
        "Dep_min",
        "Arrival_hour",
        "Arrival_min",
        "Duration_hour",
        "Duration_min",
    ]

    # Updated to accept an airline value as parameter.
    def preprocess_inputs(use_date, use_month, airline_val):
        # Extract hours & minutes for departure/arrival
        dep_hour = dep_time.hour
        dep_min = dep_time.minute
        arrival_hour = arrival_time.hour
        arrival_min = arrival_time.minute

        # Duration input hidden; default to 0 hours and 0 minutes
        dur_hour = 0
        dur_min = 0
        # (If you wish to enable duration input later, uncomment & modify this code)
        # if duration and 'h' in duration:
        #     dur_hour = int(duration.split('h')[0].strip())
        #     if 'm' in duration:
        #         dur_min_text = duration.split('h')[1].split('m')[0].strip()
        #         dur_min = int(dur_min_text) if dur_min_text else 0
        # elif duration and 'm' in duration:
        #     dur_min = int(duration.replace('m', '').strip())

        # Encode categorical features
        input_data = {
            "Airline": encoders["Airline"].transform([airline_val])[0],
            "Source": encoders["Source"].transform([source])[0],
            "Destination": encoders["Destination"].transform([destination])[0],
            "Additional_Info": encoders["Additional_Info"].transform([additional_info])[0],
            "Date": use_date,
            "Month": use_month,
            "Total_Stops": stops,
            "Dep_hour": dep_hour,
            "Dep_min": dep_min,
            "Arrival_hour": arrival_hour,
            "Arrival_min": arrival_min,
            "Duration_hour": dur_hour,
            "Duration_min": dur_min,
        }
        # Create DataFrame ordered as in training
        df = pd.DataFrame([input_data])
        df = df[COLUMN_ORDER]
        return df

    # Single prediction - if multiple airlines selected, loop over them.
    if st.button("Predict Price"):
        results = []
        for airline_val in airlines:
            single_input = preprocess_inputs(journey_date.day, journey_date.month, airline_val)
            prediction = model.predict(single_input)[0]
            results.append({"Airline": airline_val, "Predicted Price": f"₹{prediction:.2f}"})
        st.write(pd.DataFrame(results))

    # Calendar predictions: produce multi-line dataframe if multiple airlines selected.
    if start_date <= end_date:
        all_data = []
        dates = pd.date_range(start_date, end_date)
        for airline_val in airlines:
            prices = []
            for d in dates:
                input_df = preprocess_inputs(d.day, d.month, airline_val)
                prices.append(model.predict(input_df)[0])
            temp_df = pd.DataFrame({"Date": dates, "Price": prices})
            temp_df["Airline"] = airline_val
            all_data.append(temp_df)
        if all_data:
            calendar_data = pd.concat(all_data)
            # Using Altair for multi-line chart colored by airline selection
            chart = alt.Chart(calendar_data).mark_line(point=True).encode(
                x="Date:T",
                y="Price:Q",
                color="Airline:N",
                tooltip=["Date", "Price", "Airline"]
            ).properties(width=600, height=400)
            st.altair_chart(chart, use_container_width=True)
            st.write(calendar_data.sort_values(["Airline", "Date"]))
    else:
        st.error("End date must fall after start date")

with col2:
    st.subheader("Flight Route Map")
    # Check if coordinates available
    if source in city_coords and destination in city_coords:
        src_coords = city_coords[source]
        dst_coords = city_coords[destination]
        # Calculate mid-point for centering the map
        mid_lat = (src_coords[0] + dst_coords[0]) / 2
        mid_lon = (src_coords[1] + dst_coords[1]) / 2
        m = folium.Map(location=[mid_lat, mid_lon], zoom_start=5)
        # Add markers and a route line
        folium.Marker(location=src_coords, tooltip=source, icon=folium.Icon(color="blue")).add_to(m)
        folium.Marker(location=dst_coords, tooltip=destination, icon=folium.Icon(color="red")).add_to(m)
        folium.PolyLine(locations=[src_coords, dst_coords], color="green", weight=5).add_to(m)
        st_folium(m, width=700, height=450)
    else:
        st.warning("No coordinate mapping is available for the selected cities.")