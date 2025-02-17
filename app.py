import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import folium
from streamlit_folium import st_folium
import altair as alt
import pydeck as pdk

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

    # CHANGED: Removed departure time input since model was trained without it.
    # dep_time = st.time_input("Departure Time", datetime.now().time())
    arrival_time = st.time_input("Arrival Time", datetime.now().time())
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
    # CHANGED: Removed "Dep_hour"/"Dep_min" from COLUMN_ORDER; kept Year.
    COLUMN_ORDER = [
        "Airline",
        "Source",
        "Destination",
        "Additional_Info",
        "Date",
        "Month",
        "Year",  # Year is kept
        "Total_Stops",
        "Arrival_hour",
        "Arrival_min",
        "Duration_hour",
        "Duration_min",
    ]

    # Updated function to include Year and remove departure time features.
    def preprocess_inputs(use_date, use_month, use_year, airline_val):
        # CHANGED: Removed departure time features extraction.
        arrival_hour = arrival_time.hour
        arrival_min = arrival_time.minute

        # Duration: defaults to 0 hours and 0 minutes.
        dur_hour = 0
        dur_min = 0

        # Encode categorical features
        input_data = {
            "Airline": encoders["Airline"].transform([airline_val])[0],
            "Source": encoders["Source"].transform([source])[0],
            "Destination": encoders["Destination"].transform([destination])[0],
            "Additional_Info": encoders["Additional_Info"].transform([additional_info])[0],
            "Date": use_date,
            "Month": use_month,
            "Year": use_year,  # Included Year feature
            "Total_Stops": stops,
            "Arrival_hour": arrival_hour,
            "Arrival_min": arrival_min,
            "Duration_hour": dur_hour,
            "Duration_min": dur_min,
        }
        # Create DataFrame ordered as in training
        df = pd.DataFrame([input_data])
        df = df[COLUMN_ORDER]
        return df

    # Single prediction - loop over airlines if multiple selected.
    if st.button("Predict Price"):
        results = []
        for airline_val in airlines:
            single_input = preprocess_inputs(journey_date.day, journey_date.month, journey_date.year, airline_val)
            prediction = model.predict(single_input)[0]
            results.append({"Airline": airline_val, "Predicted Price": f"₹{prediction:.2f}"})
        st.write(pd.DataFrame(results))

    # Calendar predictions: create multi-line dataframe if multiple airlines selected.
    if start_date <= end_date:
        all_data = []
        dates = pd.date_range(start_date, end_date)
        for airline_val in airlines:
            prices = []
            for d in dates:
                input_df = preprocess_inputs(d.day, d.month, d.year, airline_val)
                prices.append(model.predict(input_df)[0])
            temp_df = pd.DataFrame({"Date": dates, "Price": prices})
            temp_df["Airline"] = airline_val
            all_data.append(temp_df)
        if all_data:
            calendar_data = pd.concat(all_data)
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
    st.subheader("Flight Route Map (3D)")
    if source in city_coords and destination in city_coords:
        src_coords = city_coords[source]
        dst_coords = city_coords[destination]
        mid_lat = (src_coords[0] + dst_coords[0]) / 2
        mid_lon = (src_coords[1] + dst_coords[1]) / 2
        # Prepare data for the ArcLayer
        arc_data = pd.DataFrame({
            "start_lon": [src_coords[1]],
            "start_lat": [src_coords[0]],
            "end_lon": [dst_coords[1]],
            "end_lat": [dst_coords[0]]
        })
        # CHANGED: Using PyDeck for a 3D curved flight path on a light map style.
        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v10",  # Light style for areas in white.
            initial_view_state=pdk.ViewState(
                latitude=mid_lat,
                longitude=mid_lon,
                zoom=5,
                pitch=45  # Adjust the pitch for a 3D effect.
            ),
            layers=[
                pdk.Layer(
                    "ArcLayer",
                    data=arc_data,
                    get_source_position="[start_lon, start_lat]",
                    get_target_position="[end_lon, end_lat]",
                    get_source_color=[0, 128, 200],
                    get_target_color=[200, 0, 80],
                    auto_highlight=True,
                    width_scale=0.0001,
                    width_min_pixels=2,
                    width_max_pixels=10,
                )
            ],
            tooltip={"text": f"Flight from {source} to {destination}"},
        )
        st.pydeck_chart(deck)
    else:
        st.warning("No coordinate mapping is available for the selected cities.")