import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import folium
from streamlit_folium import st_folium
import altair as alt
import pydeck as pdk

model = joblib.load("models/best_model.pkl")
encoders = joblib.load("models/encoders.pkl")

st.set_page_config(layout="wide")
st.title("Flight Price Prediction 🛫")

city_coords = {
    "Delhi": [28.7041, 77.1025],
    "New Delhi": [28.6139, 77.2090],
    "Mumbai": [19.0760, 72.8777],
    "Cochin": [9.9312, 76.2673],
    "Kolkata": [22.5726, 88.3639],
    "Banglore": [12.9716, 77.5946],
    "Hyderabad": [17.3850, 78.4867],
}

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Flight Inputs")

    airlines = st.multiselect(
        "Airline(s)",
        encoders["Airline"].classes_,
        default=[encoders["Airline"].classes_[0]]
    )
    source = st.selectbox("Source", encoders["Source"].classes_)
    destination = st.selectbox("Destination", encoders["Destination"].classes_)
    journey_date = st.date_input("Date of Journey", datetime.today())

    additional_info = st.selectbox("Additional Info", ["No check-in baggage included", "Red-eye flight", "No Info"])
    
    st.header("Calendar Predictions")
    start_date = st.date_input("Start Date", journey_date)
    end_date = st.date_input("End Date", journey_date)
    
    predict_click = st.button("Predict Price")

COLUMN_ORDER = [
    "Airline",
    "Source",
    "Destination",
    "Additional_Info",
    "Date",
    "Month",
    "Year", 
    "Duration_hour",
    "Duration_min",
]

def preprocess_inputs(use_date, use_month, use_year, airline_val):
    dur_hour = 0
    dur_min = 0


    input_data = {
        "Airline": encoders["Airline"].transform([airline_val])[0],
        "Source": encoders["Source"].transform([source])[0],
        "Destination": encoders["Destination"].transform([destination])[0],
        "Additional_Info": encoders["Additional_Info"].transform([additional_info])[0],
        "Date": use_date,
        "Month": use_month,
        "Year": use_year,
        "Duration_hour": dur_hour,
        "Duration_min": dur_min,
    }
    df_input = pd.DataFrame([input_data])
    df_input = df_input[COLUMN_ORDER]
    return df_input

# --- Main Panel ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Prediction & Calendar")
    
    if predict_click:
        results = []
        for airline_val in airlines:
            single_input = preprocess_inputs(journey_date.day, journey_date.month, journey_date.year, airline_val)
            prediction = model.predict(single_input)[0]
            results.append({"Airline": airline_val, "Predicted Price": f"₹{prediction:.2f}"})
        st.write(pd.DataFrame(results))
    
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
        arc_data = pd.DataFrame({
            "start_lon": [src_coords[1]],
            "start_lat": [src_coords[0]],
            "end_lon": [dst_coords[1]],
            "end_lat": [dst_coords[0]]
        })
        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v10",
            initial_view_state=pdk.ViewState(
                latitude=mid_lat,
                longitude=mid_lon,
                zoom=5,
                pitch=45
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