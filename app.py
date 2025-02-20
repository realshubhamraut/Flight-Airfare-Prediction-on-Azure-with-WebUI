import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import folium
from streamlit_folium import st_folium
import altair as alt
import pydeck as pdk

st.set_page_config(
    page_title="Flight Price Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

custom_css = """
<style>
    body {
        background-color: #f9f9f9;
    }
    .main {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf, #2e7bcf);
        color: #ffffff;
    }
    .stButton>button {
        background-color: #2e7bcf;
        color: white;
        border-radius: 5px;
    }
    .header-title {
        text-align: center;
        margin-bottom: 0;
    }
    .header-tagline {
        text-align: left; 
        color: #555555;
        font-size: 18px;
        margin-top: 5px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.title("Flight Price Prediction 🛫", anchor="header-title")
st.markdown("<p class='header-tagline'>Your Trusted Companion for Flight Fare Forecasting</p>", unsafe_allow_html=True)

# Load model and encoders
model = joblib.load("models/best_ml_model.pkl")
encoders = joblib.load("models/saved_encoders.pkl")

city_coords = {
    "Delhi": [28.7041, 77.1025],
    "New Delhi": [28.6139, 77.2090],
    "Mumbai": [19.0760, 72.8777],
    "Cochin": [9.9312, 76.2673],
    "Kolkata": [22.5726, 88.3639],
    "Banglore": [12.9716, 77.5946],
    "Hyderabad": [17.3850, 78.4867],
}

# --- Sidebar for Inputs (only key fields) ---
with st.sidebar:
    st.header("Flight Inputs")
    airlines = st.multiselect(
        "Select Airline(s)",
        encoders["Airline"].classes_,
        default=[encoders["Airline"].classes_[0]]
    )
    source = st.selectbox("Source", encoders["Source"].classes_)
    destination = st.selectbox("Destination", encoders["Destination"].classes_)
    journey_date = st.date_input("Date of Journey", datetime.today())
    additional_info = st.selectbox(
        "Additional Info",
        ["No check-in baggage included", "Red-eye flight", "No Info"]
    )
    
    # Hidden default values for features not needed in UI
    DEFAULT_TOTAL_STOPS = 0
    DEFAULT_ARRIVAL_TIME = datetime.strptime("13:00", "%H:%M").time()
    DEFAULT_DURATION = "3h 0m"
    
    st.markdown("---")
    st.header("Calendar Predictions")
    start_date = st.date_input("Start Date", journey_date)
    end_date = st.date_input("End Date", journey_date)
    
    predict_click = st.button("Predict Price")

# defined column order matching features used during training.
COLUMN_ORDER = [
    "Airline",
    "Source",
    "Destination",
    "Additional_Info",
    "Date",
    "Month",
    "Total_Stops",
    "Arrival_hour",
    "Arrival_min",
    "Duration_hour",
    "Duration_min",
]

def parse_duration(duration_str):
    duration_str = duration_str.lower().replace(" ", "")
    dur_hour = 0
    dur_min = 0
    if "h" in duration_str:
        parts = duration_str.split("h")
        dur_hour = int(parts[0])
        if len(parts) > 1 and "m" in parts[1]:
            dur_min = int(parts[1].replace("m", ""))
    elif "m" in duration_str:
        dur_min = int(duration_str.replace("m", ""))
    return dur_hour, dur_min

def preprocess_inputs(date_day, month, airline_val, total_stops, arrival_time, duration_str):
    dur_hour, dur_min = parse_duration(duration_str)
    input_data = {
        "Airline": encoders["Airline"].transform([airline_val])[0],
        "Source": encoders["Source"].transform([source])[0],
        "Destination": encoders["Destination"].transform([destination])[0],
        "Additional_Info": encoders["Additional_Info"].transform([additional_info])[0],
        "Date": date_day,
        "Month": month,
        "Total_Stops": total_stops,
        "Arrival_hour": arrival_time.hour,
        "Arrival_min": arrival_time.minute,
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
            single_input = preprocess_inputs(
                journey_date.day,
                journey_date.month,
                airline_val,
                DEFAULT_TOTAL_STOPS,
                DEFAULT_ARRIVAL_TIME,
                DEFAULT_DURATION
            )
            prediction = model.predict(single_input)[0]
            results.append({
                "Airline": airline_val,
                "Predicted Price": f"₹{prediction:.2f}"
            })
        st.write(pd.DataFrame(results))
    
    if start_date <= end_date:
        all_data = []
        dates = pd.date_range(start_date, end_date)
        for airline_val in airlines:
            prices = []
            for d in dates:
                input_df = preprocess_inputs(
                    d.day, d.month, airline_val,
                    DEFAULT_TOTAL_STOPS,
                    DEFAULT_ARRIVAL_TIME,
                    DEFAULT_DURATION
                )
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