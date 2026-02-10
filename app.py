import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import time
import folium
from streamlit_folium import st_folium
import altair as alt
import pydeck as pdk

# Production Monitoring
try:
    from monitoring.data_collector import (
        ModelDataCollector, 
        DataDriftDetector,
        get_metrics
    )
    MONITORING_ENABLED = True
except ImportError:
    MONITORING_ENABLED = False

st.set_page_config(
    page_title="Flight Price Prediction",
    page_icon=None,
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
st.title("Flight Price Prediction", anchor="header-title")
st.markdown("<p class='header-tagline'>Your Trusted Companion for Flight Fare Forecasting</p>", unsafe_allow_html=True)

# Load model and encoders
model = joblib.load("models/best_ml_model.pkl")
encoders = joblib.load("models/saved_encoders.pkl")

# Initialize Production Monitoring
if MONITORING_ENABLED:
    # Initialize collectors with session state for persistence
    if 'input_collector' not in st.session_state:
        st.session_state.input_collector = ModelDataCollector("flight_price", identifier="inputs")
        st.session_state.output_collector = ModelDataCollector("flight_price", identifier="outputs")
        st.session_state.drift_detector = DataDriftDetector()
        
        # Set baseline stats from training data
        st.session_state.drift_detector.set_baseline("Duration_hour", mean=8.5, std=5.2, min_val=0, max_val=48)
        st.session_state.drift_detector.set_baseline("Total_Stops", mean=1.0, std=0.8, min_val=0, max_val=4)
        st.session_state.drift_detector.set_baseline("Arrival_hour", mean=12, std=6, min_val=0, max_val=23)
    
    input_collector = st.session_state.input_collector
    output_collector = st.session_state.output_collector
    drift_detector = st.session_state.drift_detector

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


def predict_with_monitoring(input_df, raw_input_info: dict = None):
    """
    Make prediction with production monitoring.
    Tracks input features, prediction output, and latency.
    """
    start_time = time.time()
    prediction = model.predict(input_df)[0]
    latency_ms = (time.time() - start_time) * 1000
    
    if MONITORING_ENABLED:
        # Collect input data
        request_id = input_collector.collect(input_df)
        
        # Collect output with latency
        output_collector._collector.collect_output(
            output_data=float(prediction),
            request_id=request_id,
            latency_ms=latency_ms
        )
        
        # Track features for drift detection
        if raw_input_info:
            drift_detector.add_observation("Duration_hour", raw_input_info.get("Duration_hour", 0))
            drift_detector.add_observation("Total_Stops", raw_input_info.get("Total_Stops", 0))
            drift_detector.add_observation("Arrival_hour", raw_input_info.get("Arrival_hour", 0))
    
    return prediction, latency_ms


# --- Main Panel ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Prediction & Calendar")
    
    if predict_click:
        results = []
        total_latency = 0
        for airline_val in airlines:
            single_input = preprocess_inputs(
                journey_date.day,
                journey_date.month,
                airline_val,
                DEFAULT_TOTAL_STOPS,
                DEFAULT_ARRIVAL_TIME,
                DEFAULT_DURATION
            )
            dur_h, dur_m = parse_duration(DEFAULT_DURATION)
            raw_info = {
                "Duration_hour": dur_h,
                "Total_Stops": DEFAULT_TOTAL_STOPS,
                "Arrival_hour": DEFAULT_ARRIVAL_TIME.hour
            }
            prediction, latency = predict_with_monitoring(single_input, raw_info)
            total_latency += latency
            results.append({
                "Airline": airline_val,
                "Predicted Price": f"₹{prediction:.2f}",
                "Latency (ms)": f"{latency:.1f}"
            })
        
        st.write(pd.DataFrame(results))
        if MONITORING_ENABLED:
            st.caption(f"Monitored | Total latency: {total_latency:.1f}ms")
    
    if start_date <= end_date:
        all_data = []
        dates = pd.date_range(start_date, end_date)
        dur_h, dur_m = parse_duration(DEFAULT_DURATION)
        for airline_val in airlines:
            prices = []
            for d in dates:
                input_df = preprocess_inputs(
                    d.day, d.month, airline_val,
                    DEFAULT_TOTAL_STOPS,
                    DEFAULT_ARRIVAL_TIME,
                    DEFAULT_DURATION
                )
                raw_info = {
                    "Duration_hour": dur_h,
                    "Total_Stops": DEFAULT_TOTAL_STOPS,
                    "Arrival_hour": DEFAULT_ARRIVAL_TIME.hour
                }
                pred, _ = predict_with_monitoring(input_df, raw_info)
                prices.append(pred)
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

# =============================================================================
# PRODUCTION MONITORING DASHBOARD
# =============================================================================
if MONITORING_ENABLED:
    st.markdown("---")
    st.header("Production Monitoring Dashboard")
    
    # Get metrics from collectors
    input_metrics = input_collector.get_metrics()
    output_metrics = output_collector.get_metrics()
    
    # Metrics Row
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.metric(
            label="Total Requests",
            value=input_metrics.get("total_requests", 0)
        )
    
    with metric_cols[1]:
        avg_latency = output_metrics.get("avg_latency_ms", 0)
        st.metric(
            label="Avg Latency",
            value=f"{avg_latency:.1f}ms" if avg_latency else "N/A"
        )
    
    with metric_cols[2]:
        error_rate = output_metrics.get("error_rate", 0)
        st.metric(
            label="Error Rate",
            value=f"{error_rate:.2%}"
        )
    
    with metric_cols[3]:
        uptime = input_metrics.get("uptime_seconds", 0)
        uptime_mins = uptime / 60
        st.metric(
            label="Session Uptime",
            value=f"{uptime_mins:.1f} min"
        )
    
    # Latency Distribution and Drift Detection
    monitor_col1, monitor_col2 = st.columns(2)
    
    with monitor_col1:
        st.subheader("Latency Stats")
        if output_metrics.get("avg_latency_ms"):
            latency_data = {
                "Metric": ["Min", "Avg", "P50", "P95", "Max"],
                "Latency (ms)": [
                    output_metrics.get("min_latency_ms", 0),
                    output_metrics.get("avg_latency_ms", 0),
                    output_metrics.get("p50_latency_ms", 0) or output_metrics.get("avg_latency_ms", 0),
                    output_metrics.get("p95_latency_ms", "N/A"),
                    output_metrics.get("max_latency_ms", 0)
                ]
            }
            st.dataframe(pd.DataFrame(latency_data), use_container_width=True)
        else:
            st.info("Make predictions to see latency statistics")
    
    with monitor_col2:
        st.subheader("Data Drift Detection")
        drift_features = ["Duration_hour", "Total_Stops", "Arrival_hour"]
        drift_results = []
        
        for feature in drift_features:
            drift_check = drift_detector.check_drift(feature)
            if drift_check.get("status") == "insufficient_data":
                status = "Pending - Need more data"
            elif drift_check.get("is_drifted"):
                status = "DRIFT DETECTED"
            else:
                status = "No drift"
            
            drift_results.append({
                "Feature": feature,
                "Status": status,
                "Z-Score": f"{drift_check.get('z_score', 0):.2f}" if drift_check.get('z_score') else "N/A"
            })
        
        st.dataframe(pd.DataFrame(drift_results), use_container_width=True)
    
    # Recent Predictions
    with st.expander("Recent Predictions (Last 10)"):
        recent = output_collector._collector.get_recent_predictions(10)
        if recent:
            recent_df = pd.DataFrame([
                {
                    "Time": r.get("timestamp", "")[:19],
                    "Prediction": f"₹{r.get('prediction', 0):.2f}",
                    "Latency (ms)": f"{r.get('latency_ms', 0):.1f}"
                }
                for r in recent
            ])
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("No predictions yet")
    
    # Export logs button
    col_export1, col_export2 = st.columns(2)
    with col_export1:
        if st.button("Export Monitoring Logs"):
            try:
                input_file = input_collector.flush()
                output_file = output_collector.flush()
                st.success(f"Logs exported to:\n- {input_file}\n- {output_file}")
            except Exception as e:
                st.error(f"Export failed: {e}")
    
    with col_export2:
        if st.button("Refresh Metrics"):
            st.rerun()
else:
    st.markdown("---")
    st.caption("Production monitoring not enabled. Install monitoring module for real-time metrics.")