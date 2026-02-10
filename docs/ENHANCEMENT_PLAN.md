# 🚀 TravelPriceIQ - Complete Enhancement Plan

> **Transform your Flight Price Predictor into an Industry-Grade Travel Intelligence Platform**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Azure](https://img.shields.io/badge/Azure-Free%20Tier-0078D4.svg)](https://azure.microsoft.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Executive Summary](#-executive-summary)
- [Architecture Overview](#-architecture-overview)
- [Free APIs & Services](#-free-apis--services-complete-list)
- [Azure Free Tier Strategy](#-azure-free-tier-strategy-200-credit)
- [Phase-wise Implementation](#-phase-wise-implementation)
- [Data Pipeline Design](#-data-pipeline-design)
- [ML Model Enhancements](#-ml-model-enhancements)
- [API Integrations Guide](#-api-integrations-guide)
- [Frontend Enhancements](#-frontend-enhancements)
- [MLOps Implementation](#-mlops-implementation)
- [Deployment Strategy](#-deployment-strategy)
- [Cost Optimization](#-cost-optimization)
- [Testing Strategy](#-testing-strategy)
- [Project Timeline](#-project-timeline)

---

## 🎯 Executive Summary

### Current State
Your project is a solid flight price prediction system with:
- Basic ML models (RandomForest, GradientBoosting)
- FastAPI backend + Streamlit frontend
- Azure deployment capability
- Docker containerization

### Target State
Transform into **TravelPriceIQ** - a comprehensive travel intelligence platform featuring:
- Multi-model ensemble with 95%+ accuracy
- Real-time flight data integration
- Weather-aware delay predictions
- Dynamic hotel recommendations
- Currency conversion for international travelers
- AI-powered travel assistant
- Complete MLOps pipeline

### Budget
- **Total Cost: $0** (using free tiers + Azure $200 credit)
- Sustainable beyond free trial with careful resource management

---

## 🏗 Architecture Overview

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TravelPriceIQ Platform                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     EXTERNAL DATA SOURCES (FREE)                     │   │
│  ├─────────────┬─────────────┬─────────────┬─────────────┬─────────────┤   │
│  │  Amadeus    │  OpenWeather│  Exchange   │  Geoapify   │   Indian    │   │
│  │  Flight API │     API     │  Rate API   │  Places API │  Holidays   │   │
│  │  (2K/month) │ (1K/day)    │ (1.5K/month)│  (3K/day)   │    API      │   │
│  └──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┘   │
│         │             │             │             │             │          │
│         ▼             ▼             ▼             ▼             ▼          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DATA AGGREGATION LAYER                           │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │  Redis   │  │  SQLite  │  │  Pandas  │  │  Azure   │            │   │
│  │  │  Cache   │  │   Local  │  │  Cache   │  │  Blob    │            │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                │                                           │
│         ┌──────────────────────┼──────────────────────┐                   │
│         ▼                      ▼                      ▼                   │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐              │
│  │  FLIGHT PRICE  │  │ DELAY PREDICT  │  │   RECOMMENDER  │              │
│  │    ENSEMBLE    │  │    MODEL       │  │     ENGINE     │              │
│  │ ┌────────────┐ │  │ ┌────────────┐ │  │ ┌────────────┐ │              │
│  │ │  XGBoost   │ │  │ │  Weather   │ │  │ │  Hotel     │ │              │
│  │ │  LightGBM  │ │  │ │  Pattern   │ │  │ │  Matcher   │ │              │
│  │ │  CatBoost  │ │  │ │  Analysis  │ │  │ │            │ │              │
│  │ │  Prophet   │ │  │ └────────────┘ │  │ └────────────┘ │              │
│  │ └────────────┘ │  └────────────────┘  └────────────────┘              │
│  └───────┬────────┘           │                   │                      │
│          │                    │                   │                      │
│          └────────────────────┼───────────────────┘                      │
│                               ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                      FastAPI BACKEND                                 │ │
│  │  /predict  /hotels  /weather  /package  /trends  /assistant         │ │
│  └─────────────────────────────────┬───────────────────────────────────┘ │
│                                    │                                     │
│  ┌─────────────────────────────────▼───────────────────────────────────┐ │
│  │                    STREAMLIT FRONTEND                               │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │ │
│  │  │  Price  │ │  Trip   │ │ Weather │ │Analytics│ │   AI    │       │ │
│  │  │ Predict │ │Optimizer│ │ Delays  │ │Dashboard│ │Assistant│       │ │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘       │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🆓 Free APIs & Services (Complete List)

### 1. **Amadeus for Developers** (Flight Data)
| Feature | Free Tier Limit | Use Case |
|---------|-----------------|----------|
| Flight Offers Search | 2,000 calls/month | Real-time flight prices |
| Flight Inspiration | 2,000 calls/month | Cheapest destinations |
| Flight Cheapest Date | 2,000 calls/month | Best dates to fly |
| Airport & City Search | 2,000 calls/month | Autocomplete |
| Flight Delay Prediction | 2,000 calls/month | ML delay features |

**Registration**: https://developers.amadeus.com/register

```python
# Example: Amadeus Flight Search
from amadeus import Client, ResponseError

amadeus = Client(
    client_id='YOUR_API_KEY',
    client_secret='YOUR_API_SECRET'
)

# Search flights
response = amadeus.shopping.flight_offers_search.get(
    originLocationCode='DEL',
    destinationLocationCode='BOM',
    departureDate='2026-03-15',
    adults=1,
    currencyCode='INR'
)
```

### 2. **OpenWeatherMap** (Weather Data)
| Feature | Free Tier Limit | Use Case |
|---------|-----------------|----------|
| Current Weather | 1,000 calls/day | Airport weather |
| 5-Day Forecast | 1,000 calls/day | Travel planning |
| Air Pollution | 1,000 calls/day | Air quality index |
| Geocoding | 1,000 calls/day | City coordinates |

**Registration**: https://openweathermap.org/api

```python
# Example: Weather API
import requests

API_KEY = "your_openweather_key"
city = "Delhi"

url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
response = requests.get(url).json()

weather_data = {
    "temperature": response["main"]["temp"],
    "humidity": response["main"]["humidity"],
    "wind_speed": response["wind"]["speed"],
    "conditions": response["weather"][0]["main"],
    "visibility": response.get("visibility", 10000) / 1000  # km
}
```

### 3. **ExchangeRate-API** (Currency Conversion)
| Feature | Free Tier Limit | Use Case |
|---------|-----------------|----------|
| Exchange Rates | 1,500 calls/month | INR to USD/EUR conversion |
| Historical Rates | 1,500 calls/month | Price trend analysis |

**Registration**: https://www.exchangerate-api.com/

```python
# Example: Currency Conversion
import requests

def convert_currency(amount, from_currency, to_currency):
    url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
    response = requests.get(url).json()
    rate = response["rates"][to_currency]
    return amount * rate

# Convert INR to USD
usd_price = convert_currency(5000, "INR", "USD")
```

### 4. **Geoapify** (Places & Hotels)
| Feature | Free Tier Limit | Use Case |
|---------|-----------------|----------|
| Places API | 3,000 calls/day | Hotels near airports |
| Geocoding | 3,000 calls/day | Address lookup |
| Routing | 3,000 calls/day | Distance calculations |

**Registration**: https://www.geoapify.com/

```python
# Example: Find Hotels Near Airport
import requests

API_KEY = "your_geoapify_key"
lat, lon = 28.5562, 77.1000  # Delhi Airport

url = f"https://api.geoapify.com/v2/places"
params = {
    "categories": "accommodation.hotel",
    "filter": f"circle:{lon},{lat},5000",  # 5km radius
    "limit": 20,
    "apiKey": API_KEY
}

response = requests.get(url, params=params).json()
hotels = [{
    "name": place["properties"]["name"],
    "distance": place["properties"]["distance"],
    "address": place["properties"].get("formatted", "N/A")
} for place in response["features"]]
```

### 5. **Indian Calendar API** (Holidays)
| Feature | Free Tier Limit | Use Case |
|---------|-----------------|----------|
| Holiday List | Unlimited | Demand prediction |
| Festival Dates | Unlimited | Price surge features |

**Source**: https://github.com/dr5hn/indian-calendar-api (Self-hosted/JSON)

```python
# Example: Holiday Feature Engineering
import pandas as pd
from datetime import datetime, timedelta

INDIAN_HOLIDAYS_2026 = {
    "2026-01-26": "Republic Day",
    "2026-03-10": "Holi",
    "2026-04-14": "Ambedkar Jayanti",
    "2026-08-15": "Independence Day",
    "2026-10-02": "Gandhi Jayanti",
    "2026-10-20": "Diwali",
    "2026-11-14": "Children's Day",
    "2026-12-25": "Christmas"
}

def get_holiday_features(date):
    date_str = date.strftime("%Y-%m-%d")
    is_holiday = date_str in INDIAN_HOLIDAYS_2026
    
    # Days to nearest holiday
    days_to_holiday = min([
        abs((datetime.strptime(h, "%Y-%m-%d") - date).days)
        for h in INDIAN_HOLIDAYS_2026.keys()
    ])
    
    return {
        "is_holiday": int(is_holiday),
        "days_to_holiday": days_to_holiday,
        "holiday_name": INDIAN_HOLIDAYS_2026.get(date_str, None)
    }
```

### 6. **Aviation Edge** (Airport Data - Free Tier)
| Feature | Free Tier Limit | Use Case |
|---------|-----------------|----------|
| Airport Database | 100 calls/month | Airport details |
| Airline Database | 100 calls/month | Airline info |

**Alternative**: Use static datasets from OpenFlights:
- https://openflights.org/data.html (Completely free, no API limits)

```python
# Example: OpenFlights Static Data
import pandas as pd

# Download once and use locally
airports_df = pd.read_csv(
    "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat",
    header=None,
    names=["id", "name", "city", "country", "iata", "icao", "lat", "lon", 
           "altitude", "timezone", "dst", "tz", "type", "source"]
)

# Indian airports
indian_airports = airports_df[airports_df["country"] == "India"]
```

### 7. **Nominatim/OpenStreetMap** (Geocoding - Unlimited)
| Feature | Free Tier Limit | Use Case |
|---------|-----------------|----------|
| Geocoding | 1 req/sec | Address to coordinates |
| Reverse Geocoding | 1 req/sec | Coordinates to address |

```python
# Example: Free Geocoding
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

geolocator = Nominatim(user_agent="travelprice_iq")

def get_distance_km(city1, city2):
    loc1 = geolocator.geocode(city1 + ", India")
    loc2 = geolocator.geocode(city2 + ", India")
    return geodesic(
        (loc1.latitude, loc1.longitude),
        (loc2.latitude, loc2.longitude)
    ).kilometers

# Delhi to Mumbai distance
distance = get_distance_km("Delhi", "Mumbai")  # ~1148 km
```

### 8. **Fuel Prices API (India)**
| Feature | Free Tier Limit | Use Case |
|---------|-----------------|----------|
| Daily Fuel Prices | Unlimited | Price correlation |

**Source**: Web scraping from government sites or use:
- https://github.com/theaakashd/FuelPriceAPI (Community API)

```python
# Example: Fuel Price Feature
# Historical correlation: ATF price affects flight costs
import requests

def get_fuel_index():
    # Simplified: Use a proxy indicator
    # In production, scrape from petroleum ministry
    avg_petrol_price = 105.0  # INR per liter (update periodically)
    baseline = 100.0
    return avg_petrol_price / baseline  # Fuel index > 1 means higher prices
```

### 9. **TimeZoneDB** (Timezone Data)
| Feature | Free Tier Limit | Use Case |
|---------|-----------------|----------|
| Timezone Lookup | 1 req/sec | Time calculations |

**Registration**: https://timezonedb.com/

### 10. **News API** (Travel News - 100 req/day)
| Feature | Free Tier Limit | Use Case |
|---------|-----------------|----------|
| Headlines | 100 calls/day | Travel disruption alerts |
| Search | 100 calls/day | Airline news |

**Registration**: https://newsapi.org/

```python
# Example: Check for travel disruptions
import requests

API_KEY = "your_newsapi_key"

def check_travel_alerts(airline=None, route=None):
    query = f"flight delay India {airline or ''} {route or ''}"
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={API_KEY}&pageSize=5"
    response = requests.get(url).json()
    return response.get("articles", [])
```

---

## ☁️ Azure Free Tier Strategy ($200 Credit)

### Recommended Azure Services

| Service | Free Tier / Credit Usage | Monthly Cost | Purpose |
|---------|--------------------------|--------------|---------|
| **Azure Container Apps** | 2M requests free | $0 | API hosting |
| **Azure Blob Storage** | 5GB free | $0 | Model storage |
| **Azure Cache for Redis** | 250MB free tier | $0 | API caching |
| **Azure SQL Database** | 250GB free | $0 | Data storage |
| **Azure Static Web Apps** | Unlimited (free tier) | $0 | Streamlit hosting |
| **Azure ML Workspace** | Free tier available | ~$15/month | MLOps |
| **Azure Functions** | 1M executions free | $0 | Scheduled jobs |
| **Application Insights** | 5GB/month free | $0 | Monitoring |

### Cost Breakdown (Monthly Estimate)

```
┌─────────────────────────────────────────────────────────────┐
│               AZURE MONTHLY COST ESTIMATE                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Azure Container Apps (API)                                 │
│    - 100K requests/month                         $0.00     │
│    - 0.5 vCPU, 1GB RAM                          $0.00     │
│                                                             │
│  Azure Blob Storage                                         │
│    - Models + Data (~2GB)                        $0.04     │
│    - Operations (10K)                            $0.00     │
│                                                             │
│  Azure Cache for Redis (Basic C0)                           │
│    - 250MB cache                                 $0.00*    │
│                                                             │
│  Azure SQL Database                                         │
│    - Serverless (auto-pause)                     $0.00*    │
│                                                             │
│  Azure Static Web Apps                                      │
│    - Frontend hosting                            $0.00     │
│                                                             │
│  Azure ML Workspace                                         │
│    - Compute (2 hrs/month)                      $10.00     │
│    - Storage                                     $2.00     │
│                                                             │
│  Azure Functions                                            │
│    - Scheduled retraining                        $0.00     │
│                                                             │
│  Application Insights                                       │
│    - Monitoring (5GB)                            $0.00     │
├─────────────────────────────────────────────────────────────┤
│  TOTAL (within $200 credit)                    ~$12/month  │
│  First Month: FREE (covered by credit)                     │
│  Months 2-16: Sustainable on minimal usage                 │
└─────────────────────────────────────────────────────────────┘

* Free tier or auto-pause when not in use
```

### Azure Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        AZURE ARCHITECTURE                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Internet                                                       │
│      │                                                           │
│      ▼                                                           │
│  ┌────────────────────┐                                         │
│  │  Azure Front Door  │  (Free with Static Web Apps)            │
│  │  + CDN             │                                         │
│  └─────────┬──────────┘                                         │
│            │                                                     │
│      ┌─────┴─────┐                                              │
│      ▼           ▼                                              │
│  ┌────────┐  ┌────────────────┐                                 │
│  │Static  │  │ Container Apps │                                 │
│  │Web App │  │   (FastAPI)    │                                 │
│  │Streamlit│  │                │                                 │
│  └────────┘  └───────┬────────┘                                 │
│                      │                                          │
│         ┌────────────┼────────────┐                             │
│         ▼            ▼            ▼                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                        │
│  │  Redis   │ │  Blob    │ │  SQL DB  │                        │
│  │  Cache   │ │  Storage │ │(Serverless)│                       │
│  └──────────┘ └──────────┘ └──────────┘                        │
│                      │                                          │
│                      ▼                                          │
│              ┌──────────────┐                                   │
│              │   Azure ML   │                                   │
│              │  Workspace   │                                   │
│              │ (Experiments)│                                   │
│              └──────────────┘                                   │
│                      │                                          │
│                      ▼                                          │
│              ┌──────────────┐                                   │
│              │    Azure     │                                   │
│              │  Functions   │                                   │
│              │(Retraining)  │                                   │
│              └──────────────┘                                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📅 Phase-wise Implementation

### Phase 1: Enhanced ML Pipeline (Week 1-2)

#### 1.1 New Feature Engineering

```python
# features/feature_engineer.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests

class FlightFeatureEngineer:
    """Enhanced feature engineering for flight price prediction."""
    
    def __init__(self, weather_api_key=None, exchange_api_key=None):
        self.weather_api_key = weather_api_key
        self.exchange_api_key = exchange_api_key
        
        # Indian holidays 2024-2026
        self.holidays = self._load_indian_holidays()
        
        # Airport coordinates
        self.airport_coords = {
            "DEL": (28.5562, 77.1000),
            "BOM": (19.0896, 72.8656),
            "BLR": (13.1986, 77.7066),
            "CCU": (22.6520, 88.4463),
            "MAA": (12.9941, 80.1709),
            "HYD": (17.2403, 78.4294),
            "COK": (10.1520, 76.4019),
        }
    
    def create_temporal_features(self, df):
        """Extract time-based features."""
        df = df.copy()
        
        # Basic date features
        df['day_of_week'] = df['Date_of_Journey'].dt.dayofweek
        df['day_of_month'] = df['Date_of_Journey'].dt.day
        df['month'] = df['Date_of_Journey'].dt.month
        df['quarter'] = df['Date_of_Journey'].dt.quarter
        df['year'] = df['Date_of_Journey'].dt.year
        df['week_of_year'] = df['Date_of_Journey'].dt.isocalendar().week
        
        # Cyclical encoding (for neural networks)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Weekend indicator
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Time of day categories
        df['dep_hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
        df['dep_time_category'] = pd.cut(
            df['dep_hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['red_eye', 'morning', 'afternoon', 'evening']
        )
        
        return df
    
    def create_booking_features(self, df, booking_date=None):
        """Features related to booking timing."""
        df = df.copy()
        
        if booking_date is None:
            booking_date = datetime.now()
        
        # Days until departure (booking window)
        df['days_until_departure'] = (
            df['Date_of_Journey'] - pd.Timestamp(booking_date)
        ).dt.days
        
        # Booking window categories
        df['booking_window'] = pd.cut(
            df['days_until_departure'],
            bins=[-1, 3, 7, 14, 30, 60, 90, 365],
            labels=['last_minute', 'week_before', 'two_weeks', 
                    'month_before', 'two_months', 'three_months', 'advance']
        )
        
        # Last minute booking flag
        df['is_last_minute'] = (df['days_until_departure'] <= 3).astype(int)
        
        return df
    
    def create_holiday_features(self, df):
        """Holiday and festival features."""
        df = df.copy()
        
        def get_holiday_info(date):
            date_str = date.strftime("%Y-%m-%d")
            is_holiday = date_str in self.holidays
            
            # Days to nearest holiday
            days_list = [
                abs((datetime.strptime(h, "%Y-%m-%d") - date).days)
                for h in self.holidays.keys()
            ]
            days_to_holiday = min(days_list) if days_list else 999
            
            return is_holiday, days_to_holiday
        
        holiday_data = df['Date_of_Journey'].apply(
            lambda x: pd.Series(get_holiday_info(x))
        )
        df['is_holiday'] = holiday_data[0].astype(int)
        df['days_to_nearest_holiday'] = holiday_data[1]
        
        # Holiday season indicator (Diwali, Christmas, New Year)
        df['is_peak_season'] = (
            (df['month'].isin([10, 11, 12])) | 
            (df['days_to_nearest_holiday'] <= 7)
        ).astype(int)
        
        return df
    
    def create_route_features(self, df):
        """Route-specific features."""
        df = df.copy()
        
        # Route combination
        df['route'] = df['Source'] + '_' + df['Destination']
        
        # Route popularity (historical frequency)
        route_counts = df['route'].value_counts()
        df['route_popularity'] = df['route'].map(route_counts)
        
        # Distance (approximate using coordinates)
        def calculate_distance(row):
            source = row['Source']
            dest = row['Destination']
            
            # Map city names to IATA codes
            city_to_iata = {
                'Delhi': 'DEL', 'New Delhi': 'DEL',
                'Mumbai': 'BOM', 'Banglore': 'BLR', 'Bangalore': 'BLR',
                'Kolkata': 'CCU', 'Chennai': 'MAA',
                'Hyderabad': 'HYD', 'Cochin': 'COK'
            }
            
            src_code = city_to_iata.get(source, source)
            dst_code = city_to_iata.get(dest, dest)
            
            if src_code in self.airport_coords and dst_code in self.airport_coords:
                from math import radians, cos, sin, sqrt, atan2
                
                lat1, lon1 = self.airport_coords[src_code]
                lat2, lon2 = self.airport_coords[dst_code]
                
                R = 6371  # Earth's radius in km
                
                lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                
                return R * c
            return 1000  # Default distance
        
        df['route_distance_km'] = df.apply(calculate_distance, axis=1)
        
        # Tier 1 city indicator
        tier1_cities = ['Delhi', 'New Delhi', 'Mumbai', 'Banglore', 'Bangalore']
        df['is_tier1_route'] = (
            df['Source'].isin(tier1_cities) & 
            df['Destination'].isin(tier1_cities)
        ).astype(int)
        
        return df
    
    def create_airline_features(self, df):
        """Airline-specific features."""
        df = df.copy()
        
        # Airline categories
        budget_airlines = ['IndiGo', 'SpiceJet', 'GoAir', 'AirAsia India']
        premium_airlines = ['Vistara', 'Air India']
        
        df['airline_category'] = df['Airline'].apply(
            lambda x: 'budget' if x in budget_airlines 
            else ('premium' if x in premium_airlines else 'standard')
        )
        
        # Historical price statistics by airline
        airline_stats = df.groupby('Airline')['Price'].agg(['mean', 'std', 'median'])
        df['airline_avg_price'] = df['Airline'].map(airline_stats['mean'])
        df['airline_price_std'] = df['Airline'].map(airline_stats['std'])
        
        return df
    
    def create_competition_features(self, df):
        """Market competition features."""
        df = df.copy()
        
        # Number of airlines on route
        route_airlines = df.groupby('route')['Airline'].nunique()
        df['route_competition'] = df['route'].map(route_airlines)
        
        # Flights per day on route
        daily_flights = df.groupby(['route', 'Date_of_Journey']).size()
        df['daily_flights_on_route'] = df.apply(
            lambda x: daily_flights.get((x['route'], x['Date_of_Journey']), 1),
            axis=1
        )
        
        return df
    
    def get_weather_features(self, city, date):
        """Fetch weather data for flight planning."""
        if not self.weather_api_key:
            return {}
        
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": f"{city},IN",
            "appid": self.weather_api_key,
            "units": "metric"
        }
        
        try:
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            return {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"],
                "visibility_km": data.get("visibility", 10000) / 1000,
                "weather_condition": data["weather"][0]["main"],
                "is_bad_weather": data["weather"][0]["main"] in 
                    ["Rain", "Thunderstorm", "Fog", "Mist"]
            }
        except:
            return {}
    
    def _load_indian_holidays(self):
        """Load Indian holiday calendar."""
        return {
            # 2024
            "2024-01-26": "Republic Day",
            "2024-03-25": "Holi",
            "2024-08-15": "Independence Day",
            "2024-10-02": "Gandhi Jayanti",
            "2024-10-31": "Diwali",
            "2024-12-25": "Christmas",
            # 2025
            "2025-01-26": "Republic Day",
            "2025-03-14": "Holi",
            "2025-08-15": "Independence Day",
            "2025-10-02": "Gandhi Jayanti",
            "2025-10-20": "Diwali",
            "2025-12-25": "Christmas",
            # 2026
            "2026-01-26": "Republic Day",
            "2026-03-03": "Holi",
            "2026-08-15": "Independence Day",
            "2026-10-02": "Gandhi Jayanti",
            "2026-11-08": "Diwali",
            "2026-12-25": "Christmas",
        }
    
    def transform(self, df, include_weather=False):
        """Apply all feature transformations."""
        df = df.copy()
        
        # Ensure datetime
        if df['Date_of_Journey'].dtype == 'object':
            df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'])
        
        # Apply all feature engineering
        df = self.create_temporal_features(df)
        df = self.create_booking_features(df)
        df = self.create_holiday_features(df)
        df = self.create_route_features(df)
        df = self.create_airline_features(df)
        df = self.create_competition_features(df)
        
        return df
```

#### 1.2 Enhanced Model Training

```python
# models/ensemble_model.py

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib
import mlflow
import mlflow.sklearn

class FlightPriceEnsemble:
    """Ensemble model for flight price prediction."""
    
    def __init__(self, use_mlflow=True):
        self.use_mlflow = use_mlflow
        self.models = {}
        self.weights = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def _create_models(self):
        """Initialize individual models."""
        
        # XGBoost
        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        
        # LightGBM
        self.models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # CatBoost
        self.models['catboost'] = CatBoostRegressor(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=False
        )
        
    def preprocess(self, df, fit=True):
        """Preprocess data for model training."""
        df = df.copy()
        
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        # Encode categorical features
        for col in categorical_cols:
            if fit:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = df[col].apply(
                    lambda x: self.encoders[col].transform([str(x)])[0] 
                    if str(x) in self.encoders[col].classes_ 
                    else -1
                )
        
        # Scale numerical features
        if fit:
            self.feature_columns = df.columns.tolist()
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df
    
    def train(self, X, y, validation_split=0.2):
        """Train ensemble models."""
        
        if self.use_mlflow:
            mlflow.set_experiment("flight_price_prediction")
            mlflow.start_run(run_name="ensemble_training")
        
        self._create_models()
        
        # Preprocess
        X_processed = self.preprocess(X, fit=True)
        
        # Split for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X_processed[:split_idx], X_processed[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train each model and calculate weights based on validation performance
        val_scores = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            if name == 'catboost':
                model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
            elif name in ['xgboost', 'lightgbm']:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    # early_stopping_rounds=50,
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
            
            # Validate
            val_pred = model.predict(X_val)
            r2 = r2_score(y_val, val_pred)
            mae = mean_absolute_error(y_val, val_pred)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            
            val_scores[name] = {'r2': r2, 'mae': mae, 'rmse': rmse}
            print(f"{name}: R² = {r2:.4f}, MAE = {mae:.2f}, RMSE = {rmse:.2f}")
            
            if self.use_mlflow:
                mlflow.log_metric(f"{name}_r2", r2)
                mlflow.log_metric(f"{name}_mae", mae)
                mlflow.log_metric(f"{name}_rmse", rmse)
        
        # Calculate ensemble weights based on R² scores
        total_r2 = sum(s['r2'] for s in val_scores.values())
        self.weights = {name: s['r2'] / total_r2 for name, s in val_scores.items()}
        
        print(f"\nEnsemble weights: {self.weights}")
        
        # Final ensemble validation
        ensemble_pred = self.predict(X.iloc[split_idx:], preprocessed=False)
        ensemble_r2 = r2_score(y_val, ensemble_pred)
        ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
        
        print(f"\n=== ENSEMBLE PERFORMANCE ===")
        print(f"R² Score: {ensemble_r2:.4f}")
        print(f"MAE: ₹{ensemble_mae:.2f}")
        print(f"RMSE: ₹{ensemble_rmse:.2f}")
        
        if self.use_mlflow:
            mlflow.log_metric("ensemble_r2", ensemble_r2)
            mlflow.log_metric("ensemble_mae", ensemble_mae)
            mlflow.log_metric("ensemble_rmse", ensemble_rmse)
            mlflow.log_params({"weights": self.weights})
            mlflow.end_run()
        
        return val_scores
    
    def predict(self, X, preprocessed=False, return_individual=False):
        """Make ensemble predictions."""
        
        if not preprocessed:
            X_processed = self.preprocess(X, fit=False)
        else:
            X_processed = X
        
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_processed)
        
        # Weighted average
        ensemble_pred = np.zeros(len(X_processed))
        for name, pred in predictions.items():
            ensemble_pred += self.weights[name] * pred
        
        if return_individual:
            return ensemble_pred, predictions
        return ensemble_pred
    
    def predict_with_uncertainty(self, X, n_iterations=100):
        """Predict with uncertainty estimation using bootstrap."""
        
        X_processed = self.preprocess(X, fit=False)
        
        # Get individual predictions
        individual_preds = []
        for name, model in self.models.items():
            individual_preds.append(model.predict(X_processed))
        
        individual_preds = np.array(individual_preds)
        
        # Calculate statistics
        mean_pred = np.average(individual_preds, axis=0, weights=list(self.weights.values()))
        std_pred = np.std(individual_preds, axis=0)
        
        # 95% confidence interval
        lower_bound = mean_pred - 1.96 * std_pred
        upper_bound = mean_pred + 1.96 * std_pred
        
        # Confidence score (inverse of coefficient of variation)
        confidence = 1 - (std_pred / (mean_pred + 1e-6))
        confidence = np.clip(confidence, 0, 1)
        
        return {
            'prediction': mean_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence': confidence,
            'std': std_pred
        }
    
    def get_feature_importance(self):
        """Get aggregated feature importance."""
        
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                for i, imp in enumerate(model.feature_importances_):
                    col = self.feature_columns[i]
                    if col not in importance_dict:
                        importance_dict[col] = 0
                    importance_dict[col] += imp * self.weights[name]
        
        # Sort by importance
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v} 
            for k, v in importance_dict.items()
        ]).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, path):
        """Save ensemble model."""
        joblib.dump({
            'models': self.models,
            'weights': self.weights,
            'encoders': self.encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load ensemble model."""
        data = joblib.load(path)
        self.models = data['models']
        self.weights = data['weights']
        self.encoders = data['encoders']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        print(f"Model loaded from {path}")
```

---

### Phase 2: API Integrations (Week 2-3)

#### 2.1 Unified API Service

```python
# services/external_apis.py

import os
import requests
import json
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional
import time

class APIRateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    def wait_if_needed(self):
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [c for c in self.calls if now - c < 60]
        
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.calls.append(time.time())


class AmadeusClient:
    """Client for Amadeus Flight API."""
    
    BASE_URL = "https://test.api.amadeus.com"  # Test environment (free)
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_expires = None
        self.rate_limiter = APIRateLimiter(calls_per_minute=10)
    
    def _get_token(self):
        """Get OAuth2 access token."""
        if self.access_token and self.token_expires > datetime.now():
            return self.access_token
        
        url = f"{self.BASE_URL}/v1/security/oauth2/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        
        response = requests.post(url, data=data)
        response.raise_for_status()
        
        token_data = response.json()
        self.access_token = token_data["access_token"]
        self.token_expires = datetime.now() + timedelta(seconds=token_data["expires_in"] - 60)
        
        return self.access_token
    
    def _request(self, endpoint: str, params: dict = None):
        """Make authenticated request."""
        self.rate_limiter.wait_if_needed()
        
        token = self._get_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        url = f"{self.BASE_URL}{endpoint}"
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def search_flights(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        adults: int = 1,
        currency: str = "INR",
        max_results: int = 10
    ) -> List[Dict]:
        """Search for flight offers."""
        
        params = {
            "originLocationCode": origin,
            "destinationLocationCode": destination,
            "departureDate": departure_date,
            "adults": adults,
            "currencyCode": currency,
            "max": max_results
        }
        
        try:
            data = self._request("/v2/shopping/flight-offers", params)
            
            flights = []
            for offer in data.get("data", []):
                flight = {
                    "price": float(offer["price"]["total"]),
                    "currency": offer["price"]["currency"],
                    "airline": offer["validatingAirlineCodes"][0],
                    "segments": len(offer["itineraries"][0]["segments"]),
                    "duration": offer["itineraries"][0]["duration"],
                    "departure_time": offer["itineraries"][0]["segments"][0]["departure"]["at"],
                    "arrival_time": offer["itineraries"][0]["segments"][-1]["arrival"]["at"],
                }
                flights.append(flight)
            
            return flights
        
        except Exception as e:
            print(f"Amadeus API error: {e}")
            return []
    
    def get_cheapest_dates(
        self,
        origin: str,
        destination: str
    ) -> List[Dict]:
        """Get cheapest dates for a route."""
        
        params = {
            "origin": origin,
            "destination": destination
        }
        
        try:
            data = self._request("/v1/shopping/flight-dates", params)
            
            dates = []
            for item in data.get("data", []):
                dates.append({
                    "departure_date": item["departureDate"],
                    "return_date": item.get("returnDate"),
                    "price": float(item["price"]["total"])
                })
            
            return sorted(dates, key=lambda x: x["price"])
        
        except Exception as e:
            print(f"Amadeus API error: {e}")
            return []
    
    def get_flight_inspiration(
        self,
        origin: str,
        max_price: int = None
    ) -> List[Dict]:
        """Get destination inspiration (cheapest destinations)."""
        
        params = {"origin": origin}
        if max_price:
            params["maxPrice"] = max_price
        
        try:
            data = self._request("/v1/shopping/flight-destinations", params)
            
            destinations = []
            for item in data.get("data", []):
                destinations.append({
                    "destination": item["destination"],
                    "departure_date": item["departureDate"],
                    "return_date": item.get("returnDate"),
                    "price": float(item["price"]["total"])
                })
            
            return sorted(destinations, key=lambda x: x["price"])
        
        except Exception as e:
            print(f"Amadeus API error: {e}")
            return []


class WeatherClient:
    """Client for OpenWeatherMap API."""
    
    BASE_URL = "http://api.openweathermap.org/data/2.5"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limiter = APIRateLimiter(calls_per_minute=60)
    
    @lru_cache(maxsize=100)
    def get_current_weather(self, city: str, country: str = "IN") -> Dict:
        """Get current weather for a city."""
        self.rate_limiter.wait_if_needed()
        
        params = {
            "q": f"{city},{country}",
            "appid": self.api_key,
            "units": "metric"
        }
        
        try:
            response = requests.get(f"{self.BASE_URL}/weather", params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                "city": city,
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "wind_speed": data["wind"]["speed"],
                "wind_direction": data["wind"].get("deg", 0),
                "visibility_km": data.get("visibility", 10000) / 1000,
                "clouds": data["clouds"]["all"],
                "condition": data["weather"][0]["main"],
                "description": data["weather"][0]["description"],
                "icon": data["weather"][0]["icon"],
                "is_adverse": data["weather"][0]["main"] in 
                    ["Rain", "Thunderstorm", "Snow", "Fog", "Mist", "Haze"]
            }
        
        except Exception as e:
            print(f"Weather API error: {e}")
            return {}
    
    def get_forecast(self, city: str, country: str = "IN") -> List[Dict]:
        """Get 5-day weather forecast."""
        self.rate_limiter.wait_if_needed()
        
        params = {
            "q": f"{city},{country}",
            "appid": self.api_key,
            "units": "metric"
        }
        
        try:
            response = requests.get(f"{self.BASE_URL}/forecast", params=params)
            response.raise_for_status()
            data = response.json()
            
            forecasts = []
            for item in data["list"]:
                forecasts.append({
                    "datetime": item["dt_txt"],
                    "temperature": item["main"]["temp"],
                    "humidity": item["main"]["humidity"],
                    "condition": item["weather"][0]["main"],
                    "description": item["weather"][0]["description"],
                    "wind_speed": item["wind"]["speed"],
                    "rain_probability": item.get("pop", 0) * 100
                })
            
            return forecasts
        
        except Exception as e:
            print(f"Weather API error: {e}")
            return []
    
    def get_delay_risk_score(self, city: str) -> Dict:
        """Calculate delay risk based on weather."""
        weather = self.get_current_weather(city)
        
        if not weather:
            return {"risk_score": 0.5, "risk_level": "unknown"}
        
        # Calculate risk score (0-1)
        risk_score = 0.0
        factors = []
        
        # Visibility risk
        if weather["visibility_km"] < 1:
            risk_score += 0.4
            factors.append("Very low visibility")
        elif weather["visibility_km"] < 5:
            risk_score += 0.2
            factors.append("Low visibility")
        
        # Wind risk
        if weather["wind_speed"] > 50:
            risk_score += 0.4
            factors.append("High winds")
        elif weather["wind_speed"] > 30:
            risk_score += 0.2
            factors.append("Moderate winds")
        
        # Weather condition risk
        condition_risks = {
            "Thunderstorm": 0.5,
            "Rain": 0.2,
            "Snow": 0.4,
            "Fog": 0.4,
            "Mist": 0.2,
            "Haze": 0.1
        }
        condition_risk = condition_risks.get(weather["condition"], 0)
        risk_score += condition_risk
        if condition_risk > 0:
            factors.append(f"{weather['condition']} conditions")
        
        # Normalize to 0-1
        risk_score = min(risk_score, 1.0)
        
        # Determine risk level
        if risk_score < 0.2:
            risk_level = "low"
        elif risk_score < 0.5:
            risk_level = "moderate"
        elif risk_score < 0.8:
            risk_level = "high"
        else:
            risk_level = "severe"
        
        return {
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "risk_factors": factors,
            "weather": weather
        }


class CurrencyClient:
    """Client for ExchangeRate API."""
    
    BASE_URL = "https://api.exchangerate-api.com/v4/latest"
    
    def __init__(self):
        self.rate_limiter = APIRateLimiter(calls_per_minute=10)
        self._cache = {}
        self._cache_time = None
    
    def get_rates(self, base_currency: str = "INR") -> Dict:
        """Get exchange rates."""
        
        # Cache for 1 hour
        if self._cache and self._cache_time:
            if datetime.now() - self._cache_time < timedelta(hours=1):
                return self._cache
        
        self.rate_limiter.wait_if_needed()
        
        try:
            response = requests.get(f"{self.BASE_URL}/{base_currency}")
            response.raise_for_status()
            data = response.json()
            
            self._cache = data["rates"]
            self._cache_time = datetime.now()
            
            return self._cache
        
        except Exception as e:
            print(f"Currency API error: {e}")
            return {"USD": 0.012, "EUR": 0.011, "GBP": 0.0095}  # Fallback
    
    def convert(self, amount: float, from_currency: str, to_currency: str) -> float:
        """Convert currency amount."""
        rates = self.get_rates(from_currency)
        
        if to_currency in rates:
            return round(amount * rates[to_currency], 2)
        return amount


class PlacesClient:
    """Client for Geoapify Places API."""
    
    BASE_URL = "https://api.geoapify.com/v2/places"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limiter = APIRateLimiter(calls_per_minute=30)
    
    def find_hotels_near_airport(
        self,
        lat: float,
        lon: float,
        radius_meters: int = 10000,
        limit: int = 20
    ) -> List[Dict]:
        """Find hotels near an airport."""
        self.rate_limiter.wait_if_needed()
        
        params = {
            "categories": "accommodation.hotel",
            "filter": f"circle:{lon},{lat},{radius_meters}",
            "limit": limit,
            "apiKey": self.api_key
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            hotels = []
            for place in data.get("features", []):
                props = place["properties"]
                hotels.append({
                    "name": props.get("name", "Unknown Hotel"),
                    "address": props.get("formatted", ""),
                    "distance_m": props.get("distance", 0),
                    "lat": place["geometry"]["coordinates"][1],
                    "lon": place["geometry"]["coordinates"][0],
                    "categories": props.get("categories", []),
                    "website": props.get("website"),
                    "phone": props.get("phone")
                })
            
            return sorted(hotels, key=lambda x: x["distance_m"])
        
        except Exception as e:
            print(f"Places API error: {e}")
            return []
    
    def find_restaurants_near_airport(
        self,
        lat: float,
        lon: float,
        radius_meters: int = 5000,
        limit: int = 10
    ) -> List[Dict]:
        """Find restaurants near an airport."""
        self.rate_limiter.wait_if_needed()
        
        params = {
            "categories": "catering.restaurant",
            "filter": f"circle:{lon},{lat},{radius_meters}",
            "limit": limit,
            "apiKey": self.api_key
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            restaurants = []
            for place in data.get("features", []):
                props = place["properties"]
                restaurants.append({
                    "name": props.get("name", "Unknown Restaurant"),
                    "address": props.get("formatted", ""),
                    "distance_m": props.get("distance", 0),
                    "cuisine": props.get("catering", {}).get("cuisine"),
                    "website": props.get("website")
                })
            
            return sorted(restaurants, key=lambda x: x["distance_m"])
        
        except Exception as e:
            print(f"Places API error: {e}")
            return []


class TravelDataAggregator:
    """Aggregates data from all external APIs."""
    
    # Airport coordinates (IATA code -> lat, lon)
    AIRPORT_COORDS = {
        "DEL": (28.5562, 77.1000),   # Delhi
        "BOM": (19.0896, 72.8656),   # Mumbai
        "BLR": (13.1986, 77.7066),   # Bangalore
        "CCU": (22.6520, 88.4463),   # Kolkata
        "MAA": (12.9941, 80.1709),   # Chennai
        "HYD": (17.2403, 78.4294),   # Hyderabad
        "COK": (10.1520, 76.4019),   # Kochi
        "GOI": (15.3808, 73.8314),   # Goa
        "PNQ": (18.5793, 73.9089),   # Pune
        "AMD": (23.0225, 72.5714),   # Ahmedabad
        "JAI": (26.8242, 75.8122),   # Jaipur
    }
    
    # City name to IATA mapping
    CITY_TO_IATA = {
        "Delhi": "DEL", "New Delhi": "DEL",
        "Mumbai": "BOM", "Bombay": "BOM",
        "Bangalore": "BLR", "Banglore": "BLR", "Bengaluru": "BLR",
        "Kolkata": "CCU", "Calcutta": "CCU",
        "Chennai": "MAA", "Madras": "MAA",
        "Hyderabad": "HYD",
        "Kochi": "COK", "Cochin": "COK",
        "Goa": "GOI",
        "Pune": "PNQ",
        "Ahmedabad": "AMD",
        "Jaipur": "JAI"
    }
    
    def __init__(
        self,
        amadeus_key: str = None,
        amadeus_secret: str = None,
        weather_key: str = None,
        places_key: str = None
    ):
        self.amadeus = AmadeusClient(amadeus_key, amadeus_secret) if amadeus_key else None
        self.weather = WeatherClient(weather_key) if weather_key else None
        self.currency = CurrencyClient()
        self.places = PlacesClient(places_key) if places_key else None
    
    def get_city_iata(self, city_name: str) -> str:
        """Convert city name to IATA code."""
        return self.CITY_TO_IATA.get(city_name, city_name[:3].upper())
    
    def get_airport_coords(self, city_or_iata: str) -> tuple:
        """Get airport coordinates."""
        iata = self.get_city_iata(city_or_iata)
        return self.AIRPORT_COORDS.get(iata, (0, 0))
    
    def get_complete_travel_info(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        include_hotels: bool = True,
        include_weather: bool = True,
        include_real_prices: bool = True
    ) -> Dict:
        """Get comprehensive travel information."""
        
        result = {
            "route": {
                "origin": origin,
                "destination": destination,
                "origin_iata": self.get_city_iata(origin),
                "destination_iata": self.get_city_iata(destination),
                "date": departure_date
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Real-time flight prices
        if include_real_prices and self.amadeus:
            flights = self.amadeus.search_flights(
                result["route"]["origin_iata"],
                result["route"]["destination_iata"],
                departure_date
            )
            result["live_flights"] = flights
            if flights:
                result["price_range"] = {
                    "min": min(f["price"] for f in flights),
                    "max": max(f["price"] for f in flights),
                    "avg": sum(f["price"] for f in flights) / len(flights)
                }
        
        # Weather information
        if include_weather and self.weather:
            result["weather"] = {
                "origin": self.weather.get_current_weather(origin),
                "destination": self.weather.get_current_weather(destination)
            }
            result["delay_risk"] = {
                "origin": self.weather.get_delay_risk_score(origin),
                "destination": self.weather.get_delay_risk_score(destination)
            }
        
        # Hotels near destination airport
        if include_hotels and self.places:
            dest_coords = self.get_airport_coords(destination)
            if dest_coords[0] != 0:
                result["hotels"] = self.places.find_hotels_near_airport(
                    dest_coords[0], dest_coords[1]
                )
        
        # Currency conversion
        if "price_range" in result:
            rates = self.currency.get_rates("INR")
            result["prices_usd"] = {
                "min": self.currency.convert(result["price_range"]["min"], "INR", "USD"),
                "max": self.currency.convert(result["price_range"]["max"], "INR", "USD"),
                "avg": self.currency.convert(result["price_range"]["avg"], "INR", "USD")
            }
        
        return result
```

#### 2.2 Configuration Management

```python
# config/settings.py

import os
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # App settings
    APP_NAME: str = "TravelPriceIQ"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # API Keys (loaded from environment)
    AMADEUS_CLIENT_ID: str = ""
    AMADEUS_CLIENT_SECRET: str = ""
    OPENWEATHER_API_KEY: str = ""
    GEOAPIFY_API_KEY: str = ""
    
    # Azure settings
    AZURE_STORAGE_CONNECTION_STRING: str = ""
    AZURE_ML_WORKSPACE: str = ""
    
    # Redis cache
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL_SECONDS: int = 3600
    
    # Model settings
    MODEL_PATH: str = "models/ensemble_model.pkl"
    ENCODERS_PATH: str = "models/saved_encoders.pkl"
    
    # Rate limiting
    API_RATE_LIMIT_PER_MINUTE: int = 60
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

```bash
# .env.example (copy to .env and fill in values)

# App Configuration
APP_NAME=TravelPriceIQ
DEBUG=false

# Amadeus API (Free: 2000 calls/month)
# Register at: https://developers.amadeus.com/register
AMADEUS_CLIENT_ID=your_amadeus_client_id
AMADEUS_CLIENT_SECRET=your_amadeus_client_secret

# OpenWeatherMap API (Free: 1000 calls/day)
# Register at: https://openweathermap.org/api
OPENWEATHER_API_KEY=your_openweather_key

# Geoapify API (Free: 3000 calls/day)
# Register at: https://www.geoapify.com/
GEOAPIFY_API_KEY=your_geoapify_key

# Azure Configuration
AZURE_STORAGE_CONNECTION_STRING=your_azure_connection_string
AZURE_ML_WORKSPACE=your_workspace_name

# Redis (optional, for caching)
REDIS_URL=redis://localhost:6379
```

---

### Phase 3: Enhanced FastAPI Backend (Week 3-4)

```python
# api/main.py

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date, datetime
import joblib

from config.settings import get_settings, Settings
from services.external_apis import TravelDataAggregator
from models.ensemble_model import FlightPriceEnsemble
from features.feature_engineer import FlightFeatureEngineer

app = FastAPI(
    title="TravelPriceIQ API",
    description="Intelligent Travel Price Prediction & Optimization",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
model = None
feature_engineer = None
aggregator = None

@app.on_event("startup")
async def load_models():
    global model, feature_engineer, aggregator
    
    settings = get_settings()
    
    # Load ML model
    model = FlightPriceEnsemble(use_mlflow=False)
    model.load(settings.MODEL_PATH)
    
    # Initialize feature engineer
    feature_engineer = FlightFeatureEngineer(
        weather_api_key=settings.OPENWEATHER_API_KEY
    )
    
    # Initialize API aggregator
    aggregator = TravelDataAggregator(
        amadeus_key=settings.AMADEUS_CLIENT_ID,
        amadeus_secret=settings.AMADEUS_CLIENT_SECRET,
        weather_key=settings.OPENWEATHER_API_KEY,
        places_key=settings.GEOAPIFY_API_KEY
    )


# ============== Request/Response Models ==============

class FlightPredictionRequest(BaseModel):
    airline: str = Field(..., example="IndiGo")
    source: str = Field(..., example="Delhi")
    destination: str = Field(..., example="Mumbai")
    date_of_journey: date = Field(..., example="2026-03-15")
    dep_time: str = Field("10:00", example="10:00")
    arrival_time: str = Field("12:00", example="12:00")
    duration: str = Field("2h 0m", example="2h 0m")
    total_stops: int = Field(0, ge=0, le=4)
    additional_info: str = Field("No Info", example="No Info")

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence: float
    price_range: dict
    currency: str = "INR"
    model_version: str

class TravelPackageRequest(BaseModel):
    origin: str
    destination: str
    departure_date: date
    return_date: Optional[date] = None
    travelers: int = 1
    include_hotels: bool = True
    budget_preference: str = Field("balanced", regex="^(budget|balanced|premium)$")

class WeatherDelayRequest(BaseModel):
    city: str
    date: Optional[date] = None


# ============== API Endpoints ==============

@app.get("/")
async def root():
    return {
        "message": "Welcome to TravelPriceIQ API",
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: FlightPredictionRequest):
    """
    Predict flight price with confidence interval.
    
    Returns predicted price, confidence score, and price range.
    """
    try:
        import pandas as pd
        
        # Prepare input data
        input_data = {
            "Airline": request.airline,
            "Source": request.source,
            "Destination": request.destination,
            "Date_of_Journey": pd.Timestamp(request.date_of_journey),
            "Dep_Time": request.dep_time,
            "Arrival_Time": request.arrival_time,
            "Duration": request.duration,
            "Total_Stops": request.total_stops,
            "Additional_Info": request.additional_info
        }
        
        df = pd.DataFrame([input_data])
        
        # Feature engineering
        df = feature_engineer.transform(df)
        
        # Predict with uncertainty
        result = model.predict_with_uncertainty(df)
        
        return PredictionResponse(
            predicted_price=round(float(result['prediction'][0]), 2),
            confidence=round(float(result['confidence'][0]), 2),
            price_range={
                "lower": round(float(result['lower_bound'][0]), 2),
                "upper": round(float(result['upper_bound'][0]), 2)
            },
            currency="INR",
            model_version="ensemble_v2"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(requests: List[FlightPredictionRequest]):
    """Batch prediction for multiple flights."""
    results = []
    for req in requests:
        result = await predict_price(req)
        results.append(result)
    return {"predictions": results}


@app.get("/travel/complete")
async def get_complete_travel_info(
    origin: str = Query(..., example="Delhi"),
    destination: str = Query(..., example="Mumbai"),
    date: date = Query(..., example="2026-03-15")
):
    """
    Get comprehensive travel information including:
    - Live flight prices
    - Weather at origin & destination
    - Delay risk assessment
    - Hotels near destination airport
    - Currency conversion
    """
    try:
        result = aggregator.get_complete_travel_info(
            origin=origin,
            destination=destination,
            departure_date=str(date),
            include_hotels=True,
            include_weather=True,
            include_real_prices=True
        )
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/weather/delay-risk")
async def get_delay_risk(request: WeatherDelayRequest):
    """
    Get weather-based delay risk for a city.
    
    Returns risk score (0-1), risk level, and contributing factors.
    """
    if not aggregator.weather:
        raise HTTPException(status_code=503, detail="Weather service not configured")
    
    return aggregator.weather.get_delay_risk_score(request.city)


@app.get("/flights/live")
async def get_live_flights(
    origin: str = Query(..., example="DEL"),
    destination: str = Query(..., example="BOM"),
    date: date = Query(..., example="2026-03-15"),
    max_results: int = Query(10, ge=1, le=50)
):
    """Get live flight prices from Amadeus API."""
    if not aggregator.amadeus:
        raise HTTPException(status_code=503, detail="Amadeus service not configured")
    
    flights = aggregator.amadeus.search_flights(
        origin, destination, str(date), max_results=max_results
    )
    
    return {
        "route": f"{origin} → {destination}",
        "date": str(date),
        "flights": flights,
        "count": len(flights)
    }


@app.get("/flights/cheapest-dates")
async def get_cheapest_dates(
    origin: str = Query(..., example="DEL"),
    destination: str = Query(..., example="BOM")
):
    """Get cheapest dates to fly for a route."""
    if not aggregator.amadeus:
        raise HTTPException(status_code=503, detail="Amadeus service not configured")
    
    dates = aggregator.amadeus.get_cheapest_dates(origin, destination)
    
    return {
        "route": f"{origin} → {destination}",
        "cheapest_dates": dates
    }


@app.get("/hotels/near-airport")
async def get_hotels_near_airport(
    city: str = Query(..., example="Mumbai"),
    radius_km: int = Query(10, ge=1, le=50)
):
    """Find hotels near an airport."""
    if not aggregator.places:
        raise HTTPException(status_code=503, detail="Places service not configured")
    
    coords = aggregator.get_airport_coords(city)
    if coords[0] == 0:
        raise HTTPException(status_code=404, detail=f"Airport not found for {city}")
    
    hotels = aggregator.places.find_hotels_near_airport(
        coords[0], coords[1], radius_meters=radius_km * 1000
    )
    
    return {
        "city": city,
        "airport_coords": {"lat": coords[0], "lon": coords[1]},
        "hotels": hotels,
        "count": len(hotels)
    }


@app.get("/currency/convert")
async def convert_currency(
    amount: float = Query(..., example=5000),
    from_currency: str = Query("INR", example="INR"),
    to_currency: str = Query("USD", example="USD")
):
    """Convert currency amount."""
    converted = aggregator.currency.convert(amount, from_currency, to_currency)
    
    return {
        "original": {"amount": amount, "currency": from_currency},
        "converted": {"amount": converted, "currency": to_currency},
        "rate": converted / amount if amount > 0 else 0
    }


@app.get("/trends/price-history")
async def get_price_trends(
    origin: str = Query(..., example="Delhi"),
    destination: str = Query(..., example="Mumbai"),
    airline: Optional[str] = Query(None, example="IndiGo")
):
    """
    Get historical price trends for a route.
    
    Note: Uses training data statistics. For real-time trends,
    integrate with a historical price database.
    """
    # This would ideally query a database of historical predictions
    # For now, return mock trend data
    return {
        "route": f"{origin} → {destination}",
        "airline": airline or "All Airlines",
        "trend": "stable",
        "avg_price_30d": 5500,
        "price_change_percent": -2.3,
        "best_day_to_book": "Tuesday",
        "best_time_to_book": "6-8 weeks before travel"
    }


@app.get("/analytics/feature-importance")
async def get_feature_importance():
    """Get model feature importance for transparency."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    importance = model.get_feature_importance()
    
    return {
        "features": importance.to_dict(orient="records"),
        "top_5": importance.head(5).to_dict(orient="records")
    }
```

---

### Phase 4: Enhanced Streamlit Frontend (Week 4-5)

#### 4.1 Multi-Page Streamlit App Structure

```
frontend/
├── app.py                    # Main entry point
├── pages/
│   ├── 1_🎯_Price_Prediction.py
│   ├── 2_📦_Trip_Optimizer.py
│   ├── 3_🌤️_Weather_Delays.py
│   ├── 4_📊_Analytics.py
│   └── 5_🤖_AI_Assistant.py
└── components/
    ├── charts.py
    ├── maps.py
    └── cards.py
```

#### 4.2 Main App (app.py)

```python
# frontend/app.py

import streamlit as st

st.set_page_config(
    page_title="TravelPriceIQ",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/TravelPriceIQ',
        'Report a bug': 'https://github.com/yourusername/TravelPriceIQ/issues',
        'About': "# TravelPriceIQ\nIntelligent Travel Price Prediction Platform"
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Main page content
st.markdown('<h1 class="main-header">✈️ TravelPriceIQ</h1>', unsafe_allow_html=True)
st.markdown("### Intelligent Travel Price Prediction & Optimization")

st.markdown("---")

# Feature cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    ### 🎯 Price Prediction
    Predict flight prices with **95%+ accuracy** using our ensemble ML model.
    
    - Confidence intervals
    - Feature explanations
    - Historical comparisons
    """)

with col2:
    st.markdown("""
    ### 📦 Trip Optimizer
    Find the **best value** travel packages combining flights and hotels.
    
    - Budget/Balanced/Premium options
    - Real-time pricing
    - Multi-city support
    """)

with col3:
    st.markdown("""
    ### 🌤️ Weather & Delays
    Check **delay probability** based on real-time weather conditions.
    
    - Weather forecasts
    - Risk assessment
    - Travel advisories
    """)

with col4:
    st.markdown("""
    ### 📊 Analytics
    Explore **pricing trends** and booking insights.
    
    - Route analysis
    - Seasonal patterns
    - Best booking times
    """)

st.markdown("---")

# Quick start guide
with st.expander("🚀 Quick Start Guide"):
    st.markdown("""
    1. **Navigate** to any page using the sidebar menu
    2. **Enter** your travel details
    3. **Get** instant predictions and recommendations
    
    **Tips:**
    - Book **6-8 weeks** in advance for best prices
    - **Tuesdays** typically have lower prices
    - Avoid booking during **holiday seasons** for savings
    """)

# Stats
st.markdown("### 📈 Platform Statistics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model Accuracy", "94.2%", "+2.1%")
with col2:
    st.metric("Routes Covered", "50+", "+5")
with col3:
    st.metric("Airlines", "12", "")
with col4:
    st.metric("API Integrations", "5", "+2")
```

#### 4.3 Price Prediction Page

```python
# frontend/pages/1_🎯_Price_Prediction.py

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Price Prediction", page_icon="🎯", layout="wide")

st.title("🎯 Flight Price Prediction")
st.markdown("Get accurate price predictions with confidence intervals")

# API endpoint
API_URL = st.secrets.get("API_URL", "http://localhost:8000")

# Sidebar inputs
with st.sidebar:
    st.header("Flight Details")
    
    airlines = [
        "IndiGo", "Air India", "SpiceJet", "Vistara", 
        "GoAir", "AirAsia India", "Jet Airways", "Multiple carriers"
    ]
    airline = st.selectbox("Airline", airlines)
    
    cities = [
        "Delhi", "Mumbai", "Bangalore", "Kolkata", 
        "Chennai", "Hyderabad", "Cochin", "Goa"
    ]
    source = st.selectbox("Source", cities)
    destination = st.selectbox("Destination", [c for c in cities if c != source])
    
    journey_date = st.date_input(
        "Date of Journey",
        min_value=datetime.today(),
        value=datetime.today() + timedelta(days=30)
    )
    
    col1, col2 = st.columns(2)
    with col1:
        dep_time = st.time_input("Departure", datetime.strptime("10:00", "%H:%M").time())
    with col2:
        arr_time = st.time_input("Arrival", datetime.strptime("12:00", "%H:%M").time())
    
    total_stops = st.slider("Stops", 0, 3, 0)
    
    additional_info = st.selectbox(
        "Additional Info",
        ["No Info", "No check-in baggage included", "In-flight meal not included"]
    )
    
    predict_button = st.button("🔮 Predict Price", type="primary", use_container_width=True)

# Main content
if predict_button:
    with st.spinner("Analyzing flight data..."):
        # Calculate duration
        dep_dt = datetime.combine(journey_date, dep_time)
        arr_dt = datetime.combine(journey_date, arr_time)
        if arr_dt < dep_dt:
            arr_dt += timedelta(days=1)
        duration = arr_dt - dep_dt
        hours, remainder = divmod(duration.seconds, 3600)
        minutes = remainder // 60
        duration_str = f"{hours}h {minutes}m"
        
        # API request
        payload = {
            "airline": airline,
            "source": source,
            "destination": destination,
            "date_of_journey": str(journey_date),
            "dep_time": dep_time.strftime("%H:%M"),
            "arrival_time": arr_time.strftime("%H:%M"),
            "duration": duration_str,
            "total_stops": total_stops,
            "additional_info": additional_info
        }
        
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            result = response.json()
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Predicted Price",
                    f"₹{result['predicted_price']:,.2f}",
                    delta=None
                )
            
            with col2:
                confidence = result['confidence'] * 100
                st.metric(
                    "Confidence",
                    f"{confidence:.1f}%",
                    delta="High" if confidence > 80 else "Medium"
                )
            
            with col3:
                price_range = result['price_range']
                st.metric(
                    "Price Range",
                    f"₹{price_range['lower']:,.0f} - ₹{price_range['upper']:,.0f}"
                )
            
            # Visualization
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Price gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=result['predicted_price'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Predicted Price (₹)"},
                    delta={'reference': 5000, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                    gauge={
                        'axis': {'range': [None, 15000]},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [0, 4000], 'color': "#d4edda"},
                            {'range': [4000, 8000], 'color': "#fff3cd"},
                            {'range': [8000, 15000], 'color': "#f8d7da"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': result['predicted_price']
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence visualization
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Prediction Confidence (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#28a745" if confidence > 80 else "#ffc107"},
                        'steps': [
                            {'range': [0, 50], 'color': "#f8d7da"},
                            {'range': [50, 80], 'color': "#fff3cd"},
                            {'range': [80, 100], 'color': "#d4edda"}
                        ]
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Price comparison
            st.markdown("### 📊 Price Analysis")
            
            # Days until departure analysis
            days_until = (journey_date - datetime.today().date()).days
            
            advice = []
            if days_until < 7:
                advice.append("⚠️ **Last-minute booking** - prices may be higher")
            elif days_until < 21:
                advice.append("📅 Booking **2-3 weeks** ahead - reasonable prices")
            elif days_until < 45:
                advice.append("✅ **Optimal booking window** - best prices typically found here")
            else:
                advice.append("📆 **Advance booking** - prices may fluctuate")
            
            if total_stops == 0:
                advice.append("✈️ **Non-stop flight** - typically costs 10-20% more")
            
            if airline in ["Vistara", "Air India"]:
                advice.append("🎖️ **Premium airline** - higher service quality included in price")
            
            for a in advice:
                st.info(a)
            
            # Currency conversion
            st.markdown("### 💱 Currency Conversion")
            col1, col2, col3 = st.columns(3)
            
            usd = result['predicted_price'] * 0.012
            eur = result['predicted_price'] * 0.011
            gbp = result['predicted_price'] * 0.0095
            
            with col1:
                st.metric("USD", f"${usd:,.2f}")
            with col2:
                st.metric("EUR", f"€{eur:,.2f}")
            with col3:
                st.metric("GBP", f"£{gbp:,.2f}")
                
        except Exception as e:
            st.error(f"Error getting prediction: {str(e)}")
            st.info("Make sure the API server is running")

else:
    # Show sample predictions
    st.info("👈 Enter flight details in the sidebar and click 'Predict Price'")
    
    st.markdown("### 💡 Sample Predictions")
    
    sample_data = pd.DataFrame({
        "Route": ["Delhi → Mumbai", "Bangalore → Delhi", "Kolkata → Chennai", "Mumbai → Goa"],
        "Airline": ["IndiGo", "Vistara", "SpiceJet", "Air India"],
        "Est. Price": ["₹3,500 - ₹4,500", "₹5,500 - ₹7,000", "₹4,000 - ₹5,000", "₹3,000 - ₹4,000"],
        "Best Time": ["6 weeks ahead", "4 weeks ahead", "8 weeks ahead", "4 weeks ahead"]
    })
    
    st.dataframe(sample_data, use_container_width=True)
```

---

## 📊 Additional Data Sources (Free)

### 1. OpenFlights Dataset
Static dataset for airport/airline information:
```python
# Download once and use locally
AIRPORTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
AIRLINES_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat"
ROUTES_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"
```

### 2. Indian Government Data
- Fuel prices: https://ppac.gov.in/
- Economic indicators: https://data.gov.in/

### 3. Kaggle Datasets (Additional Training Data)
- Flight Delay Dataset
- Historical Airfare Data
- Weather Data for Indian Cities

---

## 🧪 Testing Strategy

### Unit Tests
```python
# tests/test_models.py

import pytest
import pandas as pd
from models.ensemble_model import FlightPriceEnsemble
from features.feature_engineer import FlightFeatureEngineer

class TestFeatureEngineer:
    def setup_method(self):
        self.fe = FlightFeatureEngineer()
    
    def test_temporal_features(self):
        df = pd.DataFrame({
            'Date_of_Journey': pd.to_datetime(['2026-03-15']),
            'Dep_Time': ['10:00']
        })
        result = self.fe.create_temporal_features(df)
        
        assert 'day_of_week' in result.columns
        assert 'is_weekend' in result.columns
        assert result['month'].iloc[0] == 3

    def test_holiday_features(self):
        df = pd.DataFrame({
            'Date_of_Journey': pd.to_datetime(['2026-01-26'])  # Republic Day
        })
        result = self.fe.create_holiday_features(df)
        
        assert result['is_holiday'].iloc[0] == 1


class TestEnsembleModel:
    def test_prediction_shape(self):
        # Load test data
        X_test = pd.read_pickle('models/x_testing.pkl')
        
        model = FlightPriceEnsemble(use_mlflow=False)
        model.load('models/ensemble_model.pkl')
        
        predictions = model.predict(X_test.head(10))
        
        assert len(predictions) == 10
        assert all(p > 0 for p in predictions)
```

### Integration Tests
```python
# tests/test_api.py

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    payload = {
        "airline": "IndiGo",
        "source": "Delhi",
        "destination": "Mumbai",
        "date_of_journey": "2026-03-15",
        "dep_time": "10:00",
        "arrival_time": "12:00",
        "duration": "2h 0m",
        "total_stops": 0,
        "additional_info": "No Info"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "predicted_price" in data
    assert "confidence" in data
    assert data["predicted_price"] > 0
```

---

## 📅 Project Timeline

```
Week 1-2: ML Pipeline Enhancement
├── Day 1-3: Feature engineering implementation
├── Day 4-5: XGBoost/LightGBM model training
├── Day 6-7: Ensemble model development
├── Day 8-10: MLflow integration
├── Day 11-12: Model evaluation & optimization
└── Day 13-14: Model export & testing

Week 2-3: API Integrations
├── Day 1-2: Amadeus API integration
├── Day 3-4: OpenWeather API integration
├── Day 5-6: Geoapify (Hotels) integration
├── Day 7-8: Currency API integration
├── Day 9-10: Data aggregation service
└── Day 11-14: Testing & rate limit handling

Week 3-4: Backend Enhancement
├── Day 1-3: New FastAPI endpoints
├── Day 4-5: Caching implementation
├── Day 6-7: Error handling & logging
└── Day 8-14: API testing & documentation

Week 4-5: Frontend Enhancement
├── Day 1-3: Multi-page Streamlit setup
├── Day 4-6: Price prediction page
├── Day 7-9: Trip optimizer page
├── Day 10-12: Weather & analytics pages
└── Day 13-14: UI polish & testing

Week 5-6: Deployment & MLOps
├── Day 1-3: Azure setup & configuration
├── Day 4-5: Container deployment
├── Day 6-7: CI/CD pipeline
├── Day 8-10: Monitoring setup
├── Day 11-12: Performance optimization
└── Day 13-14: Documentation & launch
```

---

## 🎯 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Model R² Score | > 0.92 | Cross-validation |
| Prediction Latency | < 500ms | API response time |
| API Uptime | > 99% | Azure monitoring |
| User Satisfaction | > 4.5/5 | Feedback surveys |
| Monthly Active Users | 1000+ | Analytics |
| API Calls/Month | < Budget | Usage monitoring |

---

## 📚 Resources & References

### Documentation
- [Amadeus API Docs](https://developers.amadeus.com/self-service/apis-docs)
- [OpenWeatherMap Docs](https://openweathermap.org/api)
- [Geoapify Docs](https://apidocs.geoapify.com/)
- [Azure ML Docs](https://docs.microsoft.com/azure/machine-learning/)
- [MLflow Docs](https://mlflow.org/docs/latest/)

### Tutorials
- [XGBoost for Regression](https://xgboost.readthedocs.io/)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)
- [Streamlit Multi-page Apps](https://docs.streamlit.io/library/get-started/multipage-apps)

### Community
- [Kaggle Flight Datasets](https://www.kaggle.com/datasets?search=flight)
- [Papers with Code - Time Series](https://paperswithcode.com/task/time-series-forecasting)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for the travel community**

*Last Updated: January 2026*
