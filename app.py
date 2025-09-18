import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import numpy as np

# Set page config
st.set_page_config(
    page_title="Optimistic Weather",
    page_icon="🌤️",
    layout="wide"
)

# Weather condition mappings for optimization scoring
CONDITION_SCORES = {
    # Higher scores = more optimistic (sunny/clear)
    'clear': 10, 'sunny': 10, 'fair': 9, 'clear sky': 10,
    'partly cloudy': 7, 'partly sunny': 7, 'mostly sunny': 8,
    'cloudy': 5, 'overcast': 4, 'mostly cloudy': 4, 'broken clouds': 5,
    'scattered clouds': 6, 'few clouds': 7,
    'light rain': 3, 'rain': 2, 'heavy rain': 1, 'moderate rain': 2,
    'drizzle': 3, 'showers': 2, 'thunderstorms': 1, 'thunderstorm': 1,
    'snow': 2, 'fog': 4, 'mist': 4, 'haze': 5,
    'light snow': 3, 'heavy snow': 1, 'sleet': 2
}

# WMO Weather codes for OpenMeteo
WMO_CODES = {
    0: ('Clear sky', 10),
    1: ('Mainly clear', 9),
    2: ('Partly cloudy', 7),
    3: ('Overcast', 4),
    45: ('Fog', 4),
    48: ('Depositing rime fog', 3),
    51: ('Light drizzle', 3),
    53: ('Moderate drizzle', 2),
    55: ('Dense drizzle', 2),
    56: ('Light freezing drizzle', 2),
    57: ('Dense freezing drizzle', 1),
    61: ('Slight rain', 3),
    63: ('Moderate rain', 2),
    65: ('Heavy rain', 1),
    66: ('Light freezing rain', 2),
    67: ('Heavy freezing rain', 1),
    71: ('Slight snow fall', 3),
    73: ('Moderate snow fall', 2),
    75: ('Heavy snow fall', 1),
    77: ('Snow grains', 2),
    80: ('Slight rain showers', 3),
    81: ('Moderate rain showers', 2),
    82: ('Violent rain showers', 1),
    85: ('Slight snow showers', 3),
    86: ('Heavy snow showers', 1),
    95: ('Thunderstorm', 1),
    96: ('Thunderstorm with slight hail', 1),
    99: ('Thunderstorm with heavy hail', 1)
}

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_condition_score(condition):
    """Convert weather condition to optimism score"""
    if not condition:
        return 5
    condition_lower = str(condition).lower()
    for key, score in CONDITION_SCORES.items():
        if key in condition_lower:
            return score
    return 5  # Default neutral score

def get_location_coords(location_name):
    """Get coordinates for a location using OpenMeteo's geocoding API"""
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={location_name}&count=1&language=en&format=json"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data.get('results'):
            result = data['results'][0]
            return result['latitude'], result['longitude'], result['name']
        return None, None, None
    except Exception as e:
        st.error(f"Error getting location coordinates: {e}")
        return None, None, None

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_openmeteo_data(lat, lon):
    """Fetch data from Open-Meteo (free, no API key required)"""
    try:
        # Get current weather and 7-day forecast with hourly data
        url = (f"https://api.open-meteo.com/v1/forecast?"
               f"latitude={lat}&longitude={lon}"
               f"&hourly=temperature_2m,precipitation_probability,weather_code"
               f"&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
               f"&timezone=auto&forecast_days=7")
        
        response = requests.get(url, timeout=15)
        data = response.json()
        
        forecasts = []
        
        # Process hourly data (next 24 hours for today + tomorrow)
        current_time = datetime.now()
        for i, time_str in enumerate(data['hourly']['time'][:48]):  # Next 48 hours
            dt = datetime.fromisoformat(time_str.replace('T', ' '))
            if dt >= current_time:  # Only future hours
                weather_code = data['hourly']['weather_code'][i]
                condition, condition_score = WMO_CODES.get(weather_code, ('Unknown', 5))
                
                forecasts.append({
                    'source': 'Open-Meteo',
                    'datetime': dt,
                    'date': dt.date(),
                    'hour': dt.hour,
                    'temperature': data['hourly']['temperature_2m'][i],
                    'rain_chance': data['hourly']['precipitation_probability'][i] or 0,
                    'condition': condition,
                    'condition_score': condition_score
                })
        
        # Process daily data
        for i, date_str in enumerate(data['daily']['time']):
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            weather_code = data['daily']['weather_code'][i]
            condition, condition_score = WMO_CODES.get(weather_code, ('Unknown', 5))
            
            # Use average of max and min temperature
            avg_temp = (data['daily']['temperature_2m_max'][i] + data['daily']['temperature_2m_min'][i]) / 2
            
            forecasts.append({
                'source': 'Open-Meteo',
                'datetime': datetime.combine(date_obj, datetime.min.time()),
                'date': date_obj,
                'hour': None,
                'temperature': avg_temp,
                'rain_chance': data['daily']['precipitation_probability_max'][i] or 0,
                'condition': condition,
                'condition_score': condition_score
            })
        
        return forecasts
    except Exception as e:
        st.error(f"Error fetching Open-Meteo data: {e}")
        return []

@st.cache_data(ttl=1800)
def get_openweathermap_data(api_key, lat, lon):
    """Fetch data from OpenWeatherMap"""
    if not api_key:
        return []
    
    try:
        # Current + 5-day forecast
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            st.warning(f"OpenWeatherMap API error: {response.status_code}")
            return []
            
        data = response.json()
        
        forecasts = []
        for item in data['list']:
            dt = datetime.fromtimestamp(item['dt'])
            forecasts.append({
                'source': 'OpenWeatherMap',
                'datetime': dt,
                'date': dt.date(),
                'hour': dt.hour,
                'temperature': item['main']['temp'],
                'rain_chance': item.get('pop', 0) * 100,  # Probability of precipitation
                'condition': item['weather'][0]['description'],
                'condition_score': get_condition_score(item['weather'][0]['description'])
            })
        return forecasts
    except Exception as e:
        st.warning(f"Error fetching OpenWeatherMap data: {e}")
        return []

@st.cache_data(ttl=1800)
def get_weatherapi_data(api_key, location):
    """Fetch data from WeatherAPI"""
    if not api_key:
        return []
    
    try:
        # 7-day forecast with hourly data
        url = f"https://api.weatherapi.com/v1/forecast.json?key={api_key}&q={location}&days=7&aqi=no&alerts=no"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            st.warning(f"WeatherAPI error: {response.status_code}")
            return []
            
        data = response.json()
        
        forecasts = []
        for day in data['forecast']['forecastday']:
            date_obj = datetime.strptime(day['date'], '%Y-%m-%d').date()
            
            # Add daily forecast
            forecasts.append({
                'source': 'WeatherAPI',
                'datetime': datetime.combine(date_obj, datetime.min.time()),
                'date': date_obj,
                'hour': None,  # Daily forecast
                'temperature': day['day']['avgtemp_c'],
                'rain_chance': day['day']['daily_chance_of_rain'],
                'condition': day['day']['condition']['text'],
                'condition_score': get_condition_score(day['day']['condition']['text'])
            })
            
            # Add hourly forecasts for today and tomorrow
            if date_obj <= datetime.now().date() + timedelta(days=1):
                for hour in day['hour']:
                    dt = datetime.strptime(hour['time'], '%Y-%m-%d %H:%M')
                    if dt >= datetime.now():  # Only future hours
                        forecasts.append({
                            'source': 'WeatherAPI',
                            'datetime': dt,
                            'date': dt.date(),
                            'hour': dt.hour,
                            'temperature': hour['temp_c'],
                            'rain_chance': hour['chance_of_rain'],
                            'condition': hour['condition']['text'],
                            'condition_score': get_condition_score(hour['condition']['text'])
                        })
        return forecasts
    except Exception as e:
        st.warning(f"Error fetching WeatherAPI data: {e}")
        return []

def get_openmeteo_ensemble_data(lat, lon):
    """Fetch ensemble forecast data from Open-Meteo for additional variety"""
    try:
        # Use ECMWF model for different perspective
        url = (f"https://api.open-meteo.com/v1/ecmwf?"
               f"latitude={lat}&longitude={lon}"
               f"&hourly=temperature_2m,precipitation_probability"
               f"&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max"
               f"&timezone=auto&forecast_days=7")
        
        response = requests.get(url, timeout=15)
        data = response.json()
        
        forecasts = []
        
        # Process hourly data (next 24 hours)
        current_time = datetime.now()
        for i, time_str in enumerate(data['hourly']['time'][:24]):
            dt = datetime.fromisoformat(time_str.replace('T', ' '))
            if dt >= current_time:
                # Estimate condition based on precipitation probability
                rain_chance = data['hourly']['precipitation_probability'][i] or 0
                if rain_chance < 20:
                    condition, condition_score = "Mostly sunny", 8
                elif rain_chance < 40:
                    condition, condition_score = "Partly cloudy", 7
                elif rain_chance < 60:
                    condition, condition_score = "Cloudy", 5
                elif rain_chance < 80:
                    condition, condition_score = "Light rain", 3
                else:
                    condition, condition_score = "Rain", 2
                
                forecasts.append({
                    'source': 'Open-Meteo ECMWF',
                    'datetime': dt,
                    'date': dt.date(),
                    'hour': dt.hour,
                    'temperature': data['hourly']['temperature_2m'][i],
                    'rain_chance': rain_chance,
                    'condition': condition,
                    'condition_score': condition_score
                })
        
        # Process daily data
        for i, date_str in enumerate(data['daily']['time']):
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            rain_chance = data['daily']['precipitation_probability_max'][i] or 0
            
            # Estimate condition based on precipitation probability
            if rain_chance < 20:
                condition, condition_score = "Mostly sunny", 8
            elif rain_chance < 40:
                condition, condition_score = "Partly cloudy", 7
            elif rain_chance < 60:
                condition, condition_score = "Cloudy", 5
            elif rain_chance < 80:
                condition, condition_score = "Light rain", 3
            else:
                condition, condition_score = "Rain", 2
            
            avg_temp = (data['daily']['temperature_2m_max'][i] + data['daily']['temperature_2m_min'][i]) / 2
            
            forecasts.append({
                'source': 'Open-Meteo ECMWF',
                'datetime': datetime.combine(date_obj, datetime.min.time()),
                'date': date_obj,
                'hour': None,
                'temperature': avg_temp,
                'rain_chance': rain_chance,
                'condition': condition,
                'condition_score': condition_score
            })
        
        return forecasts
    except Exception as e:
        # Don't show error for this optional data source
        return []

def calculate_optimism_score(row, perspective):
    """Calculate overall optimism score based on temperature, rain chance, and condition"""
    # Temperature scoring (5-30°C range, with 20°C being optimal)
    temp = row['temperature']
    if temp >= 20 and temp <= 25:
        temp_score = 10
    elif temp >= 15 and temp <= 30:
        temp_score = 8
    elif temp >= 10 and temp <= 35:
        temp_score = 6
    elif temp >= 5:
        temp_score = 4
    else:
        temp_score = 2
    
    rain_score = (100 - row['rain_chance']) / 10  # 0% rain = 10, 100% rain = 0
    condition_score = row['condition_score']
    
    # Weighted average (condition is most important for optimism)
    overall_score = (condition_score * 0.5 + rain_score * 0.3 + temp_score * 0.2)
    
    if perspective == "pessimistic":
        overall_score = 10 - overall_score  # Invert for pessimistic view
    
    return overall_score

def find_optimal_forecasts(df, perspective):
    """Find most optimistic or pessimistic forecasts"""
    if df.empty:
        return df
        
    df = df.copy()
    df['optimism_score'] = df.apply(lambda row: calculate_optimism_score(row, perspective), axis=1)
    
    # Group by date and hour, then find best/worst forecast for each time period
    if perspective == "optimistic":
        optimal = df.loc[df.groupby(['date', 'hour'])['optimism_score'].idxmax()]
    else:
        optimal = df.loc[df.groupby(['date', 'hour'])['optimism_score'].idxmin()]
    
    return optimal.sort_values(['date', 'hour'])

def main():
    st.title("🌤️ Optimistic Weather")
    st.subheader("Choose your weather reality - now with multiple data sources!")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("🎯 Perspective")
        
        perspective = st.radio(
            "Choose your weather outlook:",
            ["optimistic", "pessimistic"],
            format_func=lambda x: "☀️ Optimistic (sunny & dry)" if x == "optimistic" else "🌧️ Pessimistic (wet & cold)"
        )
        
        st.header("📍 Location")
        location = st.text_input("Enter city or location", value="London", help="e.g., London, New York, Tokyo")
        
        st.header("🔑 API Keys (Optional)")
        st.caption("Open-Meteo works without API keys! Add these for more data sources:")
        
        with st.expander("Add API Keys"):
            openweather_key = st.text_input("OpenWeatherMap API Key", type="password", 
                                          help="Get free key at openweathermap.org")
            weatherapi_key = st.text_input("WeatherAPI Key", type="password",
                                         help="Get free key at weatherapi.com")
        
        st.header("ℹ️ Data Sources")
        st.info("🆓 **Open-Meteo**: Always available (no API key needed)\n\n"
                "🔑 **OpenWeatherMap**: Requires free API key\n\n" 
                "🔑 **WeatherAPI**: Requires free API key")
    
    # Get location coordinates
    if location:
        with st.spinner("Getting location coordinates..."):
            lat, lon, location_name = get_location_coords(location)
        
        if lat is None:
            st.error(f"Could not find coordinates for '{location}'. Please try a different location.")
            return
        
        st.success(f"📍 Found: {location_name} ({lat:.2f}, {lon:.2f})")
    else:
        st.warning("Please enter a location to get weather forecasts.")
        return
    
    # Fetch weather data from all sources
    with st.spinner("Fetching weather forecasts from multiple sources..."):
        all_forecasts = []
        data_sources_used = []
        
        # Open-Meteo (always available)
        openmeteo_data = get_openmeteo_data(lat, lon)
        if openmeteo_data:
            all_forecasts.extend(openmeteo_data)
            data_sources_used.append("Open-Meteo")
        
        # Open-Meteo ECMWF model (additional perspective)
        ecmwf_data = get_openmeteo_ensemble_data(lat, lon)
        if ecmwf_data:
            all_forecasts.extend(ecmwf_data)
            data_sources_used.append("Open-Meteo ECMWF")
        
        # OpenWeatherMap (if API key provided)
        if openweather_key:
            owm_data = get_openweathermap_data(openweather_key, lat, lon)
            if owm_data:
                all_forecasts.extend(owm_data)
                data_sources_used.append("OpenWeatherMap")
        
        # WeatherAPI (if API key provided)
        if weatherapi_key:
            wa_data = get_weatherapi_data(weatherapi_key, location)
            if wa_data:
                all_forecasts.extend(wa_data)
                data_sources_used.append("WeatherAPI")
    
    if not all_forecasts:
        st.error("No weather data available. Please check your internet connection.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_forecasts)
    
    # Display data sources summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Data Sources", len(data_sources_used))
    with col2:
        st.metric("🔢 Total Forecasts", len(df))
    with col3:
        st.metric("⏱️ Update", "Live")
    
    st.caption(f"Active sources: {', '.join(data_sources_used)}")
    
    # Find optimal forecasts
    optimal_forecasts = find_optimal_forecasts(df, perspective)
    
    # Display results
    st.header(f"{'☀️ Most Optimistic' if perspective == 'optimistic' else '🌧️ Most Pessimistic'} Forecasts")
    st.caption("Showing the best/worst forecast from all available sources for each time period")
    
    # Today's hourly forecast
    today = datetime.now().date()
    today_hourly = optimal_forecasts[
        (optimal_forecasts['date'] == today) & 
        (optimal_forecasts['hour'].notna()) &
        (optimal_forecasts['hour'].notnull())
    ].copy()
    
    if not today_hourly.empty:
        st.subheader(f"🕐 Today ({today.strftime('%A, %B %d')}) - Hour by Hour")
        
        # Display in rows of 4 columns
        hourly_list = today_hourly.to_dict('records')
        for i in range(0, len(hourly_list), 4):
            cols = st.columns(4)
            for j, row in enumerate(hourly_list[i:i+4]):
                if j < len(cols):
                    with cols[j]:
                        # Handle potential NaN values in hour
                        hour_val = row.get('hour')
                        if pd.isna(hour_val) or hour_val is None:
                            time_str = "All Day"
                        else:
                            time_str = f"{int(hour_val):02d}:00"
                        
                        temp_str = f"{row['temperature']:.1f}°C"
                        rain_str = f"{row['rain_chance']:.0f}%"
                        
                        # Emoji based on condition
                        if row['condition_score'] >= 8:
                            emoji = "☀️"
                        elif row['condition_score'] >= 6:
                            emoji = "⛅"
                        elif row['condition_score'] >= 4:
                            emoji = "☁️"
                        else:
                            emoji = "🌧️"
                        
                        st.metric(
                            label=f"{emoji} {time_str}",
                            value=temp_str,
                            delta=f"💧 {rain_str}"
                        )
                        st.caption(f"{row['condition']}")
                        st.caption(f"📡 {row['source']}")
    else:
        st.info("No more hourly forecasts available for today.")
    
    # 7-day forecast
    st.subheader("📅 7-Day Forecast")
    
    daily_forecasts = optimal_forecasts[
        (optimal_forecasts['hour'].isna()) | 
        (optimal_forecasts['hour'].isnull())
    ].copy()
    
    if not daily_forecasts.empty:
        for _, row in daily_forecasts.iterrows():
            date_str = row['date'].strftime('%A, %B %d')
            temp_str = f"{row['temperature']:.1f}°C"
            rain_str = f"{row['rain_chance']:.0f}% chance of rain"
            
            # Emoji based on condition
            if row['condition_score'] >= 8:
                emoji = "☀️"
            elif row['condition_score'] >= 6:
                emoji = "⛅"
            elif row['condition_score'] >= 4:
                emoji = "☁️"
            else:
                emoji = "🌧️"
            
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                
                with col1:
                    st.write(f"{emoji} **{date_str}**")
                with col2:
                    st.write(f"🌡️ {temp_str}")
                with col3:
                    st.write(f"🌧️ {rain_str}")
                with col4:
                    st.write(f"📡 {row['source']}")
                
                st.caption(f"Condition: {row['condition']}")
                st.divider()
    
    # Detailed data exploration
    with st.expander("🔍 Explore All Forecast Data"):
        st.subheader("Raw Data from All Sources")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            selected_sources = st.multiselect("Filter by Source", 
                                            options=df['source'].unique(),
                                            default=df['source'].unique())
        with col2:
            show_hourly = st.checkbox("Include Hourly Data", value=True)
        
        # Filter dataframe
        filtered_df = df[df['source'].isin(selected_sources)]
        if not show_hourly:
            filtered_df = filtered_df[filtered_df['hour'].isna()]
        
        # Display filtered data
        display_df = filtered_df[['source', 'datetime', 'temperature', 'rain_chance', 'condition']].copy()
        display_df['temperature'] = display_df['temperature'].round(1)
        display_df['rain_chance'] = display_df['rain_chance'].round(0)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Summary statistics
        st.subheader("📈 Forecast Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_temp = filtered_df['temperature'].mean()
            st.metric("🌡️ Average Temperature", f"{avg_temp:.1f}°C")
        
        with col2:
            avg_rain = filtered_df['rain_chance'].mean()
            st.metric("🌧️ Average Rain Chance", f"{avg_rain:.0f}%")
        
        with col3:
            most_common_condition = filtered_df['condition'].mode().iloc[0] if not filtered_df.empty else "N/A"
            st.metric("☁️ Most Common Condition", most_common_condition)

    # Footer
    st.divider()
    st.caption("🌍 Powered by Open-Meteo, OpenWeatherMap, and WeatherAPI | Built with Streamlit")
    st.caption("💡 Tip: Add API keys in the sidebar for more comprehensive forecasts!")

if __name__ == "__main__":
    main()
