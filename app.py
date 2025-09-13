# app.py
import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timezone
import math

st.set_page_config(page_title="Optimistic Weather", page_icon="🌤️", layout="wide")

# ---------------------------
# Helpers
# ---------------------------

WEATHER_CODE_MAP = {
    0: ("Clear sky", 0),
    1: ("Mainly clear", 1),
    2: ("Partly cloudy", 2),
    3: ("Overcast", 3),
    45: ("Fog", 4), 48: ("Depositing rime fog", 5),
    51: ("Light drizzle", 6), 53: ("Moderate drizzle", 7), 55: ("Dense drizzle", 8),
    56: ("Light freezing drizzle", 9), 57: ("Dense freezing drizzle", 10),
    61: ("Slight rain", 11), 63: ("Moderate rain", 12), 65: ("Heavy rain", 13),
    66: ("Light freezing rain", 14), 67: ("Heavy freezing rain", 15),
    71: ("Slight snow", 16), 73: ("Moderate snow", 17), 75: ("Heavy snow", 18),
    77: ("Snow grains", 19),
    80: ("Rain showers: slight", 20), 81: ("Rain showers: moderate", 21), 82: ("Rain showers: violent", 22),
    85: ("Snow showers: slight", 23), 86: ("Snow showers: heavy", 24),
    95: ("Thunderstorm", 25), 96: ("Thunderstorm with slight hail", 26), 99: ("Thunderstorm with heavy hail", 27),
}

def code_to_text(code):
    """Map numeric weather codes to human-readable text and a rank (lower = nicer)."""
    try:
        if code is None:
            return "Unknown", 99
        if isinstance(code, float) and math.isnan(code):
            return "Unknown", 99
        c = int(code)
    except Exception:
        return "Unknown", 99
    return WEATHER_CODE_MAP.get(c, ("Unknown", 99))

@st.cache_data(show_spinner=False)
def geocode(query: str):
    """Return (lat, lon, name, country, timezone) using Open-Meteo geocoding."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(url, params={"name": query, "count": 1, "language": "en", "format": "json"}, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data.get("results"):
        raise ValueError("Location not found.")
    res = data["results"][0]
    return float(res["latitude"]), float(res["longitude"]), res.get("name", query), res.get("country", ""), res.get("timezone", "UTC")

# Candidate weather models available via Open-Meteo.
CANDIDATE_MODELS = [
    "gfs_seamless",          # NOAA Global Forecast System
    "icon_seamless",         # DWD ICON
    "ecmwf_ifs04",           # ECMWF IFS 0.4°
    "meteofrance_seamless",  # Météo-France
    "gem_global",            # Environment Canada
    "jma_seamless",          # JMA
    "ukmo_seamless",         # UK Met Office global (if available)
]

@st.cache_data(show_spinner=False)
def fetch_openmeteo(lat, lon, tz, models):
    """
    Fetch hourly (today) and daily (next 7 days) forecasts from Open-Meteo for each requested model.
    Returns two DataFrames: hourly_all, daily_all with a 'source' column = model name.
    """
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "precipitation_probability", "weathercode"],
        "daily": ["weathercode", "precipitation_probability_max", "temperature_2m_max", "temperature_2m_min"],
        "timezone": tz,
        "forecast_days": 8,  # today + next 7
    }

    hourly_frames = []
    daily_frames  = []

    for m in models:
        p = dict(params)
        p["models"] = m
        try:
            rr = requests.get(base, params=p, timeout=20)
            rr.raise_for_status()
            dd = rr.json()
        except Exception:
            continue

        # Hourly
        hh = dd.get("hourly", {})
        if hh and "time" in hh:
            hdf = pd.DataFrame({
                "time": pd.to_datetime(hh["time"]),
                "temperature_c": hh.get("temperature_2m"),
                "precip_prob": hh.get("precipitation_probability"),
                "weathercode": hh.get("weathercode"),
            })
            hdf["source"] = m
            # Keep only today's hours (local tz per API param)
            today_str = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
            hdf["date"] = hdf["time"].dt.strftime("%Y-%m-%d")
            hdf = hdf[hdf["date"] == today_str].drop(columns=["date"])
            hourly_frames.append(hdf)

        # Daily (today + next 7)
        ddaily = dd.get("daily", {})
        if ddaily and ddaily.get("time"):
            dti = pd.to_datetime(ddaily["time"])
            ddf = pd.DataFrame({
                "date": dti.date,  # fixed for pandas 2.x
                "wcode_day": ddaily.get("weathercode"),
                "precip_prob_max": ddaily.get("precipitation_probability_max"),
                "tmax_c": ddaily.get("temperature_2m_max"),
                "tmin_c": ddaily.get("temperature_2m_min"),
            })
            ddf["source"] = m
            ddf = ddf.iloc[:8]
            daily_frames.append(ddf)

    hourly_all = pd.concat(hourly_frames, ignore_index=True) if hourly_frames else pd.DataFrame()
    daily_all  = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    return hourly_all, daily_all

def pick_best_per_key(df: pd.DataFrame, key_cols, mode: str, is_hourly: bool):
    """
    For each timestamp (hourly) or date (daily), pick the 'best' or 'worst' row across sources.
    Tie-breakers:
      Optimistic: lowest precip_prob → highest temp → nicest weathercode rank.
      Pessimistic: highest precip_prob → lowest temp → worst weathercode rank.
    """
    if df.empty:
        return df

    out_rows = []
    group_key = "time" if is_hourly else "date"

    for _, g in df.groupby(group_key):
        g = g.copy()
        if is_hourly:
            if mode == "Optimistic":
                g["pp"] = g["precip_prob"].fillna(0)
                g["temp"] = g["temperature_c"]
            else:
                g["pp"] = g["precip_prob"].fillna(100)
                g["temp"] = g["temperature_c"]
            g["wc_rank"] = g["weathercode"].apply(lambda x: code_to_text(x)[1] if pd.notna(x) else 99)
            g = g.sort_values(
                by=["pp", "temp", "wc_rank"],
                ascending=[True, False, True] if mode == "Optimistic" else [False, True, False]
            )
            best = g.iloc[0].to_dict()
            best["condition"] = code_to_text(best.get("weathercode", None))[0]  # safe
            out_rows.append(best)
        else:
            if mode == "Optimistic":
                g["pp"] = g["precip_prob_max"].fillna(0)
                g["temp_key"] = g["tmax_c"]
            else:
                g["pp"] = g["precip_prob_max"].fillna(100)
                g["temp_key"] = g["tmin_c"]
            g["wc_rank"] = g["wcode_day"].apply(lambda x: code_to_text(x)[1] if pd.notna(x) else 99)
            g = g.sort_values(
                by=["pp", "temp_key", "wc_rank"],
                ascending=[True, False, True] if mode == "Optimistic" else [False, True, False]
            )
            best = g.iloc[0].to_dict()
            best["condition"] = code_to_text(best.get("wcode_day", None))[0]  # safe
            out_rows.append(best)

    return pd.DataFrame(out_rows)

def nice_source_name(s: str) -> str:
    mapping = {
        "gfs_seamless": "GFS (NOAA)",
        "icon_seamless": "ICON (DWD)",
        "ecmwf_ifs04": "ECMWF IFS 0.4°",
        "meteofrance_seamless": "Météo-France",
        "gem_global": "GEM (Canada)",
        "jma_seamless": "JMA",
        "ukmo_seamless": "UKMO (Met Office)",
    }
    return mapping.get(s, s)

def style_sources_column(df: pd.DataFrame):
    if "source" in df.columns:
        df = df.copy()
        df["source"] = df["source"].apply(nice_source_name)
    return df

# ---------------------------
# UI
# ---------------------------

st.title("🌤️ Optimistic Weather")
st.caption("Cherry-pick across multiple models for the rosiest (or grumpiest) forecast.")

left, right = st.columns([3, 2])
with left:
    location = st.text_input("Location", value="London")
with right:
    mode = st.radio("Forecast mood", options=["Optimistic", "Pessimistic"], horizontal=True,
                    help="Optimistic = dry, warm, sunny. Pessimistic = wet, cold, gloomy.")

models_selected = st.multiselect(
    "Sources (models)",
    options=CANDIDATE_MODELS,
    default=["gfs_seamless", "icon_seamless", "ecmwf_ifs04", "ukmo_seamless"],
    help="Each model acts like an independent forecast source."
)

if st.button("Get forecast", type="primary"):
    try:
        lat, lon, city, country, tz = geocode(location)
    except Exception as e:
        st.error(f"Could not find that location. {e}")
        st.stop()

    st.success(f"Using **{city}, {country}**  ·  ({lat:.4f}, {lon:.4f})  ·  Time zone: {tz}")

    with st.spinner("Fetching forecasts from multiple sources..."):
        hourly_all, daily_all = fetch_openmeteo(lat, lon, tz, models_selected)

    if hourly_all.empty and daily_all.empty:
        st.warning("No data returned from sources. Try fewer/different models or another location.")
        st.stop()

    # Pretty-up sources
    hourly_all = style_sources_column(hourly_all)
    daily_all = style_sources_column(daily_all)

    # ---------------------------
    # Hourly — Today
    # ---------------------------
    st.subheader("Hourly — Today")
    if not hourly_all.empty:
        hourly_pick = pick_best_per_key(hourly_all, ["time"], mode=mode, is_hourly=True)
        if not hourly_pick.empty:
            show = hourly_pick[["time", "source", "temperature_c", "precip_prob", "condition"]].copy()
            show = show.rename(columns={
                "time": "Time",
                "source": "Chosen Source",
                "temperature_c": "Temp (°C)",
                "precip_prob": "Chance of rain (%)",
                "condition": "Condition"
            }).sort_values("Time")
            st.dataframe(show, use_container_width=True, hide_index=True)
        else:
            st.info("Hourly data wasn’t available for today from the chosen sources.")

        with st.expander("See all sources (hourly)"):
            h_all = hourly_all.copy()
            h_all["condition"] = h_all["weathercode"].apply(lambda x: code_to_text(x)[0] if pd.notna(x) else "—")
            h_all = h_all.rename(columns={
                "time": "Time",
                "source": "Source",
                "temperature_c": "Temp (°C)",
                "precip_prob": "Chance of rain (%)",
                "condition": "Condition"
            }).sort_values(["Time", "Source"])
            st.dataframe(h_all[["Time", "Source", "Temp (°C)", "Chance of rain (%)", "Condition"]],
                         use_container_width=True, hide_index=True)
    else:
        st.info("No hourly data for today returned.")

    # ---------------------------
    # Daily — Next 7 Days
    # ---------------------------
    st.subheader("Daily — Next 7 Days")
    if not daily_all.empty:
        today = datetime.now().date()
        daily_all_future = daily_all[daily_all["date"] >= today]

        daily_pick = pick_best_per_key(daily_all_future, ["date"], mode=mode, is_hourly=False)
        if not daily_pick.empty:
            dshow = daily_pick[["date", "source", "tmax_c", "tmin_c", "precip_prob_max", "condition"]].copy()
            dshow = dshow.rename(columns={
                "date": "Date",
                "source": "Chosen Source",
                "tmax_c": "High (°C)",
                "tmin_c": "Low (°C)",
                "precip_prob_max": "Max chance of rain (%)",
                "condition": "Condition"
            }).sort_values("Date")
            dshow = dshow.head(8)  # safety cap
            st.dataframe(dshow, use_container_width=True, hide_index=True)
        else:
            st.info("No daily data available for the selected sources.")

        with st.expander("See all sources (daily)"):
            d_all = daily_all.copy()
            d_all["condition"] = d_all["wcode_day"].apply(lambda x: code_to_text(x)[0] if pd.notna(x) else "—")
            d_all = d_all.rename(columns={
                "date": "Date",
                "source": "Source",
                "tmax_c": "High (°C)",
                "tmin_c": "Low (°C)",
                "precip_prob_max": "Max chance of rain (%)",
                "condition": "Condition"
            }).sort_values(["Date", "Source"])
            st.dataframe(d_all[["Date", "Source", "High (°C)", "Low (°C)", "Max chance of rain (%)", "Condition"]],
                         use_container_width=True, hide_index=True)
    else:
        st.info("No daily data returned.")

    st.caption("Data via Open-Meteo (multiple numerical weather prediction models). Temperatures in Celsius.")

else:
    st.info("Enter a location, choose your mood, and click **Get forecast**. 😉")
