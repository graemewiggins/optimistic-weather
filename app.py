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
    "gfs_seamless",
    "icon_seamless",
    "ecmwf_ifs04",
    "meteofrance_seamless",
    "gem_global",
    "jma_seamless",
    "ukmo_seamless",
]

@st.cache_data(show_spinner=False)
def fetch_openmeteo(lat, lon, tz, models):
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "precipitation_probability", "weathercode"],
        "daily": ["weathercode", "precipitation_probability_max", "temperature_2m_max", "temperature_2m_min"],
        "timezone": tz,
        "forecast_days": 8,
    }

    hourly_frames, daily_frames = [], []

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
            today_str = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
            hdf["date"] = hdf["time"].dt.strftime("%Y-%m-%d")
            hdf = hdf[hdf["date"] == today_str].drop(columns=["date"])
            hourly_frames.append(hdf)

        # Daily
        ddaily = dd.get("daily", {})
        if ddaily and ddaily.get("time"):
            dti = pd.to_datetime(ddaily["time"])
            ddf = pd.DataFrame({
                "date": dti.date,
                "wcode_day": ddaily.get("weathercode"),
                "precip_prob_max": ddaily.get("precipitation_probability_max"),
                "tmax_c": ddaily.get("temperature_2m_max"),
                "tmin_c": ddaily.get("temperature_2m_min"),
            })
            ddf["source"] = m
            ddf = ddf.iloc[:8]
            daily_frames.append(ddf)

    return (
        pd.concat(hourly_frames, ignore_index=True) if hourly_frames else pd.DataFrame(),
        pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(),
    )

def independent_metric_picks(df: pd.DataFrame, mode: str, is_hourly: bool):
    """Select independent best/worst per metric."""
    if df.empty:
        return pd.DataFrame()

    key = "time" if is_hourly else "date"
    rows = []

    for k, g in df.groupby(key):
        g = g.copy()
        if is_hourly:
            # temp
            if mode == "Optimistic":
                temp_row = g.loc[g["temperature_c"].idxmax()] if g["temperature_c"].notna().any() else None
            else:
                temp_row = g.loc[g["temperature_c"].idxmin()] if g["temperature_c"].notna().any() else None
            # precip
            if mode == "Optimistic":
                pp_row = g.loc[g["precip_prob"].fillna(9999).idxmin()] if g["precip_prob"].notna().any() else None
            else:
                pp_row = g.loc[g["precip_prob"].fillna(-1).idxmax()] if g["precip_prob"].notna().any() else None
            # condition
            g["wc_rank"] = g["weathercode"].apply(lambda x: code_to_text(x)[1] if pd.notna(x) else 99)
            cond_row = g.loc[g["wc_rank"].idxmin()] if mode == "Optimistic" else g.loc[g["wc_rank"].idxmax()]

            rows.append({
                "Time": k,
                "Temp (°C)": None if temp_row is None else temp_row.get("temperature_c"),
                "Chance of rain (%)": None if pp_row is None else pp_row.get("precip_prob"),
                "Condition": None if cond_row is None else code_to_text(cond_row.get("weathercode", None))[0],
            })
        else:
            # temp
            if mode == "Optimistic":
                temp_row = g.loc[g["tmax_c"].idxmax()] if g["tmax_c"].notna().any() else None
                temp_val = None if temp_row is None else temp_row.get("tmax_c")
            else:
                temp_row = g.loc[g["tmin_c"].idxmin()] if g["tmin_c"].notna().any() else None
                temp_val = None if temp_row is None else temp_row.get("tmin_c")
            # precip
            if mode == "Optimistic":
                pp_row = g.loc[g["precip_prob_max"].fillna(9999).idxmin()] if g["precip_prob_max"].notna().any() else None
            else:
                pp_row = g.loc[g["precip_prob_max"].fillna(-1).idxmax()] if g["precip_prob_max"].notna().any() else None
            # condition
            g["wc_rank"] = g["wcode_day"].apply(lambda x: code_to_text(x)[1] if pd.notna(x) else 99)
            cond_row = g.loc[g["wc_rank"].idxmin()] if mode == "Optimistic" else g.loc[g["wc_rank"].idxmax()]

            rows.append({
                "Date": k,
                "Temp (°C)": temp_val,
                "Chance of rain (%)": None if pp_row is None else pp_row.get("precip_prob_max"),
                "Condition": None if cond_row is None else code_to_text(cond_row.get("wcode_day", None))[0],
            })

    return pd.DataFrame(rows)

def side_by_side(df: pd.DataFrame, is_hourly: bool):
    """Produce side-by-side optimistic vs pessimistic table."""
    opt = independent_metric_picks(df, "Optimistic", is_hourly)
    pes = independent_metric_picks(df, "Pessimistic", is_hourly)
    key = "Time" if is_hourly else "Date"
    merged = opt.merge(pes, on=key, suffixes=(" (Optimistic)", " (Pessimistic)"))
    # rename columns
    merged = merged.rename(columns={
        "Temp (°C) (Optimistic)": "Optimistic Temp",
        "Temp (°C) (Pessimistic)": "Pessimistic Temp",
        "Chance of rain (%) (Optimistic)": "Optimistic Chance of rain",
        "Chance of rain (%) (Pessimistic)": "Pessimistic Chance of rain",
        "Condition (Optimistic)": "Optimistic Condition",
        "Condition (Pessimistic)": "Pessimistic Condition",
    })
    return merged

# ---------------------------
# UI
# ---------------------------

st.title("🌤️ Optimistic Weather")
st.caption("Cherry-pick forecasts across models: optimistic, pessimistic, or both side by side.")

left, right = st.columns([3, 2])
with left:
    location = st.text_input("Location", value="London")
with right:
    mode = st.radio(
        "Forecast mode",
        options=["Optimistic", "Pessimistic", "Side by side"],
        horizontal=True,
    )

models_selected = st.multiselect(
    "Sources (models)",
    options=CANDIDATE_MODELS,
    default=["gfs_seamless", "icon_seamless", "ecmwf_ifs04", "ukmo_seamless"],
)

if st.button("Get forecast", type="primary"):
    try:
        lat, lon, city, country, tz = geocode(location)
    except Exception as e:
        st.error(f"Could not find that location. {e}")
        st.stop()

    st.success(f"Using **{city}, {country}**  ·  ({lat:.4f}, {lon:.4f})  ·  Time zone: {tz}")

    with st.spinner("Fetching forecasts..."):
        hourly_all, daily_all = fetch_openmeteo(lat, lon, tz, models_selected)

    if hourly_all.empty and daily_all.empty:
        st.warning("No data returned. Try fewer/different models or another location.")
        st.stop()

    # ---------------------------
    # Hourly
    # ---------------------------
    st.subheader("Hourly — Today")
    if not hourly_all.empty:
        if mode in ["Optimistic", "Pessimistic"]:
            hourly_ind = independent_metric_picks(hourly_all, mode=mode, is_hourly=True)
            st.dataframe(hourly_ind.sort_values("Time"), use_container_width=True, hide_index=True)
        else:
            hourly_ss = side_by_side(hourly_all, is_hourly=True)
            st.dataframe(hourly_ss.sort_values("Time"), use_container_width=True, hide_index=True)
    else:
        st.info("No hourly data for today returned.")

    # ---------------------------
    # Daily
    # ---------------------------
    st.subheader("Daily — Next 7 Days")
    if not daily_all.empty:
        today = datetime.now().date()
        daily_all_future = daily_all[daily_all["date"] >= today]
        if mode in ["Optimistic", "Pessimistic"]:
            daily_ind = independent_metric_picks(daily_all_future, mode=mode, is_hourly=False)
            st.dataframe(daily_ind.sort_values("Date").head(8), use_container_width=True, hide_index=True)
        else:
            daily_ss = side_by_side(daily_all_future, is_hourly=False)
            st.dataframe(daily_ss.sort_values("Date").head(8), use_container_width=True, hide_index=True)
    else:
        st.info("No daily data returned.")

    st.caption("Data via Open-Meteo. Temperatures in Celsius.")

else:
    st.info("Enter a location, choose a forecast mode, and click **Get forecast**. 😉")
