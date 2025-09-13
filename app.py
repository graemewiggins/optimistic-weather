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
    For each timestamp (hourly) or date (daily), pick the single 'best' or 'worst' row across sources
    using combined rules. (Kept for reference/expanded view.)
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
            best["condition"] = code_to_text(best.get("weathercode", None))[0]
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
            best["condition"] = code_to_text(best.get("wcode_day", None))[0]
            out_rows.append(best)

    return pd.DataFrame(out_rows)

def independent_metric_picks(df: pd.DataFrame, mode: str, is_hourly: bool):
    """
    For each time/date group, independently select:
      - Temp (max for optimistic, min for pessimistic) + its source
      - Chance of rain (min for optimistic, max for pessimistic) + its source
      - Condition (best rank for optimistic, worst rank for pessimistic) + its source
    Returns a tidy DataFrame with one row per time/date and value+source columns.
    """
    if df.empty:
        return pd.DataFrame()

    key = "time" if is_hourly else "date"
    rows = []

    for k, g in df.groupby(key):
        g = g.copy()

        if is_hourly:
            # temp pick
            temp_col = "temperature_c"
            if mode == "Optimistic":
                temp_row = g.loc[g[temp_col].idxmax()] if g[temp_col].notna().any() else None
            else:
                temp_row = g.loc[g[temp_col].idxmin()] if g[temp_col].notna().any() else None

            # precip pick
            pp_col = "precip_prob"
            if mode == "Optimistic":
                pp_row = g.loc[g[pp_col].fillna(9999).idxmin()] if g[pp_col].notna().any() else None
            else:
                pp_row = g.loc[g[pp_col].fillna(-1).idxmax()] if g[pp_col].notna().any() else None

            # condition pick
            g["wc_rank"] = g["weathercode"].apply(lambda x: code_to_text(x)[1] if pd.notna(x) else 99)
            if mode == "Optimistic":
                cond_row = g.loc[g["wc_rank"].idxmin()] if g["wc_rank"].notna().any() else None
            else:
                cond_row = g.loc[g["wc_rank"].idxmax()] if g["wc_rank"].notna().any() else None

            rows.append({
                "Time": k,
                "Temp (°C)": None if temp_row is None else temp_row.get("temperature_c"),
                "Chance of rain (%)": None if pp_row is None else pp_row.get("precip_prob"),
                "Condition": None if cond_row is None else code_to_text(cond_row.get("weathercode", None))[0],
                "Temp Source": None if temp_row is None else temp_row.get("source"),
                "Chance of rain Source": None if pp_row is None else pp_row.get("source"),
                "Condition Source": None if cond_row is None else cond_row.get("source"),
            })

        else:
            # daily
            # temp pick uses tmax for optimistic, tmin for pessimistic
            if mode == "Optimistic":
                temp_col = "tmax_c"
                temp_row = g.loc[g[temp_col].idxmax()] if g[temp_col].notna().any() else None
            else:
                temp_col = "tmin_c"
                temp_row = g.loc[g[temp_col].idxmin()] if g[temp_col].notna().any() else None

            # precip pick
            pp_col = "precip_prob_max"
            if mode == "Optimistic":
                pp_row = g.loc[g[pp_col].fillna(9999).idxmin()] if g[pp_col].notna().any() else None
            else:
                pp_row = g.loc[g[pp_col].fillna(-1).idxmax()] if g[pp_col].notna().any() else None

            # condition pick
            g["wc_rank"] = g["wcode_day"].apply(lambda x: code_to_text(x)[1] if pd.notna(x) else 99)
            if mode == "Optimistic":
                cond_row = g.loc[g["wc_rank"].idxmin()] if g["wc_rank"].notna().any() else None
            else:
                cond_row = g.loc[g["wc_rank"].idxmax()] if g["wc_rank"].notna().any() else None

            # Display a single "Temp (°C)" column using the chosen temp value (max for optimistic, min for pessimistic)
            temp_value = None
            if temp_row is not None:
                temp_value = temp_row.get(temp_col)

            rows.append({
                "Date": k,
                "Temp (°C)": temp_value,
                "Chance of rain (%)": None if pp_row is None else pp_row.get("precip_prob_max"),
                "Condition": None if cond_row is None else code_to_text(cond_row.get("wcode_day", None))[0],
                "Temp Source": None if temp_row is None else temp_row.get("source"),
                "Chance of rain Source": None if pp_row is None else pp_row.get("source"),
                "Condition Source": None if cond_row is None else cond_row.get("source"),
            })

    return pd.DataFrame(rows)

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

def style_source_names_in_table(df: pd.DataFrame, cols):
    """Apply nice_source_name to given source columns if present."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: nice_source_name(x) if pd.notna(x) else x)
    return df

# ---------------------------
# UI
# ---------------------------

st.title("🌤️ Optimistic Weather")
st.caption("Cherry-pick across multiple models for the rosiest (or grumpiest) forecast — now with per-metric picks.")

left, right = st.columns([3, 2])
with left:
    location = st.text_input("Location", value="London")
with right:
    mode = st.radio("Forecast mood", options=["Optimistic", "Pessimistic"], horizontal=True,
                    help="Optimistic = highest temp, lowest rain chance, nicest condition. Pessimistic = opposite.")

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

    # Pretty-up sources (for expanded 'all sources' views)
    hourly_all = style_sources_column(hourly_all)
    daily_all = style_sources_column(daily_all)

    # ---------------------------
    # Hourly — Independent per-metric picks (Today)
    # ---------------------------
    st.subheader("Hourly — Independent picks (Today)")
    if not hourly_all.empty:
        hourly_ind = independent_metric_picks(hourly_all, mode=mode, is_hourly=True)
        if not hourly_ind.empty:
            hourly_ind = style_source_names_in_table(
                hourly_ind,
                ["Temp Source", "Chance of rain Source", "Condition Source"]
            ).sort_values("Time")
            st.dataframe(hourly_ind, use_container_width=True, hide_index=True)
        else:
            st.info("Hourly data wasn’t available for today from the chosen sources.")
    else:
        st.info("No hourly data for today returned.")

    # ---------------------------
    # Daily — Independent per-metric picks (Next 7 Days)
    # ---------------------------
    st.subheader("Daily — Independent picks (Next 7 Days)")
    if not daily_all.empty:
        today = datetime.now().date()
        daily_all_future = daily_all[daily_all["date"] >= today]
        daily_ind = independent_metric_picks(daily_all_future, mode=mode, is_hourly=False)
        if not daily_ind.empty:
            daily_ind = style_source_names_in_table(
                daily_ind,
                ["Temp Source", "Chance of rain Source", "Condition Source"]
            ).sort_values("Date")
            daily_ind = daily_ind.head(8)  # safety cap
            st.dataframe(daily_ind, use_container_width=True, hide_index=True)
        else:
            st.info("No daily data available for the selected sources.")
    else:
        st.info("No daily data returned.")

    # ---------------------------
    # Expanded views (optional)
    # ---------------------------
    with st.expander("See combined best/worst per time (single-source pick)"):
        st.caption("This collapses all metrics into one 'best/worst' source per time (original behavior).")
        # Hourly combined
        if not hourly_all.empty:
            hourly_pick = pick_best_per_key(hourly_all, ["time"], mode=mode, is_hourly=True)
            if not hourly_pick.empty:
                show = hourly_pick[["time", "source", "temperature_c", "precip_prob", "weathercode"]].copy()
                show["Condition"] = show["weathercode"].apply(lambda x: code_to_text(x)[0] if pd.notna(x) else "—")
                show = show.drop(columns=["weathercode"]).rename(columns={
                    "time": "Time",
                    "source": "Chosen Source",
                    "temperature_c": "Temp (°C)",
                    "precip_prob": "Chance of rain (%)",
                }).sort_values("Time")
                show["Chosen Source"] = show["Chosen Source"].apply(nice_source_name)
                st.dataframe(show, use_container_width=True, hide_index=True)
        # Daily combined
        if not daily_all.empty:
            daily_pick = pick_best_per_key(daily_all, ["date"], mode=mode, is_hourly=False)
            if not daily_pick.empty:
                dshow = daily_pick[["date", "source", "tmax_c", "tmin_c", "precip_prob_max", "wcode_day"]].copy()
                dshow["Condition"] = dshow["wcode_day"].apply(lambda x: code_to_text(x)[0] if pd.notna(x) else "—")
                dshow = dshow.drop(columns=["wcode_day"]).rename(columns={
                    "date": "Date",
                    "source": "Chosen Source",
                    "tmax_c": "High (°C)",
                    "tmin_c": "Low (°C)",
                    "precip_prob_max": "Max chance of rain (%)",
                }).sort_values("Date").head(8)
                dshow["Chosen Source"] = dshow["Chosen Source"].apply(nice_source_name)
                st.dataframe(dshow, use_container_width=True, hide_index=True)

    st.caption("Data via Open-Meteo (multiple numerical weather prediction models). Temperatures in Celsius.")

else:
    st.info("Enter a location, choose your mood, and click **Get forecast**. 😉")
