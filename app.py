# app.py
import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timezone
import math

st.set_page_config(page_title="Optimistic Weather", page_icon="🌤️", layout="wide")

# ---------------------------
# Helpers & mapping
# ---------------------------

WEATHER_CODE_MAP = {
    0: ("Clear sky", 0), 1: ("Mainly clear", 1), 2: ("Partly cloudy", 2), 3: ("Overcast", 3),
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
EMOJI_FOR_TEXT = {
    "Clear sky": "☀️", "Mainly clear": "🌤️", "Partly cloudy": "⛅", "Overcast": "☁️",
    "Fog": "🌫️", "Light drizzle": "🌦️", "Moderate drizzle": "🌦️", "Dense drizzle": "🌧️",
    "Slight rain": "🌧️", "Moderate rain": "🌧️", "Heavy rain": "🌧️",
    "Slight snow": "🌨️", "Moderate snow": "🌨️", "Heavy snow": "❄️",
    "Thunderstorm": "⛈️", "Thunderstorm with slight hail": "⛈️", "Thunderstorm with heavy hail": "⛈️",
}

def code_to_text(code):
    try:
        if code is None:
            return "Unknown", 99
        if isinstance(code, float) and math.isnan(code):
            return "Unknown", 99
        c = int(code)
    except Exception:
        return "Unknown", 99
    return WEATHER_CODE_MAP.get(c, ("Unknown", 99))

def emoji_for(text):
    return EMOJI_FOR_TEXT.get(text, "🌡️" if text != "Unknown" else "❔")

def nice_source_name(s: str) -> str:
    mapping = {
        "gfs_seamless": "GFS (NOAA)", "icon_seamless": "ICON (DWD)", "ecmwf_ifs04": "ECMWF IFS 0.4°",
        "meteofrance_seamless": "Météo-France", "gem_global": "GEM (Canada)", "jma_seamless": "JMA",
        "ukmo_seamless": "UKMO (Met Office)",
    }
    return mapping.get(s, s)

# ---------------------------
# Data fetch
# ---------------------------

@st.cache_data(show_spinner=False)
def geocode(query: str):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(url, params={"name": query, "count": 1, "language": "en", "format": "json"}, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data.get("results"):
        raise ValueError("Location not found.")
    res = data["results"][0]
    return float(res["latitude"]), float(res["longitude"]), res.get("name", query), res.get("country", ""), res.get("timezone", "UTC")

CANDIDATE_MODELS = [
    "gfs_seamless", "icon_seamless", "ecmwf_ifs04", "meteofrance_seamless", "gem_global", "jma_seamless", "ukmo_seamless",
]

@st.cache_data(show_spinner=False)
def fetch_openmeteo(lat, lon, tz, models):
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "timezone": tz, "forecast_days": 8,
        "hourly": ["temperature_2m", "precipitation_probability", "weathercode"],
        "daily": ["weathercode", "precipitation_probability_max", "precipitation_probability_min",
                  "temperature_2m_max", "temperature_2m_min"],
    }
    hourly_frames, daily_frames = [], []
    for m in models:
        p = dict(params); p["models"] = m
        try:
            rr = requests.get(base, params=p, timeout=20); rr.raise_for_status(); dd = rr.json()
        except Exception:
            continue

        hh = dd.get("hourly", {})
        if hh and "time" in hh:
            hdf = pd.DataFrame({
                "time": pd.to_datetime(hh["time"]),
                "temperature_c": hh.get("temperature_2m"),
                "precip_prob": hh.get("precipitation_probability"),
                "weathercode": hh.get("weathercode"),
                "source": m,
            })
            today_str = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
            hdf["date"] = hdf["time"].dt.strftime("%Y-%m-%d")
            hdf = hdf[hdf["date"] == today_str].drop(columns=["date"])
            hourly_frames.append(hdf)

        ddaily = dd.get("daily", {})
        if ddaily and ddaily.get("time"):
            dti = pd.to_datetime(ddaily["time"])
            ddf = pd.DataFrame({
                "date": dti.date,
                "wcode_day": ddaily.get("weathercode"),
                "precip_prob_max": ddaily.get("precipitation_probability_max"),
                "precip_prob_min": ddaily.get("precipitation_probability_min"),
                "tmax_c": ddaily.get("temperature_2m_max"),
                "tmin_c": ddaily.get("temperature_2m_min"),
                "source": m,
            }).iloc[:8]
            daily_frames.append(ddf)

    hourly_all = pd.concat(hourly_frames, ignore_index=True) if hourly_frames else pd.DataFrame()
    daily_all  = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    return hourly_all, daily_all

# ---------------------------
# Picking logic (with Unknown filtered for condition)
# ---------------------------

def independent_metric_picks(df: pd.DataFrame, mode: str, is_hourly: bool):
    if df.empty: return pd.DataFrame()
    key = "time" if is_hourly else "date"
    rows = []
    for k, g in df.groupby(key):
        g = g.copy()
        if is_hourly:
            # temp
            temp_row = (g.loc[g["temperature_c"].idxmax()] if mode=="Optimistic"
                        else g.loc[g["temperature_c"].idxmin()]) if g["temperature_c"].notna().any() else None
            # rain
            pp_series = g["precip_prob"]
            if mode == "Optimistic":
                pp_row = g.loc[pp_series.fillna(9999).idxmin()] if pp_series.notna().any() else None
            else:
                pp_row = g.loc[pp_series.fillna(-1).idxmax()] if pp_series.notna().any() else None
            # condition (ignore Unknown)
            g["wc_text"] = g["weathercode"].apply(lambda x: code_to_text(x)[0])
            g["wc_rank"] = g["weathercode"].apply(lambda x: code_to_text(x)[1])
            gv = g[g["wc_text"]!="Unknown"]
            if not gv.empty:
                cond_row = gv.loc[gv["wc_rank"].idxmin()] if mode=="Optimistic" else gv.loc[gv["wc_rank"].idxmax()]
                cond_val = code_to_text(cond_row.get("weathercode", None))[0]
                cond_src = nice_source_name(cond_row.get("source"))
            else:
                cond_val, cond_src = "Unknown", None
            rows.append({
                "Time": k,
                "Temp (°C)": None if temp_row is None else temp_row.get("temperature_c"),
                "Temp Source": None if temp_row is None else nice_source_name(temp_row.get("source")),
                "Chance of rain (%)": None if pp_row is None else pp_row.get("precip_prob"),
                "Chance of rain Source": None if pp_row is None else nice_source_name(pp_row.get("source")),
                "Condition": cond_val, "Condition Source": cond_src,
            })
        else:
            # temp (opt: highest tmax, pes: lowest tmin)
            if mode=="Optimistic":
                temp_row = g.loc[g["tmax_c"].idxmax()] if g["tmax_c"].notna().any() else None
                temp_val = None if temp_row is None else temp_row.get("tmax_c")
            else:
                temp_row = g.loc[g["tmin_c"].idxmin()] if g["tmin_c"].notna().any() else None
                temp_val = None if temp_row is None else temp_row.get("tmin_c")
            # rain (opt: highest max, pes: lowest min)
            if mode=="Optimistic":
                pp_row = g.loc[g["precip_prob_max"].fillna(-1).idxmax()] if g["precip_prob_max"].notna().any() else None
                pp_val = None if pp_row is None else pp_row.get("precip_prob_max")
            else:
                pp_row = g.loc[g["precip_prob_min"].fillna(9999).idxmin()] if g["precip_prob_min"].notna().any() else None
                pp_val = None if pp_row is None else pp_row.get("precip_prob_min")
            # condition (ignore Unknown)
            g["wc_text"] = g["wcode_day"].apply(lambda x: code_to_text(x)[0])
            g["wc_rank"] = g["wcode_day"].apply(lambda x: code_to_text(x)[1])
            gv = g[g["wc_text"]!="Unknown"]
            if not gv.empty:
                cond_row = gv.loc[gv["wc_rank"].idxmin()] if mode=="Optimistic" else gv.loc[gv["wc_rank"].idxmax()]
                cond_val = code_to_text(cond_row.get("wcode_day", None))[0]
                cond_src = nice_source_name(cond_row.get("source"))
            else:
                cond_val, cond_src = "Unknown", None
            rows.append({
                "Date": k,
                "Temp (°C)": temp_val, "Temp Source": None if temp_row is None else nice_source_name(temp_row.get("source")),
                "Chance of rain (%)": pp_val, "Chance of rain Source": None if pp_row is None else nice_source_name(pp_row.get("source")),
                "Condition": cond_val, "Condition Source": cond_src,
            })
    return pd.DataFrame(rows)

def side_by_side(df: pd.DataFrame, is_hourly: bool):
    opt = independent_metric_picks(df, "Optimistic", is_hourly)
    pes = independent_metric_picks(df, "Pessimistic", is_hourly)
    key = "Time" if is_hourly else "Date"
    merged = opt.merge(pes, on=key, suffixes=(" (Optimistic)", " (Pessimistic)"))
    # flatten names
    return merged.rename(columns={
        "Temp (°C) (Optimistic)": "Optimistic Temp",
        "Temp (°C) (Pessimistic)": "Pessimistic Temp",
        "Temp Source (Optimistic)": "Optimistic Temp Source",
        "Temp Source (Pessimistic)": "Pessimistic Temp Source",
        "Chance of rain (%) (Optimistic)": "Optimistic Chance of rain",
        "Chance of rain (%) (Pessimistic)": "Pessimistic Chance of rain",
        "Chance of rain Source (Optimistic)": "Optimistic Chance of rain Source",
        "Chance of rain Source (Pessimistic)": "Pessimistic Chance of rain Source",
        "Condition (Optimistic)": "Optimistic Condition",
        "Condition (Pessimistic)": "Pessimistic Condition",
        "Condition Source (Optimistic)": "Optimistic Condition Source",
        "Condition Source (Pessimistic)": "Pessimistic Condition Source",
    })

# ---------------------------
# UI Pieces (iOS-style headline + lists)
# ---------------------------

def render_headline(city, daily_df):
    """Show today's headline: City, High/Low, Condition, Rain (slimmer iOS-style)."""
    if daily_df.empty:
        return
    today = datetime.now().date()
    today_rows = daily_df[daily_df["Date"] == today]
    if today_rows.empty:
        return
    row = today_rows.iloc[0]
    cond = row["Condition"] or "—"
    emoji = emoji_for(cond)
    temp = "—" if pd.isna(row["Temp (°C)"]) else f"{round(row['Temp (°C)'])}°"
    rain = "—" if pd.isna(row["Chance of rain (%)"]) else f"{int(round(row['Chance of rain (%)']))}%"

    st.markdown(
        f"""
        <div class="headline">
          <div class="headline-city">{city}</div>
          <div class="headline-today">Today</div>
          <div class="headline-condition">{emoji} {cond} · {temp} · {rain} rain</div>
        </div>
        """, unsafe_allow_html=True
    )
 # When in optimistic mode, high is tmax; pessimistic, low is tmin. If both wanted we'd compute separately.

def render_headline_side_by_side(city, daily_opt, daily_pes):
    st.markdown(f"<div class='headline-city'>{city}</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if not daily_opt.empty:
            r = daily_opt.iloc[0]
            cond = r["Condition"] or "—"
            emoji = emoji_for(cond)
            temp = "—" if pd.isna(r["Temp (°C)"]) else f"{round(r['Temp (°C)'])}°"
            rain = "—" if pd.isna(r["Chance of rain (%)"]) else f"{int(round(r['Chance of rain (%)']))}%"
            st.markdown(
                f"""
                <div class="headline" style="border:1px solid #ddd; border-radius:8px;">
                  <div class="headline-today">Optimistic</div>
                  <div class="headline-condition">{emoji} {cond} · {temp} · {rain} rain</div>
                </div>
                """, unsafe_allow_html=True
            )
    with c2:
        if not daily_pes.empty:
            r = daily_pes.iloc[0]
            cond = r["Condition"] or "—"
            emoji = emoji_for(cond)
            temp = "—" if pd.isna(r["Temp (°C)"]) else f"{round(r['Temp (°C)'])}°"
            rain = "—" if pd.isna(r["Chance of rain (%)"]) else f"{int(round(r['Chance of rain (%)']))}%"
            st.markdown(
                f"""
                <div class="headline" style="border:1px solid #ddd; border-radius:8px;">
                  <div class="headline-today">Pessimistic</div>
                  <div class="headline-condition">{emoji} {cond} · {temp} · {rain} rain</div>
                </div>
                """, unsafe_allow_html=True
            )


def render_daily_vertical(daily_df):
    """Daily vertical list: Day (dd/mm), Condition, High/Low, Rain."""
    if daily_df.empty: return
    for _, r in daily_df.iterrows():
        date = r["Date"]
        day = pd.to_datetime(date).strftime("%a")
        date_str = pd.to_datetime(date).strftime("%-d/%-m")
        cond = r["Condition"]; emoji = emoji_for(cond)
        rain = r["Chance of rain (%)"]
        temp = r["Temp (°C)"]
        st.markdown(
            f"{day} ({date_str})  &nbsp;&nbsp; {emoji} **{cond}**  &nbsp;&nbsp; "
            f"**{temp if temp is not None else '—'}°**  &nbsp;&nbsp; "
            f"{'' if rain is None else str(int(round(rain)))}%"
        )

def render_hourly_stacked_side_by_side(hourly_ss):
    """
    Hourly tiles across the page; each tile shows Optimistic row (top) and Pessimistic row (bottom).
    Sources are shown as tooltips via the 'help' argument on st.metric.
    """
    if hourly_ss.empty: return
    hourly_ss = hourly_ss.sort_values("Time")
    hours = list(hourly_ss["Time"].unique())
    # Lay out in rows of N columns for responsiveness
    N = 4  # columns per row
    for i in range(0, len(hours), N):
        cols = st.columns(min(N, len(hours) - i))
        for j, col in enumerate(cols, start=i):
            t = hours[j]
            row = hourly_ss[hourly_ss["Time"] == t].iloc[0]
            with col:
                st.markdown(f"<div style='text-align:center; font-weight:600;'>{pd.to_datetime(t).strftime('%H:%M')}</div>", unsafe_allow_html=True)
                # Optimistic block
                with st.container(border=True):
                    st.caption("Optimistic")
                    st.metric("Condition", f"{emoji_for(row['Optimistic Condition'])} {row['Optimistic Condition'] or '—'}",
                              help=f"Condition source: {row.get('Optimistic Condition Source') or '—'}")
                    st.metric("Temp", f"{'—' if pd.isna(row['Optimistic Temp']) else round(row['Optimistic Temp'],1)}°",
                              help=f"Temp source: {row.get('Optimistic Temp Source') or '—'}")
                    st.metric("Rain", f"{'—' if pd.isna(row['Optimistic Chance of rain']) else int(round(row['Optimistic Chance of rain']))}%",
                              help=f"Rain source: {row.get('Optimistic Chance of rain Source') or '—'}")
                # Pessimistic block
                with st.container(border=True):
                    st.caption("Pessimistic")
                    st.metric("Condition", f"{emoji_for(row['Pessimistic Condition'])} {row['Pessimistic Condition'] or '—'}",
                              help=f"Condition source: {row.get('Pessimistic Condition Source') or '—'}")
                    st.metric("Temp", f"{'—' if pd.isna(row['Pessimistic Temp']) else round(row['Pessimistic Temp'],1)}°",
                              help=f"Temp source: {row.get('Pessimistic Temp Source') or '—'}")
                    st.metric("Rain", f"{'—' if pd.isna(row['Pessimistic Chance of rain']) else int(round(row['Pessimistic Chance of rain']))}%",
                              help=f"Rain source: {row.get('Pessimistic Chance of rain Source') or '—'}")

# ---------------------------
# Main UI
# ---------------------------

st.title("🌤️ Optimistic Weather")
st.caption("iOS-style layout with optimistic/pessimistic picking across multiple models. Sources show on hover.")
st.markdown("""
<style>
.headline {
    text-align: center;
    padding: 6px;
    margin-bottom: 12px;
}
.headline-city {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 2px;
}
.headline-today {
    font-size: 14px;
    color: #444;
}
.headline-condition {
    font-size: 15px;
    font-weight: 500;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)



left, right = st.columns([3, 2])
with left:
    location = st.text_input("Location", value="London")
with right:
    mode = st.radio("Forecast mode", options=["Optimistic", "Pessimistic", "Side by side"], horizontal=True)

models_selected = st.multiselect(
    "Sources (models)", options=CANDIDATE_MODELS,
    default=["gfs_seamless", "icon_seamless", "ecmwf_ifs04", "ukmo_seamless"],
)

if st.button("Get forecast", type="primary"):
    try:
        lat, lon, city, country, tz = geocode(location)
    except Exception as e:
        st.error(f"Could not find that location. {e}"); st.stop()

    st.success(f"Using **{city}, {country}** · ({lat:.4f}, {lon:.4f}) · TZ: {tz}")

    with st.spinner("Fetching forecasts..."):
        hourly_all, daily_all = fetch_openmeteo(lat, lon, tz, models_selected)

    if hourly_all.empty and daily_all.empty:
        st.warning("No data returned. Try different models or a nearby location."); st.stop()

    # Build views
    # Hourly (today) dataset
    hourly_today = hourly_all.copy()

    # Daily (>= today)
    today = datetime.now().date()
    daily_future = daily_all[daily_all["date"] >= today].copy()

    # Compose independent picks for current mode
    def daily_for_mode(m):
        df = independent_metric_picks(daily_future, mode=m, is_hourly=False)
        # Normalize for headline: keep only today row if present
        return df.sort_values("Date").head(8)

    def hourly_for_mode(m):
        return independent_metric_picks(hourly_today, mode=m, is_hourly=True).sort_values("Time")

    if mode in ["Optimistic", "Pessimistic"]:
        # HEADLINE
        daily_mode = daily_for_mode(mode)
        render_headline(city, daily_mode)
        # HOURLY (horizontal, each column = hour, top-to-bottom: condition, temp, rain)
        if not hourly_today.empty:
            hours = hourly_for_mode(mode)
            # horizontally scrollable feel: use rows of columns
            N = 6
            for i in range(0, len(hours), N):
                cols = st.columns(min(N, len(hours) - i))
                for j, col in enumerate(cols, start=i):
                    r = hours.iloc[j]
                    with col:
                        st.markdown(f"<div style='text-align:center; font-weight:600;'>{pd.to_datetime(r['Time']).strftime('%H:%M')}</div>", unsafe_allow_html=True)
                        # Three stacked lines with tooltips (source in help)
                        st.metric("Condition", f"{emoji_for(r['Condition'])} {r['Condition'] or '—'}",
                                  help=f"Condition source: {r.get('Condition Source') or '—'}")
                        st.metric("Temp", f"{'—' if pd.isna(r['Temp (°C)']) else round(r['Temp (°C)'],1)}°",
                                  help=f"Temp source: {r.get('Temp Source') or '—'}")
                        st.metric("Rain", f"{'—' if pd.isna(r['Chance of rain (%)']) else int(round(r['Chance of rain (%)']))}%",
                                  help=f"Rain source: {r.get('Chance of rain Source') or '—'}")
        else:
            st.info("No hourly data for today returned.")

        # DAILY (vertical list with date)
        daily_mode = daily_for_mode(mode)
        st.markdown("### Daily — Next 7 Days")
        render_daily_vertical(daily_mode)

    else:
        # SIDE BY SIDE
        # Build side-by-side hourly and daily frames
        hourly_ss = side_by_side(hourly_today, is_hourly=True)
        daily_opt = daily_for_mode("Optimistic")
        daily_pes = daily_for_mode("Pessimistic")
        # HEADLINE (two mini cards)
        st.markdown(f"### {city}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Optimistic**")
            if not daily_opt.empty:
                r = daily_opt.iloc[0]
                st.metric("Condition", f"{emoji_for(r['Condition'])} {r['Condition'] or '—'}",
                          help=f"Condition source: {r.get('Condition Source') or '—'}")
                st.metric("Temp", f"{'—' if pd.isna(r['Temp (°C)']) else round(r['Temp (°C)'],1)}°",
                          help=f"Temp source: {r.get('Temp Source') or '—'}")
                st.metric("Rain", f"{'—' if pd.isna(r['Chance of rain (%)']) else int(round(r['Chance of rain (%)']))}%",
                          help=f"Rain source: {r.get('Chance of rain Source') or '—'}")
        with c2:
            st.markdown("**Pessimistic**")
            if not daily_pes.empty:
                r = daily_pes.iloc[0]
                st.metric("Condition", f"{emoji_for(r['Condition'])} {r['Condition'] or '—'}",
                          help=f"Condition source: {r.get('Condition Source') or '—'}")
                st.metric("Temp", f"{'—' if pd.isna(r['Temp (°C)']) else round(r['Temp (°C)'],1)}°",
                          help=f"Temp source: {r.get('Temp Source') or '—'}")
                st.metric("Rain", f"{'—' if pd.isna(r['Chance of rain (%)']) else int(round(r['Chance of rain (%)']))}%",
                          help=f"Rain source: {r.get('Chance of rain Source') or '—'}")

        # HOURLY — stacked opt/pes per hour (sources in tooltips)
        st.markdown("### Hourly — Today (Optimistic over Pessimistic)")
        render_hourly_stacked_side_by_side(hourly_ss)

        # DAILY — two brackets per row (we’ll keep sources as tooltips via two columns)
        st.markdown("### Daily — Next 7 Days")
        # Build per-day paired rows
        daily_opt = daily_opt.set_index("Date")
        daily_pes = daily_pes.set_index("Date")
        all_dates = sorted(set(daily_opt.index).union(daily_pes.index))[:8]
        for d in all_dates:
            day = pd.to_datetime(d).strftime("%a")
            date_str = pd.to_datetime(d).strftime("%-d/%-m")
            st.markdown(f"**{day} ({date_str})**")
            colL, colR = st.columns(2)
            with colL:
                if d in daily_opt.index:
                    r = daily_opt.loc[d]
                    st.caption("Optimistic")
                    st.metric("Condition", f"{emoji_for(r['Condition'])} {r['Condition'] or '—'}",
                              help=f"Condition source: {r.get('Condition Source') or '—'}")
                    st.metric("Temp", f"{'—' if pd.isna(r['Temp (°C)']) else round(r['Temp (°C)'],1)}°",
                              help=f"Temp source: {r.get('Temp Source') or '—'}")
                    st.metric("Rain", f"{'—' if pd.isna(r['Chance of rain (%)']) else int(round(r['Chance of rain (%)']))}%",
                              help=f"Rain source: {r.get('Chance of rain Source') or '—'}")
            with colR:
                if d in daily_pes.index:
                    r = daily_pes.loc[d]
                    st.caption("Pessimistic")
                    st.metric("Condition", f"{emoji_for(r['Condition'])} {r['Condition'] or '—'}",
                              help=f"Condition source: {r.get('Condition Source') or '—'}")
                    st.metric("Temp", f"{'—' if pd.isna(r['Temp (°C)']) else round(r['Temp (°C)'],1)}°",
                              help=f"Temp source: {r.get('Temp Source') or '—'}")
                    st.metric("Rain", f"{'—' if pd.isna(r['Chance of rain (%)']) else int(round(r['Chance of rain (%)']))}%",
                              help=f"Rain source: {r.get('Chance of rain Source') or '—'}")

    st.caption("Data via Open-Meteo. Temperatures in Celsius.")

else:
    st.info("Enter a location, choose a mode, and click **Get forecast**. 😉")
