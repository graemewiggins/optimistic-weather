# app.py
import math
from datetime import datetime, timezone, timedelta
import pytz
import pandas as pd
import requests
import streamlit as st
import altair as alt


st.set_page_config(page_title="Optimistic Weather", page_icon="🌤️", layout="wide")

# =========================
# CSS — slim, iOS-style + horizontal scroller
# =========================
st.markdown("""
<style>
/* Headline */
.headline { text-align:center; padding: 6px; margin-bottom: 12px; }
.headline-city { font-size: 20px; font-weight: 600; margin-bottom: 2px; }
.headline-today { font-size: 14px; color: #444; }
.headline-condition { font-size: 15px; font-weight: 500; margin-top: 4px; }

/* Generic slim tiles/text */
.weather-card { text-align:center; font-size:13px; padding:4px; }
.weather-time { font-weight:600; margin: 2px 0 4px 0; text-align:center; }
.weather-condition { font-size:14px; white-space: normal; line-height:1.2; }
.weather-temp { font-size:13px; }
.weather-rain { font-size:12px; color:#555; }

/* Subtle card look */
.card { border:1px solid #ddd; border-radius:8px; padding:6px; }
.card + .card { margin-top:4px; }
.badge { font-size:10px; color:#888; text-transform:uppercase; letter-spacing:.3px; text-align:center; margin-bottom:2px; }

/* NEW: horizontal scroller for hourly tiles */
.hscroll {
  overflow-x: auto;
  overflow-y: hidden;
  white-space: nowrap;
  -webkit-overflow-scrolling: touch;
  padding-bottom: 6px;
  margin: 6px 0 2px 0;
  border-bottom: 1px solid #eee;
  scroll-snap-type: x proximity; /* nice optional snap feel */
}

/* A single slim hour tile */
.hour-tile {
  display: inline-block;
  vertical-align: top;
  width: 84px;            /* tweak thinner/wider here */
  box-sizing: border-box;
  margin-right: 6px;
  padding: 6px 6px 8px 6px;
  border: 1px solid #ddd;
  border-radius: 8px;
  background: #fff;
  scroll-snap-align: start;
}

/* Mini stacks inside a tile */
.hour-time { text-align:center; font-weight:600; font-size:12px; margin-bottom:4px; }
.hour-cond  { 
  text-align:center; 
  font-size:13px; 
  white-space: normal;   /* allow wrapping */
  word-wrap: break-word; /* break long words if needed */
  line-height: 1.2;      /* tighter lines */
}

.hour-temp  { text-align:center; font-size:12px; }
.hour-rain  { text-align:center; font-size:11px; color:#555; }

/* In side-by-side: two compact stacks */
.hour-half {
  border-top: 1px dashed #eee;
  margin-top: 6px;
  padding-top: 6px;
}
.hour-badge { font-size:10px; color:#888; text-transform:uppercase; letter-spacing:.3px; text-align:center; margin-bottom:2px; }
</style>
""", unsafe_allow_html=True)

# =========================
# Weather mapping
# =========================
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
    "Snow grains": "🌨️", "Slight snow": "🌨️", "Moderate snow": "🌨️", "Heavy snow": "❄️",
    "Thunderstorm": "⛈️", "Thunderstorm with slight hail": "⛈️", "Thunderstorm with heavy hail": "⛈️",
}

def code_to_text(code):
    """Return (label, niceness_rank) — robust to None/NaN/strings."""
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

# =========================
# Data fetch
# =========================
CANDIDATE_MODELS = [
    "gfs_seamless", "icon_seamless", "ecmwf_ifs04", "meteofrance_seamless",
    "gem_global", "jma_seamless", "ukmo_seamless",
]

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

@st.cache_data(show_spinner=False)
def fetch_openmeteo(lat, lon, tz, models):
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "timezone": tz, "forecast_days": 8,
        "hourly": ["temperature_2m", "precipitation_probability", "weathercode"],
        "daily": [
            "weathercode",
            "precipitation_probability_max",  # for optimistic daily rain
            "precipitation_probability_min",  # for pessimistic daily rain
            "temperature_2m_max",            # for optimistic daily temp
            "temperature_2m_min",            # for pessimistic daily temp
        ],
    }
    hourly_frames, daily_frames = [], []

    for m in models:
        p = dict(params); p["models"] = m
        try:
            rr = requests.get(base, params=p, timeout=20)
            rr.raise_for_status()
            dd = rr.json()
        except Exception:
            continue

        hh = dd.get("hourly", {})
        if hh and "time" in hh:
            h = pd.DataFrame({
                "time": pd.to_datetime(hh["time"]),
                "temperature_c": hh.get("temperature_2m"),
                "precip_prob": hh.get("precipitation_probability"),
                "weathercode": hh.get("weathercode"),
                "source": m,
            })
            hourly_frames.append(h)

        ddaily = dd.get("daily", {})
        if ddaily and ddaily.get("time"):
            dti = pd.to_datetime(ddaily["time"])
            d = pd.DataFrame({
                "date": dti.date,
                "wcode_day": ddaily.get("weathercode"),
                "precip_prob_max": ddaily.get("precipitation_probability_max"),
                "precip_prob_min": ddaily.get("precipitation_probability_min"),
                "tmax_c": ddaily.get("temperature_2m_max"),
                "tmin_c": ddaily.get("temperature_2m_min"),
                "source": m,
            }).iloc[:8]
            daily_frames.append(d)

    hourly_all = pd.concat(hourly_frames, ignore_index=True) if hourly_frames else pd.DataFrame()
    daily_all  = pd.concat(daily_frames,  ignore_index=True) if daily_frames  else pd.DataFrame()
    return hourly_all, daily_all

# =========================
# Picking logic
# =========================
def independent_metric_picks(df: pd.DataFrame, mode: str, is_hourly: bool):
    """
    Return per-metric values and their sources for each time/date.
    - Hourly Temp: max/min temperature
    - Hourly Rain: min/max precip_prob
    - Daily Temp: Optimistic=highest tmax, Pessimistic=lowest tmin (and we also compute ↑High/↓Low every time)
    - Daily Rain: Optimistic=highest precip_prob_max, Pessimistic=lowest precip_prob_min
    - Condition: best/worst rank, ignoring 'Unknown'
    """
    if df.empty:
        return pd.DataFrame()

    key = "time" if is_hourly else "date"
    out = []

    for k, g in df.groupby(key):
        g = g.copy()

        if is_hourly:
            # Temp
            temp_row = None
            if g["temperature_c"].notna().any():
                temp_row = g.loc[g["temperature_c"].idxmax()] if mode == "Optimistic" else g.loc[g["temperature_c"].idxmin()]
            # Rain
            pp_row = None
            if g["precip_prob"].notna().any():
                series = g["precip_prob"]
                pp_row = g.loc[series.fillna(9999).idxmin()] if mode == "Optimistic" else g.loc[series.fillna(-1).idxmax()]
            # Condition — ignore Unknown
            g["wc_text"] = g["weathercode"].apply(lambda x: code_to_text(x)[0])
            g["wc_rank"] = g["weathercode"].apply(lambda x: code_to_text(x)[1])
            gv = g[g["wc_text"] != "Unknown"]
            if not gv.empty:
                cond_row = gv.loc[gv["wc_rank"].idxmin()] if mode == "Optimistic" else gv.loc[gv["wc_rank"].idxmax()]
                cond_val = code_to_text(cond_row.get("weathercode"))[0]
                cond_src = nice_source_name(cond_row.get("source"))
            else:
                cond_val, cond_src = "Unknown", None

            out.append({
                "Time": k,
                "Temp (°C)": None if temp_row is None else temp_row.get("temperature_c"),
                "Temp Source": None if temp_row is None else nice_source_name(temp_row.get("source")),
                "Chance of rain (%)": None if pp_row is None else pp_row.get("precip_prob"),
                "Chance of rain Source": None if pp_row is None else nice_source_name(pp_row.get("source")),
                "Condition": cond_val,
                "Condition Source": cond_src,
            })

        else:
            # === DAILY: compute High/Low and sources (always) ===
            high_row = g.loc[g["tmax_c"].idxmax()] if g["tmax_c"].notna().any() else None
            low_row  = g.loc[g["tmin_c"].idxmin()] if g["tmin_c"].notna().any() else None
            high_val = None if high_row is None else high_row.get("tmax_c")
            low_val  = None if low_row  is None else low_row.get("tmin_c")
            high_src = None if high_row is None else nice_source_name(high_row.get("source"))
            low_src  = None if low_row  is None else nice_source_name(low_row.get("source"))

            # Mode-specific single "Temp (°C)" field (for other tables/headline)
            if mode == "Optimistic":
                temp_val = high_val
                temp_src = high_src
            else:
                temp_val = low_val
                temp_src = low_src

            # --- Rain (DAILY) — use precip_prob_max only (min for Optimistic, max for Pessimistic) ---
            pp = g[["precip_prob_min", "precip_prob_max", "source"]].copy()
            
            # Keep rows where max exists (that’s the metric we’re selecting on)
            pp = pp[pp["precip_prob_max"].notna()].copy()
            
            # Optional robustness: if ANY provider shows non-zero max, treat 0/0 pairs as suspect and drop them
            any_nonzero_max = (pp["precip_prob_max"].fillna(0) > 0).any()
            if any_nonzero_max:
                zero_zero_mask = (
                    pp["precip_prob_max"].fillna(0).eq(0) &
                    pp["precip_prob_min"].fillna(0).eq(0)
                )
                pp = pp[~zero_zero_mask]
            
            pp_row = None
            pp_val = None
            pp_src = None
            
            if not pp.empty:
                if mode == "Optimistic":
                    # driest => lowest precip_prob_max
                    idx = pp["precip_prob_max"].idxmin()
                else:
                    # wettest => highest precip_prob_max
                    idx = pp["precip_prob_max"].idxmax()
            
                # Use pp (or g; indices align)
                row = pp.loc[idx]
                pp_row = row
                pp_val = row.get("precip_prob_max")
                pp_src = nice_source_name(row.get("source"))
            else:
                pp_val, pp_src = None, None



            # Condition — ignore Unknown
            g["wc_text"] = g["wcode_day"].apply(lambda x: code_to_text(x)[0])
            g["wc_rank"] = g["wcode_day"].apply(lambda x: code_to_text(x)[1])
            gv = g[g["wc_text"] != "Unknown"]
            if not gv.empty:
                cond_pick = gv.loc[gv["wc_rank"].idxmin()] if mode == "Optimistic" else gv.loc[gv["wc_rank"].idxmax()]
                cond_val = code_to_text(cond_pick.get("wcode_day"))[0]
                cond_src = nice_source_name(cond_pick.get("source"))
            else:
                cond_val, cond_src = "Unknown", None

            out.append({
                "Date": k,
                "Temp (°C)": temp_val,                # mode-specific single temp for that day
                "Temp Source": temp_src,

                "Chance of rain (%)": pp_val,
                "Chance of rain Source": pp_src,

                "Condition": cond_val,
                "Condition Source": cond_src,

                # Always include high/low + sources for UI
                "High (°C)": high_val,
                "High Temp Source": high_src,
                "Low (°C)": low_val,
                "Low Temp Source": low_src,
            })

    return pd.DataFrame(out)

def side_by_side(df: pd.DataFrame, is_hourly: bool):
    opt = independent_metric_picks(df, "Optimistic", is_hourly)
    pes = independent_metric_picks(df, "Pessimistic", is_hourly)
    key = "Time" if is_hourly else "Date"
    merged = opt.merge(pes, on=key, suffixes=(" (Optimistic)", " (Pessimistic)"))
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

# =========================
# UI helpers (renderers)
# =========================
def render_headline(city, daily_df):
    """Single-mode headline: city, today, emoji/cond, ↑high/↓low, rain."""
    if daily_df.empty:
        return
    today = datetime.now().date()
    today_rows = daily_df[daily_df["Date"] == today]
    if today_rows.empty:
        return
    row = today_rows.iloc[0]

    cond = row.get("Condition") or "—"
    emoji = emoji_for(cond)

    hi = "—" if pd.isna(row.get("High (°C)")) else f"{round(row['High (°C)'])}°"
    lo = "—" if pd.isna(row.get("Low (°C)")) else f"{round(row['Low (°C)'])}°"
    rain = "—" if pd.isna(row.get("Chance of rain (%)")) else f"{int(round(row['Chance of rain (%)']))}%"

    st.markdown(
        f"""
        <div class="headline">
          <div class="headline-city">{city}</div>
          <div class="headline-today">Today</div>
          <div class="headline-condition">{emoji} {cond} · ↑ {hi} / ↓ {lo} · {rain} rain</div>
        </div>
        """, unsafe_allow_html=True
    )

def render_headline_side_by_side(city, daily_opt, daily_pes):
    st.markdown(f"<div class='headline-city'>{city}</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if not daily_opt.empty:
        r = daily_opt.iloc[0]
        cond = r.get("Condition") or "—"; emoji = emoji_for(cond)
        hi = "—" if pd.isna(r.get("High (°C)")) else f"{round(r['High (°C)'])}°"
        lo = "—" if pd.isna(r.get("Low (°C)")) else f"{round(r['Low (°C)'])}°"
        rain = "—" if pd.isna(r.get("Chance of rain (%)")) else f"{int(round(r['Chance of rain (%)']))}%"
        with c1:
            st.markdown(
                f"""
                <div class="headline card">
                  <div class="headline-today">Optimistic</div>
                  <div class="headline-condition">{emoji} {cond} · ↑ {hi} / ↓ {lo} · {rain} rain</div>
                </div>
                """, unsafe_allow_html=True
            )
    if not daily_pes.empty:
        r = daily_pes.iloc[0]
        cond = r.get("Condition") or "—"; emoji = emoji_for(cond)
        hi = "—" if pd.isna(r.get("High (°C)")) else f"{round(r['High (°C)'])}°"
        lo = "—" if pd.isna(r.get("Low (°C)")) else f"{round(r['Low (°C)'])}°"
        rain = "—" if pd.isna(r.get("Chance of rain (%)")) else f"{int(round(r['Chance of rain (%)']))}%"
        with c2:
            st.markdown(
                f"""
                <div class="headline card">
                  <div class="headline-today">Pessimistic</div>
                  <div class="headline-condition">{emoji} {cond} · ↑ {hi} / ↓ {lo} · {rain} rain</div>
                </div>
                """, unsafe_allow_html=True
            )



def render_hourly_ios(hours_df):
    """Single-mode hourly: slim tiles in a horizontal scroll strip (no leading spaces)."""
    if hours_df.empty:
        return
    tiles = []
    for _, r in hours_df.sort_values("Time").iterrows():
        t = pd.to_datetime(r["Time"]).strftime("%H:%M")
        cond = r.get("Condition") or "—"
        emoji = emoji_for(cond)
        temp = "—" if pd.isna(r.get("Temp (°C)")) else f"{round(r['Temp (°C)'])}°"
        rain = "—" if pd.isna(r.get("Chance of rain (%)")) else f"{int(round(r['Chance of rain (%)']))}%"
        cond_src = r.get("Condition Source") or "—"
        temp_src = r.get("Temp Source") or "—"
        rain_src = r.get("Chance of rain Source") or "—"

        tiles.append(
            '<div class="hour-tile" title="Condition: '
            + f'{cond_src}&#10;Temp: {temp_src}&#10;Rain: {rain_src}">'
            + f'<div class="hour-time">{t}</div>'
            + f'<div class="hour-cond">{emoji} {cond}</div>'
            + f'<div class="hour-temp">{temp}</div>'
            + f'<div class="hour-rain">{rain}</div>'
            + '</div>'
        )
    html = '<div class="hscroll">' + ''.join(tiles) + '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_hourly_stacked_side_by_side(hourly_ss):
    """Side-by-side: stacked optimistic/pessimistic per hour, horizontally scrollable (no leading spaces)."""
    if hourly_ss.empty:
        return
    tiles = []
    for _, row in hourly_ss.sort_values("Time").iterrows():
        t = pd.to_datetime(row["Time"]).strftime("%H:%M")

        # Optimistic
        o_cond = row.get("Optimistic Condition") or "—"
        o_emoji = emoji_for(o_cond)
        o_temp = "—" if pd.isna(row.get("Optimistic Temp")) else f"{round(row['Optimistic Temp'])}°"
        o_rain = "—" if pd.isna(row.get("Optimistic Chance of rain")) else f"{int(round(row['Optimistic Chance of rain']))}%"
        o_cond_src = row.get("Optimistic Condition Source") or "—"
        o_temp_src = row.get("Optimistic Temp Source") or "—"
        o_rain_src = row.get("Optimistic Chance of rain Source") or "—"

        # Pessimistic
        p_cond = row.get("Pessimistic Condition") or "—"
        p_emoji = emoji_for(p_cond)
        p_temp = "—" if pd.isna(row.get("Pessimistic Temp")) else f"{round(row['Pessimistic Temp'])}°"
        p_rain = "—" if pd.isna(row.get("Pessimistic Chance of rain")) else f"{int(round(row['Pessimistic Chance of rain']))}%"
        p_cond_src = row.get("Pessimistic Condition Source") or "—"
        p_temp_src = row.get("Pessimistic Temp Source") or "—"
        p_rain_src = row.get("Pessimistic Chance of rain Source") or "—"

        tiles.append(
            '<div class="hour-tile" title="Opt: Cond '
            + f'{o_cond_src}, Temp {o_temp_src}, Rain {o_rain_src}&#10;Pes: Cond {p_cond_src}, Temp {p_temp_src}, Rain {p_rain_src}">'
            + f'<div class="hour-time">{t}</div>'
            + '<div class="hour-badge">Optimistic</div>'
            + f'<div class="hour-cond">{o_emoji} {o_cond}</div>'
            + f'<div class="hour-temp">{o_temp}</div>'
            + f'<div class="hour-rain">{o_rain}</div>'
            + '<div class="hour-half">'
            + '<div class="hour-badge">Pessimistic</div>'
            + f'<div class="hour-cond">{p_emoji} {p_cond}</div>'
            + f'<div class="hour-temp">{p_temp}</div>'
            + f'<div class="hour-rain">{p_rain}</div>'
            + '</div>'
            + '</div>'
        )
    html = '<div class="hscroll">' + ''.join(tiles) + '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_daily_ios(daily_df):
    """Single-mode daily: vertical list with Day (dd/mm), Condition, ↑High/↓Low, Rain. Tooltips show sources."""
    if daily_df.empty:
        return
    for _, r in daily_df.iterrows():
        date = pd.to_datetime(r["Date"])
        day = date.strftime("%a")
        # Linux-friendly day/month (no leading zeros); adjust if needed
        date_str = date.strftime("%-d/%-m")
        cond = r.get("Condition") or "—"
        emoji = emoji_for(cond)
        hi = "—" if pd.isna(r.get("High (°C)")) else f"{round(r['High (°C)'])}°"
        lo = "—" if pd.isna(r.get("Low (°C)")) else f"{round(r['Low (°C)'])}°"
        rain = "—" if pd.isna(r.get("Chance of rain (%)")) else f"{int(round(r['Chance of rain (%)']))}%"
        cond_src = r.get("Condition Source") or "—"
        hi_src = r.get("High Temp Source") or "—"
        lo_src = r.get("Low Temp Source") or "—"
        rain_src = r.get("Chance of rain Source") or "—"

        st.markdown(
            f"""
            <div class="weather-card" style="text-align:left;">
              <strong>{day} ({date_str})</strong>&nbsp;&nbsp;
              <span class="weather-condition" title="Condition source: {cond_src}">{emoji} {cond}</span>&nbsp;&nbsp;
              <span class="weather-temp" title="High source: {hi_src}">↑ {hi}</span>&nbsp;/&nbsp;
              <span class="weather-temp" title="Low source: {lo_src}">↓ {lo}</span>&nbsp;&nbsp;
              <span class="weather-rain" title="Rain source: {rain_src}">{rain}</span>
            </div>
            """, unsafe_allow_html=True
        )

def render_daily_side_by_side_inline(daily_opt: pd.DataFrame, daily_pes: pd.DataFrame):
    """Side-by-side daily list as two stacked lines per day under a single date label."""
    if daily_opt.empty and daily_pes.empty:
        return

    opt_idx = daily_opt.set_index("Date") if not daily_opt.empty else pd.DataFrame().set_index(pd.Index([]))
    pes_idx = daily_pes.set_index("Date") if not daily_pes.empty else pd.DataFrame().set_index(pd.Index([]))
    all_dates = sorted(set(opt_idx.index).union(set(pes_idx.index)))[:8]

    for d in all_dates:
        date_dt = pd.to_datetime(d)
        day = date_dt.strftime("%a")
        date_str = date_dt.strftime("%-d/%-m")  # 13/9 format

        # Build line for Optimistic
        if d in opt_idx.index:
            ro = opt_idx.loc[d]
            o_cond = ro.get("Condition") or "—"
            o_emoji = emoji_for(o_cond)
            o_hi = "—" if pd.isna(ro.get("High (°C)")) else f"{round(ro['High (°C)'])}°"
            o_lo = "—" if pd.isna(ro.get("Low (°C)")) else f"{round(ro['Low (°C)'])}°"
            o_rain = "—" if pd.isna(ro.get("Chance of rain (%)")) else f"{int(round(ro['Chance of rain (%)']))}%"
            o_cond_src = ro.get("Condition Source") or "—"
            o_hi_src = ro.get("High Temp Source") or "—"
            o_lo_src = ro.get("Low Temp Source") or "—"
            o_rain_src = ro.get("Chance of rain Source") or "—"
            opt_html = (
                '<span class="badge" style="margin-right:6px;">Optimistic:</span>'
                f'<span class="weather-condition" title="Condition source: {o_cond_src}">{o_emoji} {o_cond}</span>'
                '&nbsp;&nbsp;'
                f'<span class="weather-temp" title="High source: {o_hi_src}">↑ {o_hi}</span>'
                ' / '
                f'<span class="weather-temp" title="Low source: {o_lo_src}">↓ {o_lo}</span>'
                '&nbsp;&nbsp;'
                f'<span class="weather-rain" title="Rain source: {o_rain_src}">{o_rain}</span>'
            )
        else:
            opt_html = '<span class="badge">Optimistic:</span> —'

        # Build line for Pessimistic
        if d in pes_idx.index:
            rp = pes_idx.loc[d]
            p_cond = rp.get("Condition") or "—"
            p_emoji = emoji_for(p_cond)
            p_hi = "—" if pd.isna(rp.get("High (°C)")) else f"{round(rp['High (°C)'])}°"
            p_lo = "—" if pd.isna(rp.get("Low (°C)")) else f"{round(rp['Low (°C)'])}°"
            p_rain = "—" if pd.isna(rp.get("Chance of rain (%)")) else f"{int(round(rp['Chance of rain (%)']))}%"
            p_cond_src = rp.get("Condition Source") or "—"
            p_hi_src = rp.get("High Temp Source") or "—"
            p_lo_src = rp.get("Low Temp Source") or "—"
            p_rain_src = rp.get("Chance of rain Source") or "—"
            pes_html = (
                '<span class="badge" style="margin-right:6px;">Pessimistic:</span>'
                f'<span class="weather-condition" title="Condition source: {p_cond_src}">{p_emoji} {p_cond}</span>'
                '&nbsp;&nbsp;'
                f'<span class="weather-temp" title="High source: {p_hi_src}">↑ {p_hi}</span>'
                ' / '
                f'<span class="weather-temp" title="Low source: {p_lo_src}">↓ {p_lo}</span>'
                '&nbsp;&nbsp;'
                f'<span class="weather-rain" title="Rain source: {p_rain_src}">{p_rain}</span>'
            )
        else:
            pes_html = '<span class="badge">Pessimistic:</span> —'

        # Render block: date header line + two indented lines
        block_html = (
            f'<div class="weather-card" style="text-align:left;">'
            f'<strong>{day} ({date_str})</strong>'
            f'<div style="margin-top:2px;">{opt_html}</div>'
            f'<div style="margin-top:2px;">{pes_html}</div>'
            f'</div>'
        )
        st.markdown(block_html, unsafe_allow_html=True)

def render_hourly_temp_chart(hourly_ss: pd.DataFrame):
        """Line chart of hourly optimistic vs pessimistic temps."""
        if hourly_ss.empty:
            return
        
        df_plot = hourly_ss.copy()
        df_plot = df_plot[["Time", "Optimistic Temp", "Pessimistic Temp"]].melt(
            id_vars="Time", var_name="Scenario", value_name="Temp (°C)"
        )
        df_plot["Time"] = pd.to_datetime(df_plot["Time"])
        
        chart = (
            alt.Chart(df_plot)
            .mark_line(point=True)
            .encode(
            x=alt.X("Time:T", title="Hour"),
            y=alt.Y(
                "Temp (°C):Q",
                title="Temperature (°C)",
                scale=alt.Scale(zero=False)  # <-- key line: don't force baseline at 0
            ),
            color=alt.Color("Scenario:N", title="", sort=["Optimistic Temp","Pessimistic Temp"]),
            tooltip=["Time:T", "Scenario:N", alt.Tooltip("Temp (°C):Q", format=".1f")]
        )
            .properties(height=300)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

def render_hourly_rain_chart(hourly_ss: pd.DataFrame):
    """Line chart of hourly optimistic vs pessimistic rain chance (%)."""
    if hourly_ss.empty:
        return

    df_plot = hourly_ss.copy()
    df_plot = df_plot[["Time", "Optimistic Chance of rain", "Pessimistic Chance of rain"]].melt(
        id_vars="Time", var_name="Scenario", value_name="Chance of rain (%)"
    )
    df_plot["Time"] = pd.to_datetime(df_plot["Time"])

    chart = (
        alt.Chart(df_plot)
        .mark_line(point=True)
        .encode(
            x=alt.X("Time:T", title="Hour"),
            y=alt.Y(
                "Chance of rain (%):Q",
                title="Chance of rain (%)",
                scale=alt.Scale(domain=[0, 100])  # <-- lock axis to 0–100
            ),
            color=alt.Color("Scenario:N", title="", sort=["Optimistic Chance of rain", "Pessimistic Chance of rain"]),
            tooltip=["Time:T", "Scenario:N", alt.Tooltip("Chance of rain (%):Q", format=".0f")]
        )
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

# ========= iOS-style Hourly Temperature Chart (new) =========
import altair as alt
import pandas as pd
import pytz
from datetime import datetime, timedelta

def prepare_ios_hourly_df_from_ss(hourly_ss: pd.DataFrame, *, tz: str) -> pd.DataFrame:
    """
    Build the input for the iOS-style chart from your side_by_side() hourly table.

    Returns columns:
      ts (tz-aware), temp_pess, temp_opt, icon_url (optional; left empty here)
    """
    if hourly_ss is None or hourly_ss.empty:
        return pd.DataFrame(columns=["ts", "temp_pess", "temp_opt", "icon_url"])

    df = hourly_ss.copy()
    tzinfo = pytz.timezone(tz)

    # Ensure timestamp
    df["ts"] = pd.to_datetime(df["Time"], errors="coerce")
    # Treat as local tz (Open-Meteo already returned local tz earlier)
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize(tzinfo)
    else:
        df["ts"] = df["ts"].dt.tz_convert(tzinfo)

    # Temperatures
    df["temp_pess"] = pd.to_numeric(df["Pessimistic Temp"], errors="coerce")
    df["temp_opt"]  = pd.to_numeric(df["Optimistic Temp"], errors="coerce")

    # No icon URLs yet (you can wire later if you have an assets set)
    df["icon_url"] = pd.NA

    return df[["ts", "temp_pess", "temp_opt", "icon_url"]].dropna(subset=["ts"])


def render_ios_hourly_temp_chart(
    df: pd.DataFrame,
    *,
    tz: str = "Europe/London",
    width: int | None = None,
    main_height: int = 180,
    icons_height: int = 28,
):
    """
    Expect df with columns:
      - ts (datetime64[ns], tz-aware or naive)
      - temp_pess (float)
      - temp_opt (float)
      - icon_url (optional str)
    """
    if df is None or df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_text().properties(height=main_height)

    tzinfo = pytz.timezone(tz)

    # --- Ensure timestamps are in local tz, then make them NAIVE (no tzinfo) for Altair/Vega-Lite
    d = df.copy()
    ts = pd.to_datetime(d["ts"], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(tzinfo)     # assume local tz if none
    else:
        ts = ts.dt.tz_convert(tzinfo)      # convert to local tz if aware
    d["ts"] = ts.dt.tz_localize(None)      # <-- make naive for Vega-Lite

    # --- Use naive "now" and day window corresponding to local tz
    now_local = datetime.now(tzinfo).replace(tzinfo=None)
    start_of_day = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1)

    d = d.sort_values("ts")

    # keep only today's 24h window (naive)
    d = d[(d["ts"] >= start_of_day) & (d["ts"] < end_of_day)].copy()
    if d.empty:
        # fallback: use all provided rows, but still naive/locally aligned
        d = df.copy()
        ts = pd.to_datetime(d["ts"], errors="coerce")
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(tzinfo)
        else:
            ts = ts.dt.tz_convert(tzinfo)
        d["ts"] = ts.dt.tz_localize(None)
        d = d.sort_values("ts")

    # split past/future using naive timestamps
    past = d[d["ts"] <= now_local]
    future = d[d["ts"] > now_local]

    # find H/L on pessimistic (guard if empty/NaN)
    if d["temp_pess"].notna().any():
        low_idx = d["temp_pess"].idxmin()
        high_idx = d["temp_pess"].idxmax()
        marks_df = pd.DataFrame(
            [
                {"ts": d.loc[low_idx, "ts"], "temp": d.loc[low_idx, "temp_pess"], "label": "L"},
                {"ts": d.loc[high_idx, "ts"], "temp": d.loc[high_idx, "temp_pess"], "label": "H"},
            ]
        )
    else:
        marks_df = pd.DataFrame(columns=["ts", "temp", "label"])

    # axes encodings (right-side Y), x ticks at 00/06/12/18
    tick_times = [start_of_day + timedelta(hours=h) for h in (0, 6, 12, 18)]
    x_enc = alt.X(
        "ts:T",
        title=None,
        scale=alt.Scale(domain=[start_of_day, end_of_day]),
        axis=alt.Axis(
            values=tick_times,
            labelExpr="timeFormat(datum.value, '%H')",
            labelColor="#d6d6d6",
            tickColor="#3a3a3a",
            tickSize=4,
            labelPadding=8,
            grid=False,
        ),
    )
    y_right = alt.Y(
        "temp_pess:Q",
        title=None,
        axis=alt.Axis(
            orient="right",
            labelColor="#d6d6d6",
            tickColor="#3a3a3a",
            domain=False,
            labelPadding=4,
            grid=True,                 
            gridOpacity=0.15,
            gridColor="#3a3a3a"
        ),
        scale=alt.Scale(nice=True),
    )

    # vertical gridlines (00/06/12/18) and current hour (all naive)
    six_hr_rules = (
        alt.Chart(pd.DataFrame({"ts": tick_times}))
        .mark_rule(stroke="#7a7a7a", strokeOpacity=0.25, strokeWidth=1)
        .encode(x="ts:T")
    )
    now_rule = (
        alt.Chart(pd.DataFrame({"ts": [now_local]}))
        .mark_rule(stroke="#cfcfcf", strokeOpacity=0.6, strokeWidth=1)
        .encode(x="ts:T")
    )

    # past panel shading (naive domain)
    past_rect = (
        alt.Chart(pd.DataFrame({"x0": [start_of_day], "x1": [min(now_local, end_of_day)]}))
        .mark_rect(opacity=0.25, color="#0b0c0e")
        .encode(x="x0:T", x2="x1:T")
    )

    # gradient fills (future brighter, past darker)
    grad_future = alt.Gradient(
        gradient="linear", x1=0, x2=0, y1=1, y2=0,
        stops=[alt.GradientStop(color="#0a7b8c", offset=0), alt.GradientStop(color="#f3e766", offset=1)],
    )
    grad_past = alt.Gradient(
        gradient="linear", x1=0, x2=0, y1=1, y2=0,
        stops=[alt.GradientStop(color="#075d69", offset=0), alt.GradientStop(color="#c7bc4f", offset=1)],
    )

    area_future = alt.Chart(future).mark_area(interpolate="monotone", opacity=0.8, color=grad_future)\
        .encode(x=x_enc, y=y_right)
    area_past = alt.Chart(past).mark_area(interpolate="monotone", opacity=0.8, color=grad_past)\
        .encode(x=x_enc, y=y_right)

    # pessimistic line (solid), darker in past
    line_future = alt.Chart(future).mark_line(interpolate="monotone", strokeWidth=2.5, color="#f3e766")\
        .encode(x=x_enc, y="temp_pess:Q")
    line_past = alt.Chart(past).mark_line(interpolate="monotone", strokeWidth=2.5, color="#c7bc4f")\
        .encode(x=x_enc, y="temp_pess:Q")

    # optimistic comparison (dashed), tone split
    opt_future = alt.Chart(future).mark_line(interpolate="monotone", strokeWidth=2, color="#f3e766",
                                             strokeDash=[5, 4], opacity=0.9)\
        .encode(x=x_enc, y="temp_opt:Q")
    opt_past = alt.Chart(past).mark_line(interpolate="monotone", strokeWidth=2, color="#c7bc4f",
                                         strokeDash=[5, 4], opacity=0.9)\
        .encode(x=x_enc, y="temp_opt:Q")

    # H/L markers (only if we had valid temps)
    main_layers = [
        past_rect, six_hr_rules, now_rule,
        area_past, area_future,
        line_past, line_future,
        opt_past, opt_future,
    ]
    if not marks_df.empty:
        hl_points = alt.Chart(marks_df).mark_point(size=70, filled=True, color="#0d0d0d",
                                                   stroke="#f6f6f6", strokeWidth=1.5)\
            .encode(x="ts:T", y="temp:Q")
        hl_rings = alt.Chart(marks_df).mark_point(size=130, filled=False, stroke="#f6f6f6", strokeWidth=1.2)\
            .encode(x="ts:T", y="temp:Q")
        hl_labels = alt.Chart(marks_df).mark_text(align="center", baseline="bottom", dy=-10, fontSize=12,
                                                  color="#f6f6f6", fontWeight="bold")\
            .encode(x="ts:T", y="temp:Q", text="label:N")
        main_layers.extend([hl_points, hl_rings, hl_labels])

    main_chart = alt.layer(*main_layers).properties(height=main_height)
    if width is not None:
        main_chart = main_chart.properties(width=width)


    # icon row every 2h (only if icon_url present)
    icons = d.copy()
    icons["hour"] = icons["ts"].dt.hour
    icons = icons[(icons["hour"] % 2 == 0) & icons["icon_url"].notna()]
    if not icons.empty:
        icon_chart = alt.Chart(icons).mark_image(width=18, height=18).encode(x=x_enc, url="icon_url:N")\
            .properties(height=icons_height)
        if width is not None:
            icon_chart = icon_chart.properties(width=width)

        divider = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(stroke="#7a7a7a", strokeOpacity=0.25)\
            .encode(y=alt.value(icons_height-1))
        top = alt.layer(icon_chart, divider).resolve_scale(y="independent")
        chart = alt.vconcat(top, main_chart).resolve_scale(x="shared")
    else:
        chart = main_chart

    # dark theme
    chart = chart.configure_view(
        stroke="#2a2a2a", fill="#111215"
    ).configure_axisX(
        grid=False, labelFontSize=11, title=None
    ).configure_axisY(
        grid=True, gridOpacity=0.15, gridColor="#3a3a3a", labelFontSize=11, title=None
    ).configure(
        background="#111215"
    )
    return chart



# =========================
# App UI
# =========================
st.title("🌤️ Optimistic Weather")
st.caption("iOS-style layout; optimistic/pessimistic picks across multiple models. Hover for sources.")

left, right = st.columns([3, 2])
with left:
    location = st.text_input("Location", value="London")
with right:
    mode = st.radio("Forecast mode", options=["Optimistic", "Pessimistic", "Side by side"], horizontal=True)

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

    st.success(f"Using **{city}, {country}** · ({lat:.4f}, {lon:.4f}) · TZ: {tz}")

    with st.spinner("Fetching forecasts..."):
        hourly_all, daily_all = fetch_openmeteo(lat, lon, tz, models_selected)

   # with st.expander("🔍 Debug: Raw daily data"):
   #     st.dataframe(daily_all)

   # with st.expander("🔍 Debug: Raw hourly data"):
   #     st.dataframe(hourly_all)

    if hourly_all.empty and daily_all.empty:
        st.warning("No data returned. Try different models or a nearby location.")
        st.stop()

    # Prepare datasets
    # Build a rolling 24h window in the location's timezone
    # (Open-Meteo returned times are already in that tz because we requested timezone=tz)
    start = pd.Timestamp.now(tz).floor("H").tz_localize(None)
    end = start + pd.Timedelta(hours=23)
    hourly_rolling = hourly_all[(hourly_all["time"] >= start) & (hourly_all["time"] <= end)].copy()
    
    # full 00:00–24:00 window for TODAY (local tz, naive)
    day_start = pd.Timestamp.now(tz).normalize().tz_localize(None)
    day_end   = day_start + pd.Timedelta(days=1)
    hourly_today = hourly_all[(hourly_all["time"] >= day_start) & (hourly_all["time"] < day_end)].copy()


  
    today = datetime.now().date()
    daily_future = daily_all[daily_all["date"] >= today].copy()

    # Helpers to build mode-specific tables
    def hourly_for_mode(m):
        return independent_metric_picks(hourly_rolling, mode=m, is_hourly=True).sort_values("Time")

    def daily_for_mode(m):
        return independent_metric_picks(daily_future, mode=m, is_hourly=False).sort_values("Date").head(8)

    if mode in ["Optimistic", "Pessimistic"]:
        daily_mode = daily_for_mode(mode)
        # Headline
        if not daily_mode.empty:
            # render headline requires the high/low fields which are present
            st.markdown("", unsafe_allow_html=True)
        render_headline(city, daily_mode)

        # Hourly — slim scroll strip
        st.markdown("### Hourly — Next 24 Hours")
        hours = hourly_for_mode(mode)
        if not hours.empty:
            render_hourly_ios(hours)
            # iOS-style temperature chart (pessimistic solid + optimistic dashed)
            hourly_ss = side_by_side(hourly_today, is_hourly=True)
            ios_df = prepare_ios_hourly_df_from_ss(hourly_ss, tz=tz)
            st.markdown("### Temperature — iOS style")
            st.altair_chart(render_ios_hourly_temp_chart(ios_df, tz=tz), use_container_width=True)
        else:
            st.info("No hourly data for today.")

        # Daily — vertical list
        st.markdown("### Daily — Next 7 Days")
        if not daily_mode.empty:
            render_daily_ios(daily_mode)
        else:
            st.info("No daily data available for the selected models.")

    else:
        # SIDE BY SIDE
        daily_opt = daily_for_mode("Optimistic")
        daily_pes = daily_for_mode("Pessimistic")
        render_headline_side_by_side(city, daily_opt, daily_pes)

        st.markdown("### Hourly — Next 24 Hours (Optimistic over Pessimistic)")
        hourly_ss = side_by_side(hourly_rolling, is_hourly=True)
        if not hourly_ss.empty:
            render_hourly_stacked_side_by_side(hourly_ss)

            st.markdown("### Temperature — iOS style")
            ios_df = prepare_ios_hourly_df_from_ss(hourly_today, tz=tz)
            st.altair_chart(render_ios_hourly_temp_chart(ios_df, tz=tz), use_container_width=True)

