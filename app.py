"""Streamlit dashboard for campus EV charging and demonstration telemetry."""

from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from charts import grid_solar, temperature_energy, user_cost
from demo_data import TELEMETRY, TIMESTAMPS, attach_demo
from pipeline import SOURCE, load_sessions


@st.cache_data
def read_data(file_versions):
    """Load, validate and cache the data until one of the CSVs changes."""
    return attach_demo(load_sessions())


def filter_period(data, start, end):
    """Filter inclusive dates by using an exclusive next-day endpoint."""
    return data.loc[
        data["Timestamp"].ge(pd.Timestamp(start))
        & data["Timestamp"].lt(pd.Timestamp(end) + pd.Timedelta(days=1))
    ]


def alerts(data, threshold):
    """Find sessions above both strict high-risk cutoffs."""
    return data.loc[
        data["Overstay_Mins"].gt(20) & data["Battery_Temp_C"].gt(threshold)
    ]


def kpis(current, previous, threshold, has_baseline):
    """Display absolute differences versus the previous equal-length period."""
    metrics = (
        ("Total Energy Delivered", "Energy_kWh", "kWh"),
        ("Carbon Offset", "Carbon_Savings_kgCO2", "kg CO₂"),
        ("Total Revenue", "Net_Billable_MYR", "MYR"),
        ("Active Alert Count", None, ""),
    )
    for card, (label, field, unit) in zip(st.columns(4), metrics):
        if field:
            value, prior = current[field].sum(), previous[field].sum()
            value_text = f"{value:,.2f} {unit}"
            delta = f"{value - prior:+,.2f} {unit}"
        else:
            value = len(alerts(current, threshold))
            prior = len(alerts(previous, threshold))
            value_text, delta = str(value), f"{value - prior:+,d}"
        card.metric(
            label, value_text, delta if has_baseline else None, border=True,
            delta_color="normal" if field else "inverse",
        )
    st.caption(
        "Change vs. the preceding period of equal length under the same filters."
        if has_baseline else "The full preceding period falls outside dataset coverage."
    )


def show_overview(data):
    """Render the three dashboard panels for the shared selection."""
    st.subheader("Campus operations")
    if data.empty:
        st.info("No sessions match the selected filters.")
        return
    st.caption("Campus telemetry is shown for hours with matching sessions.")
    for builder in (grid_solar, user_cost, temperature_energy):
        figure = builder(data)
        st.pyplot(figure)
        plt.close(figure)


def show_alerts(data, threshold):
    """Display high-risk sessions and severity banners."""
    st.subheader("High-risk overstay")
    matches = alerts(data, threshold)
    st.caption(f"Rule: overstay > 20 min and battery temperature > {threshold} °C.")
    if matches.empty:
        st.info("No sessions match the alert rule.")
        return
    st.warning(f"{len(matches)} high-risk sessions in the selection.")
    critical = matches.loc[matches["Battery_Temp_C"].gt(threshold + 5)]
    if not critical.empty:
        st.error(f"{len(critical)} exceed the temperature cutoff by more than 5 °C.")
    columns = [
        "Session_ID", "Timestamp", "Charger_Type", "Overstay_Mins", "Battery_Temp_C"
    ]
    st.dataframe(
        matches.sort_values("Battery_Temp_C", ascending=False)[columns],
        hide_index=True,
    )


def show_data(data):
    """Show and download the same filtered records."""
    st.subheader("Session records")
    st.caption(f"{len(data):,} matching sessions")
    st.dataframe(data, hide_index=True)
    st.download_button(
        "Download filtered CSV", data.to_csv(index=False).encode("utf-8"),
        file_name="filtered_energy_sessions.csv", mime="text/csv",
        disabled=data.empty,
    )


def main():
    """Configure global controls and route to one of three pages."""
    st.set_page_config(page_title="Campus Energy & EV Fleet", layout="wide")
    st.title("Campus Energy & EV Fleet")
    st.caption("ShopEasy Logistics · Taylor’s Smart Campus Division")
    st.info(
        "Demo data: session dates, ambient temperature, grid load and solar "
        "power are synthetic."
    )
    try:
        files = (SOURCE, TIMESTAMPS, TELEMETRY)
        data = read_data(tuple(path.stat().st_mtime_ns for path in files))
    except (OSError, ValueError) as error:
        st.error(f"Unable to load session data. {error}")
        st.stop()

    first = data["Timestamp"].min().date()
    last = data["Timestamp"].max().date()
    st.sidebar.header("Filters")
    dates = st.sidebar.date_input(
        "Session dates", (max(first, last - timedelta(days=6)), last),
        min_value=first, max_value=last,
    )
    energy = st.sidebar.slider(
        "Minimum energy (kWh)", 0.0, max(1.0, float(data["Energy_kWh"].max())),
        0.0, step=0.1,
    )
    chargers = st.sidebar.multiselect("Charger types", ["AC", "DC"], ["AC", "DC"])
    with st.sidebar:
        threshold = st.slider("Alert temperature (°C)", 25, 60, 40)

    if len(dates) != 2:
        st.info("Choose both the start and end dates.")
        st.stop()
    start, end = dates
    eligible = data.loc[
        data["Energy_kWh"].ge(energy) & data["Charger_Type"].isin(chargers)
    ]
    current = filter_period(eligible, start, end)
    prior_end = start - timedelta(days=1)
    prior_start = start - timedelta(days=(end - start).days + 1)
    previous = filter_period(eligible, prior_start, prior_end)
    kpis(current, previous, threshold, prior_start >= first)

    navigation = st.navigation(
        [
            st.Page(lambda: show_overview(current), title="Overview", default=True),
            st.Page(lambda: show_alerts(current, threshold), title="Alert centre",
                    url_path="alerts"),
            st.Page(lambda: show_data(current), title="Data & export", url_path="data"),
        ],
        position="sidebar",
    )
    navigation.run()


if __name__ == "__main__":
    main()
