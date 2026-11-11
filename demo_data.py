"""Generate explicitly synthetic timestamps and campus telemetry for the demo."""

import numpy as np
import pandas as pd

from pipeline import ROOT, SOURCE, write_csv

TIMESTAMPS = ROOT / "demo_session_timestamps.csv"
TELEMETRY = ROOT / "demo_campus_telemetry.csv"
READINGS = ["Ambient_Temp_C", "Grid_Load_kW", "Solar_kW"]


def build_demo() -> None:
    """Write reproducible fixtures without altering any provided CSV."""
    rng = np.random.default_rng(42)
    raw = pd.read_csv(SOURCE)
    days = rng.permutation(np.arange(len(raw)) % 20)
    start = pd.Timestamp("2026-04-01")
    timestamps = start + pd.to_timedelta(days * 24 + raw["Start_Hour"], unit="h")
    write_csv(
        pd.DataFrame({"Session_ID": raw["Session_ID"], "Timestamp": timestamps}),
        TIMESTAMPS,
    )
    hourly = pd.date_range(start, periods=20 * 24, freq="h")
    hour = hourly.hour.to_numpy()
    daylight = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
    ambient = 27 + 7 * daylight + rng.normal(0, 1.1, len(hour))
    grid = 105 + 8 * daylight + 45 * ((hour >= 12) & (hour < 18))
    grid = grid + rng.normal(0, 7, len(hour))
    solar = daylight * rng.uniform(55, 85, len(hour))
    telemetry = pd.DataFrame(dict(zip(READINGS, (ambient, grid, solar))))
    telemetry.insert(0, "Timestamp", hourly)
    write_csv(telemetry.round(2), TELEMETRY)


def attach_demo(sessions: pd.DataFrame) -> pd.DataFrame:
    """Join each session to one timestamp and each hour to one telemetry row."""
    times = pd.read_csv(TIMESTAMPS, parse_dates=["Timestamp"])
    telemetry = pd.read_csv(TELEMETRY, parse_dates=["Timestamp"])
    data = sessions.merge(times, on="Session_ID", how="left", validate="1:1")
    if data["Timestamp"].isna().any():
        raise ValueError("Missing demo timestamps. Run python demo_data.py.")
    if not data["Timestamp"].dt.hour.eq(data["Start_Hour"]).all():
        raise ValueError("Demo timestamp and session start hour disagree.")
    data = data.merge(telemetry, on="Timestamp", how="left", validate="m:1")
    if data[READINGS].isna().any().any():
        raise ValueError("Missing telemetry for one or more demo hours.")
    return data.assign(
        Timestamp_Source="synthetic_demo", Telemetry_Source="synthetic_demo"
    )


if __name__ == "__main__":
    build_demo()
