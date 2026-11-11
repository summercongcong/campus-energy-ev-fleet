"""Section A1 examples of NumPy slicing, masking and Pandas operations."""

import numpy as np
import pandas as pd

from pipeline import SOURCE, load_sessions


def main() -> None:
    """Show telemetry array operations and session table transformations."""
    raw = pd.read_csv(SOURCE)

    # Example feeder readings, indexed by feeder and hour of day.
    feeders_kw = np.arange(48, dtype=float).reshape(2, 24)
    peak_load_kw = feeders_kw[:, 12:18]
    battery_c = raw["Battery_Temp_C"].to_numpy(dtype=float)
    hot_readings = battery_c[battery_c > 45]

    # Derive a power series that retains missing source measurements.
    duration_hours = raw["Duration_Mins"].div(60)
    power = raw["Energy_kWh"].div(duration_hours)
    raw["Power_kW"] = power.fillna(power.median())
    clean = load_sessions()
    revenue = clean.groupby("Charger_Type")["Net_Billable_MYR"].sum()

    print("Illustrative peak-hour feeder readings (kW):")
    print(peak_load_kw)
    print(f"Battery readings above 45 °C: {hot_readings.size}")
    print(f"Power values filled with the median: {power.isna().sum()}")
    print("Billed revenue by charger type (MYR):")
    print(revenue.to_string())


if __name__ == "__main__":
    main()
