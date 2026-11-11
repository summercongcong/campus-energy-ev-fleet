"""Validate EV session data, calculate billing, and export the clean table."""

import argparse
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "ev_campus_dataset.csv"
OUTPUT = ROOT / "output" / "cleaned_energy_sales_data.csv"
CARBON_FACTOR = 0.475
RATES = {"AC": 6.0, "DC": 15.0}
USERS = ("Student", "Staff", "Visitor")
CATEGORIES = {
    "Charger_Type": tuple(RATES),
    "User_Type": USERS,
    "Grid_Status_Label": ("Normal", "Grid_Surge", "Overheated"),
}
MEASURES = ("Duration_Mins", "Energy_kWh", "Battery_Temp_C")
REQUIRED = {
    "Session_ID", "Start_Hour", "Overstay_Mins", *CATEGORIES, *MEASURES
}


def clean_sessions(raw: pd.DataFrame) -> pd.DataFrame:
    """Check schema and operational fields, then median-fill measurements."""
    missing = REQUIRED.difference(raw.columns)
    if missing or raw.empty:
        raise ValueError(f"Expected EV sessions. Missing columns: {sorted(missing)}")

    data = raw.copy()
    ids = data["Session_ID"].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Session_ID must be present and unique.")
    data["Session_ID"] = ids

    for column, valid in CATEGORIES.items():
        names = {value.casefold(): value for value in valid}
        clean = data[column].astype("string").str.strip().str.casefold()
        data[column] = clean.map(names)
        if data[column].isna().any():
            raise ValueError(f"Invalid or missing {column}. Expected {valid}.")

    for column in (*MEASURES, "Start_Hour", "Overstay_Mins"):
        values = pd.to_numeric(data[column], errors="coerce").astype(float)
        invalid = (data[column].notna() & values.isna()) | np.isinf(values)
        if invalid.any():
            raise ValueError(f"{column} contains invalid numeric values.")
        if column in MEASURES:
            if values.isna().all():
                raise ValueError(f"No usable observations for {column}.")
            logging.info("%s: %d values filled", column, values.isna().sum())
            values = values.fillna(values.median())
        elif values.isna().any() or values.mod(1).ne(0).any():
            raise ValueError(f"{column} must contain whole numbers.")
        data[column] = values

    if not data["Start_Hour"].between(0, 23).all():
        raise ValueError("Start_Hour must be between 0 and 23.")
    if (data["Duration_Mins"] <= 0).any():
        raise ValueError("Charging duration must be positive.")
    if (data[["Energy_kWh", "Overstay_Mins"]] < 0).any().any():
        raise ValueError("Energy and overstay cannot be negative.")
    data[["Start_Hour", "Overstay_Mins"]] = data[
        ["Start_Hour", "Overstay_Mins"]
    ].astype(int)
    return data


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute session metrics using the tariff inferred from the supplied CSV."""
    result = data.copy()
    result["Duration_Hours"] = hours = result["Duration_Mins"] / 60
    energy = result["Energy_kWh"]
    result["Power_Rate_kW"] = energy / hours
    result["Carbon_Savings_kgCO2"] = energy * CARBON_FACTOR
    result["Base_Rate_MYR"] = result["Charger_Type"].map(RATES)
    result["Base_Fee_MYR"] = base = hours * result["Base_Rate_MYR"]
    peak = result["Start_Hour"].between(12, 17)
    result["Peak_Surge_MYR"] = np.where(peak, 0.25 * base, 0)
    result["Overstay_Fee_MYR"] = (
        result["Overstay_Mins"].sub(15).clip(lower=0) * 0.50
    )
    result["Net_Billable_MYR"] = (
        base + result["Peak_Surge_MYR"] + result["Overstay_Fee_MYR"]
    ).round(2)
    return result


def load_sessions(path: Path = SOURCE) -> pd.DataFrame:
    """Read a CSV and return validated, enriched session records."""
    return add_features(clean_sessions(pd.read_csv(path)))


def write_csv(data: pd.DataFrame, destination: Path) -> None:
    """Write via a sibling temporary file to avoid a partial output on failure."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        data.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    """Run the pipeline CLI and print descriptive statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        data = load_sessions(args.input)
        write_csv(data, args.output)
    except (OSError, ValueError) as error:
        logging.error("%s", error)
        return 1
    columns = [
        "Energy_kWh", "Power_Rate_kW", "Carbon_Savings_kgCO2", "Net_Billable_MYR"
    ]
    print(f"Exported {len(data):,} sessions to {args.output}")
    print(data[columns].describe().round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
