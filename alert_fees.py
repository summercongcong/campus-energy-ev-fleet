"""Section A2: calculate a 15% surcharge for high-risk charging sessions."""

from decimal import Decimal, ROUND_HALF_UP
from functools import reduce

RATE = Decimal("0.15")
CENT = Decimal("0.01")


def add_surcharge(session):
    """Return a new record with the surcharge and revised energy fee."""
    base = Decimal(str(session["Base_Fee_MYR"]))
    adjusted = base * (1 + RATE)
    surcharge = (adjusted - base).quantize(CENT, rounding=ROUND_HALF_UP)
    return {**session, "Surcharge_MYR": surcharge,
            "Adjusted_Fee_MYR": base + surcharge}


def total_surcharge(sessions):
    """Sum only the surcharge component for sessions above both cutoffs."""
    # The short, pure lambda keeps both strict cutoffs visible. It captures
    # no mutable state and produces one Boolean per record.
    high_risk = filter(
        lambda row: row["Overstay_Mins"] > 20 and row["Battery_Temp_C"] > 40,
        sessions,
    )
    # map lazily transforms accepted rows. No intermediate list of alerts
    # or surcharged records is allocated, even when sessions is a list.
    priced = map(add_surcharge, high_risk)
    # This pure lambda keeps only the running Decimal total. The explicit
    # zero also defines the answer when no session meets both cutoffs.
    return reduce(
        lambda total, row: total + row["Surcharge_MYR"],
        priced,
        Decimal("0.00"),
    )


if __name__ == "__main__":
    from pipeline import load_sessions

    columns = ["Overstay_Mins", "Battery_Temp_C", "Base_Fee_MYR"]
    records = load_sessions()[columns].round(2).to_dict("records")
    print(f"Alert surcharge revenue: MYR {total_surcharge(records):,.2f}")
