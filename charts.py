"""Matplotlib and Seaborn figures shared by the app and report."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns

from demo_data import attach_demo
from pipeline import ROOT, USERS, load_sessions

BLUE = "#214d79"
TEAL = "#127e78"
GOLD = "#bd6817"


def axes(title, xlabel, ylabel):
    """Create a common figure canvas with a readable chart title and units."""
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(8.2, 3.35), layout="constrained")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    return fig, ax


def hourly_revenue(data):
    """Compare AC and DC revenue using the session start hour."""
    fig, ax = axes("Charging revenue by start hour", "Start hour", "Revenue (MYR)")
    hourly = data.groupby(["Start_Hour", "Charger_Type"])["Net_Billable_MYR"]
    totals = hourly.sum().unstack().reindex(
        index=range(24), columns=["AC", "DC"]
    ).fillna(0)
    for name, shift, color in (("AC", -0.2, BLUE), ("DC", 0.2, TEAL)):
        ax.bar(totals.index + shift, totals[name], 0.4, label=name, color=color)
    ax.set_xticks(range(0, 24, 2))
    ax.legend(title="Charger", frameon=False)
    return fig


def correlation(data):
    """Show session-level Pearson correlations with synthetic telemetry."""
    values = data[["Ambient_Temp_C", "Duration_Mins", "Grid_Load_kW"]].copy()
    values.columns = ["Ambient temp.", "Charge duration", "Grid load"]
    fig, ax = axes("Pearson correlation | demo telemetry", "", "")
    sns.heatmap(
        values.corr(), annot=True, fmt=".2f", cmap="vlag",
        vmin=-1, vmax=1, center=0, linewidths=1, ax=ax,
        cbar_kws={"label": "Pearson r"},
    )
    ax.tick_params(axis="y", labelrotation=0)
    return fig


def grid_solar(data):
    """Plot one campus reading per selected session hour, leaving gaps visible."""
    fig, ax = axes("Hourly grid peak and solar | demo telemetry", "Date", "Power (kW)")
    hourly = data.groupby("Timestamp").agg(
        grid_peak=("Grid_Load_kW", "max"), solar=("Solar_kW", "mean")
    ).asfreq("h")
    ax.plot(hourly.index, hourly["grid_peak"], ".-", lw=1, ms=2,
            color=BLUE, label="Grid demand peak")
    ax.plot(hourly.index, hourly["solar"], ".-", lw=1, ms=2,
            color=GOLD, label="Solar generation")
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend(frameon=False, ncols=2)
    return fig


def user_cost(data):
    """Compare net session costs by user type."""
    fig, ax = axes("Session cost by user type", "User type", "Net fee (MYR)")
    sns.boxplot(
        data=data, x="User_Type", y="Net_Billable_MYR", order=USERS,
        color="#63b0ab", width=0.5, fliersize=3, ax=ax,
    )
    return fig


def temperature_energy(data):
    """Plot observed battery temperature, delivered energy, and linear fit."""
    fig, ax = axes(
        "Delivered energy by battery temperature", "Battery temperature (°C)",
        "Energy delivered (kWh)",
    )
    x = data["Battery_Temp_C"]
    sns.regplot(
        data=data, x="Battery_Temp_C", y="Energy_kWh",
        fit_reg=len(data) >= 2 and x.nunique() >= 2, ci=None, ax=ax,
        scatter_kws={"s": 18, "alpha": 0.34, "color": BLUE},
        line_kws={"color": GOLD, "linewidth": 2.2},
    )
    return fig


def export_figures():
    """Write all five report figures as 220-dpi PNGs."""
    destination = ROOT / "output" / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    sessions = attach_demo(load_sessions())
    figures = (hourly_revenue, correlation, grid_solar, user_cost, temperature_energy)
    for chart in figures:
        figure = chart(sessions)
        figure.savefig(destination / f"{chart.__name__}.png", dpi=220)
        plt.close(figure)


if __name__ == "__main__":
    export_figures()
