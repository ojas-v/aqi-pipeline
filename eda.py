"""Exploratory analysis for the UCI Beijing dataset.

Run after the first download with: python eda.py
Figures are saved to artifacts/eda/. Read the accompanying EDA_NOTES.md before
interpreting correlations or comparing stations.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from forecasting import load_data

OUT = Path(__file__).resolve().parent / "artifacts" / "eda"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data()
    numeric = [c for c in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"] if c in df]
    # 1. Target distribution (right skew is expected in particulate concentration).
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df["PM2.5"].dropna(), bins=60, kde=True, ax=ax)
    ax.set(title="PM2.5 concentration distribution", xlabel="PM2.5 (µg/m³)", ylabel="Hourly records")
    fig.tight_layout(); fig.savefig(OUT / "pm25_distribution.png", dpi=160); plt.close(fig)
    # 2. Missingness by variable, before dropping/imputing any sensor values.
    fig, ax = plt.subplots(figsize=(9, 5))
    missing = df[numeric].isna().mean().sort_values(ascending=False).mul(100)
    sns.barplot(x=missing.values, y=missing.index, ax=ax)
    ax.set(title="Missing observations by variable", xlabel="Missing (%)", ylabel="")
    fig.tight_layout(); fig.savefig(OUT / "missingness.png", dpi=160); plt.close(fig)
    # 3. Correlation matrix; association is not causal evidence.
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[numeric].corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Pearson correlations (pairwise complete observations)")
    fig.tight_layout(); fig.savefig(OUT / "correlation_heatmap.png", dpi=160); plt.close(fig)
    # 4. Diurnal pattern averaged across all stations/years.
    df["hour"] = df["timestamp"].dt.hour
    hourly = df.groupby("hour")["PM2.5"].agg(["median", "mean"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hourly["hour"], hourly["median"], marker="o", label="Median")
    ax.plot(hourly["hour"], hourly["mean"], marker=".", label="Mean")
    ax.set(title="PM2.5 by hour of day", xlabel="Hour", ylabel="PM2.5 (µg/m³)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "diurnal_pm25.png", dpi=160); plt.close(fig)
    # 5. Seasonal pattern.
    df["month"] = df["timestamp"].dt.month
    monthly = df.groupby("month")["PM2.5"].median()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(monthly.index, monthly.values, marker="o")
    ax.set(title="Monthly median PM2.5", xlabel="Month", ylabel="Median PM2.5 (µg/m³)", xticks=range(1, 13))
    fig.tight_layout(); fig.savefig(OUT / "seasonal_pm25.png", dpi=160); plt.close(fig)
    # 6. Station coverage and median level.
    station = df.groupby("station")["PM2.5"].agg(["count", "median"]).sort_values("median")
    station.to_csv(OUT / "station_summary.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    station["median"].plot(kind="barh", ax=ax)
    ax.set(title="Median PM2.5 by station", xlabel="Median PM2.5 (µg/m³)", ylabel="Station")
    fig.tight_layout(); fig.savefig(OUT / "station_medians.png", dpi=160); plt.close(fig)
    print(f"Saved EDA plots and station summary to {OUT}")

if __name__ == "__main__":
    main()
