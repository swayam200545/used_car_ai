"""
Generates a realistic Indian used-car dataset (data/used_cars.csv).

WHY SYNTHETIC?
  Public datasets (e.g. the CarDekho CSV on Kaggle) can be dropped in instead -
  just give them the same columns (see README). This generator exists so the
  project runs end-to-end out of the box and the neural network has enough
  rows to learn non-linear effects (depreciation curve, diesel age limits,
  EV depreciation, ownership, kilometre wear).

Run:  python -m ml.make_dataset --rows 15000
"""
import argparse
import os

import numpy as np
import pandas as pd

REF_YEAR = 2026

# price = approx. ex-showroom (Rs lakh) of a mid variant, cc/bhp/seats, fuels offered,
# auto = probability a used unit is an automatic, w = popularity weight
def _m(price, cc, bhp, seats, fuels, auto, w=1.0):
    return dict(price=price, cc=cc, bhp=bhp, seats=seats, fuels=fuels, auto=auto, w=w)

CATALOG = {
    "Maruti Suzuki": {
        "Swift":   _m(7.0, 1197, 82, 5, ["Petrol", "CNG"], 0.25, 2.0),
        "Baleno":  _m(7.8, 1197, 88, 5, ["Petrol", "CNG"], 0.28, 1.8),
        "Dzire":   _m(7.8, 1197, 82, 5, ["Petrol", "CNG"], 0.22, 1.6),
        "Brezza":  _m(10.5, 1462, 103, 5, ["Petrol", "CNG"], 0.30, 1.4),
        "WagonR":  _m(6.0, 998, 66, 5, ["Petrol", "CNG"], 0.15, 2.0),
        "Ertiga":  _m(10.5, 1462, 103, 7, ["Petrol", "CNG"], 0.20, 1.4),
    },
    "Hyundai": {
        "i20":          _m(8.5, 1197, 82, 5, ["Petrol", "Diesel"], 0.30, 1.5),
        "Creta":        _m(15.0, 1497, 113, 5, ["Petrol", "Diesel"], 0.40, 1.6),
        "Venue":        _m(10.5, 1197, 82, 5, ["Petrol", "Diesel"], 0.30, 1.0),
        "Verna":        _m(12.5, 1497, 113, 5, ["Petrol", "Diesel"], 0.40, 0.8),
        "Grand i10 Nios": _m(6.8, 1197, 82, 5, ["Petrol", "CNG"], 0.20, 1.0),
    },
    "Tata": {
        "Nexon":  _m(11.0, 1199, 118, 5, ["Petrol", "Diesel", "Electric"], 0.30, 1.6),
        "Harrier": _m(20.0, 1956, 167, 5, ["Diesel"], 0.45, 0.6),
        "Punch":  _m(7.5, 1199, 86, 5, ["Petrol", "CNG"], 0.25, 1.0),
        "Altroz": _m(8.2, 1199, 86, 5, ["Petrol", "Diesel"], 0.20, 0.8),
        "Tiago":  _m(6.2, 1199, 86, 5, ["Petrol", "CNG"], 0.20, 1.2),
    },
    "Mahindra": {
        "XUV700":   _m(18.0, 1999, 197, 7, ["Petrol", "Diesel"], 0.50, 0.9),
        "Scorpio N": _m(17.0, 2184, 172, 7, ["Diesel", "Petrol"], 0.35, 0.9),
        "Thar":     _m(15.5, 2184, 130, 4, ["Diesel", "Petrol"], 0.35, 0.8),
        "XUV300":   _m(10.0, 1197, 110, 5, ["Petrol", "Diesel"], 0.25, 0.8),
    },
    "Honda": {
        "City":  _m(13.0, 1498, 119, 5, ["Petrol"], 0.45, 1.2),
        "Amaze": _m(8.0, 1199, 89, 5, ["Petrol", "Diesel"], 0.30, 0.9),
    },
    "Toyota": {
        "Innova Crysta": _m(23.0, 2393, 148, 7, ["Diesel", "Petrol"], 0.25, 1.0),
        "Fortuner":      _m(38.0, 2755, 201, 7, ["Diesel", "Petrol"], 0.60, 0.6),
        "Glanza":        _m(8.0, 1197, 88, 5, ["Petrol", "CNG"], 0.30, 0.6),
    },
    "Kia": {
        "Seltos": _m(15.0, 1497, 113, 5, ["Petrol", "Diesel"], 0.45, 1.0),
        "Sonet":  _m(10.0, 1197, 82, 5, ["Petrol", "Diesel"], 0.30, 0.8),
        "Carens": _m(12.0, 1497, 113, 7, ["Petrol", "Diesel"], 0.35, 0.5),
    },
    "Volkswagen": {
        "Polo":   _m(8.5, 999, 108, 5, ["Petrol"], 0.35, 0.6),
        "Virtus": _m(14.0, 999, 113, 5, ["Petrol"], 0.50, 0.4),
    },
    "Skoda": {
        "Slavia": _m(14.0, 999, 113, 5, ["Petrol"], 0.50, 0.4),
        "Kushaq": _m(15.0, 999, 113, 5, ["Petrol"], 0.50, 0.5),
    },
    "Renault": {
        "Kwid":  _m(5.0, 999, 67, 5, ["Petrol"], 0.20, 0.8),
        "Kiger": _m(8.0, 999, 72, 5, ["Petrol"], 0.30, 0.5),
    },
    "Ford": {
        "EcoSport": _m(10.5, 1497, 121, 5, ["Petrol", "Diesel"], 0.30, 0.6),
    },
}

# how well a brand holds resale value
BRAND_RETENTION = {
    "Toyota": 1.08, "Maruti Suzuki": 1.04, "Mahindra": 1.03, "Honda": 1.02,
    "Hyundai": 1.00, "Kia": 1.00, "Tata": 0.95, "Volkswagen": 0.90,
    "Skoda": 0.90, "Renault": 0.88, "Ford": 0.80,
}
FUEL_NEW_PREMIUM = {"Petrol": 1.0, "Diesel": 1.12, "CNG": 0.97, "Electric": 1.50}
FUEL_KM_PER_YEAR = {"Petrol": 10500, "Diesel": 15000, "CNG": 13500, "Electric": 9000}


def generate(n=15000, seed=42):
    rng = np.random.default_rng(seed)
    entries = [(b, m, s) for b, ms in CATALOG.items() for m, s in ms.items()]
    weights = np.array([s["w"] for _, _, s in entries], dtype=float)
    weights /= weights.sum()

    ages = np.arange(0, 16)
    age_p = np.exp(-0.17 * ages)
    age_p /= age_p.sum()

    rows = []
    for _ in range(n):
        brand, model, s = entries[rng.choice(len(entries), p=weights)]

        fw = np.array([{"Petrol": 0.55, "Diesel": 0.30, "CNG": 0.12, "Electric": 0.10}[f] for f in s["fuels"]])
        fuel = s["fuels"][rng.choice(len(fw), p=fw / fw.sum())]
        trans = "Automatic" if rng.random() < s["auto"] else "Manual"
        owner = int(rng.choice([1, 2, 3], p=[0.70, 0.22, 0.08]))
        age = int(rng.choice(ages, p=age_p))
        year = REF_YEAR - age

        rate = FUEL_KM_PER_YEAR[fuel] * rng.lognormal(0, 0.28)
        km = max(500.0, (age * rate + 1500) * rng.lognormal(0, 0.18))
        km = float(min(km, 320000))

        # --- ground-truth pricing process (hidden from the model) ---
        new_price = s["price"] * FUEL_NEW_PREMIUM[fuel] * (1.10 if trans == "Automatic" else 1.0)
        new_price *= rng.lognormal(0, 0.06)  # unobserved trim / variant

        retention = 0.86 * np.exp(-0.085 * age)
        retention *= BRAND_RETENTION[brand]
        excess = (km - age * FUEL_KM_PER_YEAR[fuel]) / 100000.0
        retention *= np.exp(-0.35 * excess)
        retention *= {1: 1.0, 2: 0.92, 3: 0.84}[owner]
        if fuel == "Electric":
            retention *= np.exp(-0.035 * age)
        if fuel == "Diesel" and age > 9:
            retention *= 0.88  # 10-year diesel norms
        if fuel == "CNG":
            retention *= 0.96

        price = max(0.35, new_price * retention * rng.lognormal(0, 0.045))

        bhp = 127 if fuel == "Electric" else s["bhp"]
        cc = 0 if fuel == "Electric" else s["cc"]
        rows.append((brand, model, year, round(km), fuel, trans, owner, cc, bhp,
                     s["seats"], s["price"], round(price, 2)))

    return pd.DataFrame(rows, columns=[
        "brand", "model", "year", "km_driven", "fuel", "transmission", "owner",
        "engine_cc", "power_bhp", "seats", "showroom_price_lakh", "price_lakh"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join("data", "used_cars.csv"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df = generate(a.rows, a.seed)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df):,} rows -> {a.out}")
    print(df.describe().round(2).T[["mean", "min", "max"]])
