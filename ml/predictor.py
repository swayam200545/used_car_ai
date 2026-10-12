"""
Loads the trained network and turns a web request into a rich valuation.

For every request we run ONE batched forward pass that contains:
  * 60 stochastic passes (Monte-Carlo Dropout) -> how unsure is the network?
  * the same car at every model year           -> year-wise price trend
  * the same car at many odometer readings     -> effect of kilometres
"""
import json
import os
import threading

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from tensorflow import keras

from ml.features import OWNER_LABELS, Preprocessor
from ml.advisory import evaluate_advisory, DEFECT_COSTS

MC_PASSES = 60
Z80 = 1.2816  # 80 % interval


def inr(n):
    """Indian digit grouping: 1234567 -> 12,34,567"""
    s = str(int(round(n)))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


class ModelNotTrained(RuntimeError):
    pass


class ValidationError(ValueError):
    def __init__(self, field, message):
        super().__init__(message)
        self.field, self.message = field, message


class Predictor:
    def __init__(self, artifacts_dir="artifacts"):
        model_path = os.path.join(artifacts_dir, "model.keras")
        pp_path = os.path.join(artifacts_dir, "preprocess.json")
        if not (os.path.exists(model_path) and os.path.exists(pp_path)):
            raise ModelNotTrained("Model not trained yet. Run:  python -m ml.train")
        self.model = keras.models.load_model(model_path, compile=False)
        self.pp = Preprocessor.load(pp_path)
        self.metrics = {}
        mp = os.path.join(artifacts_dir, "metrics.json")
        if os.path.exists(mp):
            with open(mp) as f:
                self.metrics = json.load(f)
        self.resid_std = self.metrics.get("resid_std_log", 0.08)
        self.mc_median = self.metrics.get("mc_std_median")
        self._lock = threading.Lock()  # keras models are not guaranteed thread-safe

    # ----------------------------------------------------------- validation
    def _validate(self, p):
        brand, model = str(p.get("brand", "")).strip(), str(p.get("model", "")).strip()
        key = f"{brand}|{model}"
        if key not in self.pp.specs:
            raise ValidationError("model", "Choose a brand and model from the list.")
        spec = self.pp.specs[key]

        fuel = str(p.get("fuel", ""))
        if fuel not in spec["fuels"]:
            raise ValidationError("fuel", f"{brand} {model} is not sold with {fuel or 'that'} fuel.")

        trans = str(p.get("transmission", ""))
        if trans not in self.pp.transmissions:
            raise ValidationError("transmission", "Choose Manual or Automatic.")

        try:
            year = int(p.get("year"))
        except (TypeError, ValueError):
            raise ValidationError("year", "Enter the model year as a number.")
        if not (self.pp.min_year <= year <= self.pp.ref_year):
            raise ValidationError("year", f"Year must be between {self.pp.min_year} and {self.pp.ref_year}.")

        try:
            km = float(p.get("km"))
        except (TypeError, ValueError):
            raise ValidationError("km", "Enter the kilometres driven as a number.")
        if not (0 <= km <= 400000):
            raise ValidationError("km", "Kilometres must be between 0 and 4,00,000.")

        try:
            owner = int(p.get("owner", 1))
        except (TypeError, ValueError):
            owner = 0
        if owner not in (1, 2, 3):
            raise ValidationError("owner", "Choose 1st, 2nd or 3rd+ owner.")

        asking = p.get("asking")
        if asking in (None, ""):
            asking = None
        else:
            try:
                asking = float(asking)
            except (TypeError, ValueError):
                raise ValidationError("asking", "Asking price must be a number.")
            if asking <= 0:
                asking = None
        return dict(brand=brand, model=model, fuel=fuel, transmission=trans,
                    year=year, km=km, owner=owner, asking=asking, spec=spec)

    # ------------------------------------------------------------- building
    def _row(self, v, year=None, km=None):
        s = v["spec"]["fuels"][v["fuel"]]
        return dict(brand=v["brand"], model=v["model"], year=year if year is not None else v["year"],
                    km_driven=km if km is not None else v["km"], fuel=v["fuel"],
                    transmission=v["transmission"], owner=v["owner"],
                    engine_cc=s["cc"], power_bhp=s["bhp"], seats=s["seats"],
                    showroom_price_lakh=v["spec"]["showroom"])

    # -------------------------------------------------------------- predict
    def predict(self, payload):
        v = self._validate(payload)
        ref = self.pp.ref_year
        age = max(0, ref - v["year"])

        # usage pattern -> used to draw the year-wise curve for "a car driven like yours"
        rate = float(np.clip(v["km"] / age, 4000, 30000)) if age > 0 else 11000.0
        years = list(range(self.pp.min_year, ref + 1))
        year_rows = [self._row(v, year=y, km=max(500.0, rate * (ref - y))) for y in years]

        km_top = max(200000.0, v["km"] * 1.15)
        km_points = np.linspace(2000, km_top, 24)
        km_rows = [self._row(v, km=float(k)) for k in km_points]

        rows = [self._row(v)] + year_rows + km_rows
        det = self.pp.transform(pd.DataFrame(rows))

        mc_in = self.pp.transform(pd.DataFrame([self._row(v)] * MC_PASSES))

        with self._lock:
            det_log = self.model.predict(det, verbose=0, batch_size=256).ravel()
            mc_log = self.model(mc_in, training=True).numpy().ravel()  # dropout stays ON

        point_log = float(det_log[0])
        mc = float(mc_log.std())
        if self.mc_median:   # calibrated: typical error, widened/narrowed by this car's own uncertainty
            spread = self.resid_std * float(np.clip(mc / self.mc_median, 0.6, 1.8))
        else:
            spread = float(np.sqrt(mc ** 2 + self.resid_std ** 2))
        price = float(np.exp(point_log))
        low, high = np.exp(point_log - Z80 * spread), np.exp(point_log + Z80 * spread)

        round1k = lambda lakh: int(round(lakh * 100000 / 1000.0) * 1000)
        price_inr, low_inr, high_inr = round1k(price), round1k(low), round1k(high)

        year_vals = np.exp(det_log[1:1 + len(years)])
        # "nearly new" = the network's value for this exact car if it were this year's model
        new_lakh = float(year_vals[-1])
        km_vals = np.exp(det_log[1 + len(years):])

        advisory = evaluate_advisory(
            car_specs={
                "brand": v["brand"],
                "model": v["model"],
                "year": v["year"],
                "km": v["km"],
                "fuel": v["fuel"],
                "transmission": v["transmission"],
                "owner": OWNER_LABELS[v["owner"] - 1],
                "new_lakh": new_lakh,
            },
            price_pred=price_inr,
            price_low=low_inr,
            price_high=high_inr,
            asking_price=v["asking"],
            image_analysis=payload.get("image_analysis"),
            inspection_flags=payload.get("inspection_flags", []),
        )

        result = {
            "price": price_inr, "low": low_inr, "high": high_inr,
            "uncertainty_pct": round(spread * Z80 * 100, 1),
            "age": age,
            "new_lakh": round(new_lakh, 2),
            "retained_pct": round(min(price / new_lakh, 1.0) * 100, 1),
            "loss_per_year": round1k(max(new_lakh - price, 0) / age) if age else 0,
            "car": {"brand": v["brand"], "model": v["model"], "year": v["year"], "km": int(v["km"]),
                    "fuel": v["fuel"], "transmission": v["transmission"],
                    "owner": OWNER_LABELS[v["owner"] - 1],
                    "specs": v["spec"]["fuels"][v["fuel"]]},
            "curve_year": {"labels": years, "values": [round(float(x), 2) for x in year_vals],
                           "current": v["year"], "km_per_year": int(rate)},
            "curve_km": {"labels": [int(k) for k in km_points],
                         "values": [round(float(x), 2) for x in km_vals], "current": int(v["km"])},
            "deal": self._deal(v["asking"], price_inr, low_inr, high_inr),
            "advisory": advisory,
        }
        return result

    @staticmethod
    def _deal(asking, price, low, high):
        if not asking:
            return None
        diff = (asking - price) / price * 100
        if asking < low:
            if diff < -25:
                verdict, msg = "check", "Verify the condition, service history and papers before paying."
            else:
                verdict, msg = "great", "A strong deal if the car's condition and papers check out."
        elif asking <= high:
            verdict, msg = "fair", "Buyer and seller should both be comfortable at this price."
        elif diff <= 15:
            verdict, msg = "high", f"Offer around ₹{inr(price)}."
        else:
            verdict, msg = "over", f"Try negotiating towards ₹{inr(price)}."
        return {"asking": int(asking), "diff_pct": round(diff, 1), "verdict": verdict, "message": msg}

    # --------------------------------------------------------------- meta
    def meta(self):
        return {
            "catalog": self.pp.catalog(),
            "transmissions": self.pp.transmissions,
            "owners": [{"value": i + 1, "label": l} for i, l in enumerate(OWNER_LABELS)],
            "min_year": self.pp.min_year, "max_year": self.pp.ref_year,
            "inspection_catalog": DEFECT_COSTS,
        }
