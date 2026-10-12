"""
Feature engineering shared by training (ml/train.py) and serving (ml/predictor.py).

Categorical inputs  -> integer ids -> learned EMBEDDINGS inside the network
Numeric inputs      -> standardised (z-score)

Keeping this in one place guarantees the web app preprocesses a request exactly
the way the training data was preprocessed.
"""
import json

import numpy as np
import pandas as pd

NUM_COLS = ["age", "log_km", "engine_cc", "power_bhp", "seats", "log_showroom"]
CAT_COLS = ["brand", "model", "fuel", "transmission", "owner"]
OWNER_LABELS = ["1st owner", "2nd owner", "3rd+ owner"]


class Preprocessor:
    def __init__(self, ref_year=2026):
        self.ref_year = ref_year
        self.brands, self.models, self.fuels, self.transmissions = [], [], [], []
        self.mean, self.std = {}, {}
        self.specs = {}      # "Brand|Model" -> {"showroom": x, "fuels": {fuel: {cc,bhp,seats}}}
        self.min_year = self.ref_year - 15

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame):
        df = self._fill_specs(df.copy())
        df["key"] = df["brand"] + "|" + df["model"]
        self.brands = sorted(df["brand"].unique())
        self.models = sorted(df["key"].unique())
        self.fuels = sorted(df["fuel"].unique())
        self.transmissions = sorted(df["transmission"].unique())
        self.min_year = int(df["year"].min())

        for key, g in df.groupby("key"):
            fuels = {}
            for fuel, gf in g.groupby("fuel"):
                fuels[fuel] = {
                    "cc": int(gf["engine_cc"].median()),
                    "bhp": int(gf["power_bhp"].median()),
                    "seats": int(gf["seats"].median()),
                }
            self.specs[key] = {
                "showroom": float(g["showroom_price_lakh"].median()),
                "fuels": fuels,
                "transmissions": sorted(g["transmission"].unique()),
            }

        num = self._numeric(df)
        for i, c in enumerate(NUM_COLS):
            self.mean[c] = float(num[:, i].mean())
            self.std[c] = float(num[:, i].std() or 1.0)
        return self

    @staticmethod
    def _fill_specs(df):
        """Real-world CSVs often lack spec columns - impute so training never breaks."""
        for col, default in [("engine_cc", 1200), ("power_bhp", 90), ("seats", 5)]:
            if col not in df.columns:
                df[col] = np.nan
            grp = df.groupby(["brand", "model", "fuel"])[col].transform("median")
            df[col] = df[col].fillna(grp).fillna(df[col].median()).fillna(default)
        if "showroom_price_lakh" not in df.columns:
            df["showroom_price_lakh"] = np.nan
        grp = (df.groupby(["brand", "model"])["price_lakh"].transform("median") * 1.8
               if "price_lakh" in df.columns else None)
        if grp is not None:
            df["showroom_price_lakh"] = df["showroom_price_lakh"].fillna(grp)
        df["showroom_price_lakh"] = df["showroom_price_lakh"].fillna(df["showroom_price_lakh"].median())
        return df

    # ------------------------------------------------------------ transform
    def _numeric(self, df):
        age = (self.ref_year - df["year"]).clip(lower=0)
        return np.column_stack([
            age,
            np.log1p(df["km_driven"].clip(lower=0)),
            df["engine_cc"],
            df["power_bhp"],
            df["seats"],
            np.log(df["showroom_price_lakh"]),
        ]).astype("float32")

    def transform(self, df: pd.DataFrame):
        df = df.copy()
        df["key"] = df["brand"] + "|" + df["model"]
        num = self._numeric(df)
        for i, c in enumerate(NUM_COLS):
            num[:, i] = (num[:, i] - self.mean[c]) / self.std[c]
        return {
            "brand": np.array([self.brands.index(b) for b in df["brand"]], dtype="int32")[:, None],
            "model": np.array([self.models.index(k) for k in df["key"]], dtype="int32")[:, None],
            "fuel": np.array([self.fuels.index(f) for f in df["fuel"]], dtype="int32")[:, None],
            "transmission": np.array([self.transmissions.index(t) for t in df["transmission"]], dtype="int32")[:, None],
            "owner": (df["owner"].clip(1, 3).astype(int).values - 1).astype("int32")[:, None],
            "num": num,
        }

    # -------------------------------------------------------------- catalog
    def catalog(self):
        """Structure the frontend uses to build dependent dropdowns."""
        out = {}
        for key, spec in self.specs.items():
            brand, model = key.split("|", 1)
            out.setdefault(brand, {})[model] = spec
        return out

    # ----------------------------------------------------------- persistence
    def to_json(self):
        return {
            "ref_year": self.ref_year, "min_year": self.min_year,
            "brands": self.brands, "models": self.models, "fuels": self.fuels,
            "transmissions": self.transmissions, "mean": self.mean, "std": self.std,
            "specs": self.specs,
        }

    @classmethod
    def from_json(cls, d):
        p = cls(d["ref_year"])
        p.__dict__.update({k: d[k] for k in
                           ["min_year", "brands", "models", "fuels", "transmissions", "mean", "std", "specs"]})
        return p

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_json(json.load(f))
