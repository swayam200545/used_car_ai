"""
Trains the DEEP LEARNING price model.

    python -m ml.train                     # uses data/used_cars.csv
    python -m ml.train --data my_cars.csv  # your own dataset (see README for columns)

Outputs (in artifacts/):
    model.keras        trained neural network
    preprocess.json    encoders, scaling stats, model catalogue
    metrics.json       scores, baselines, loss curves, feature importance (shown in the UI)

NETWORK
    brand / model / fuel / transmission / owner  --> Embedding layers (learned vectors)
    age, log(km), cc, bhp, seats, log(showroom)  --> standardised numbers
    concat --> Dense(256) --> Dense(128) --> Dense(64) --> Dense(1)  (ReLU + Dropout)
    target = log(price): prices span 1 - 45 lakh, log keeps errors proportional.

    Dropout is also used at prediction time (Monte-Carlo Dropout) to estimate
    how unsure the network is - this becomes the price range in the UI.
"""
import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers

from ml.features import Preprocessor

SEED = 7
ART = "artifacts"


def build_model(pp: Preprocessor) -> keras.Model:
    def cat_input(name, size, dim):
        inp = keras.Input(shape=(1,), dtype="int32", name=name)
        emb = layers.Embedding(size, dim, name=f"emb_{name}")(inp)
        return inp, layers.Flatten()(emb)

    ins, embs = zip(*[
        cat_input("brand", len(pp.brands), 4),
        cat_input("model", len(pp.models), 10),
        cat_input("fuel", len(pp.fuels), 3),
        cat_input("transmission", len(pp.transmissions), 2),
        cat_input("owner", 3, 2),
    ])
    num = keras.Input(shape=(6,), name="num")

    x = layers.Concatenate()([*embs, num])
    for units, drop in [(256, 0.15), (128, 0.15), (64, 0.10)]:
        x = layers.Dense(units, activation="relu",
                         kernel_regularizer=regularizers.l2(1e-5))(x)
        x = layers.Dropout(drop)(x)
    out = layers.Dense(1, name="log_price")(x)

    model = keras.Model([*ins, num], out, name="used_car_price_dnn")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def scores(y_true, y_pred):
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "r2": round(1 - ss_res / ss_tot, 4),
        "mae_lakh": round(float(np.mean(np.abs(err))), 3),
        "rmse_lakh": round(float(np.sqrt(np.mean(err ** 2))), 3),
        "mape_pct": round(float(np.mean(np.abs(err) / y_true) * 100), 2),
        "within_10pct": round(float(np.mean(np.abs(err) / y_true <= 0.10) * 100), 1),
    }


def mc_std(model, X, passes=30):
    """Monte-Carlo Dropout: spread of predictions with dropout left ON."""
    preds = np.stack([model(X, training=True).numpy().ravel() for _ in range(passes)])
    return preds.std(axis=0)


Z80 = 1.2816


def baseline_frame(pp, df):
    out = pd.DataFrame({
        "brand": df["brand"], "model": df["brand"] + " " + df["model"],
        "fuel": df["fuel"], "transmission": df["transmission"], "owner": df["owner"].astype(str),
        "age": (pp.ref_year - df["year"]).clip(lower=0), "km": df["km_driven"],
        "cc": df["engine_cc"], "bhp": df["power_bhp"], "seats": df["seats"],
        "showroom": df["showroom_price_lakh"],
    })
    return out


def permutation_importance(model, pp, df_test, inputs, y_log, repeats=5):
    """How much does R2 (log space) fall if a feature group is scrambled?"""
    rng = np.random.default_rng(SEED)
    base_pred = model.predict(inputs, verbose=0).ravel()
    base = 1 - np.sum((base_pred - y_log) ** 2) / np.sum((y_log - y_log.mean()) ** 2)

    groups = {
        "Model & brand": ["brand", "model", "num:2,3,4,5"],
        "Age (year)": ["num:0"],
        "Kilometres driven": ["num:1"],
        "Fuel type": ["fuel"],
        "Transmission": ["transmission"],
        "Ownership": ["owner"],
    }
    drops = {}
    for name, parts in groups.items():
        vals = []
        for _ in range(repeats):
            perm = rng.permutation(len(y_log))
            shuffled = {k: v.copy() for k, v in inputs.items()}
            for p in parts:
                if p.startswith("num:"):
                    cols = [int(c) for c in p[4:].split(",")]
                    shuffled["num"][:, cols] = inputs["num"][perm][:, cols]
                else:
                    shuffled[p] = inputs[p][perm]
            pred = model.predict(shuffled, verbose=0).ravel()
            r2 = 1 - np.sum((pred - y_log) ** 2) / np.sum((y_log - y_log.mean()) ** 2)
            vals.append(base - r2)
        drops[name] = max(float(np.mean(vals)), 0.0)
    total = sum(drops.values()) or 1.0
    return [{"feature": k, "importance": round(v / total * 100, 1)}
            for k, v in sorted(drops.items(), key=lambda kv: -kv[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "used_cars.csv"))
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ref-year", type=int, default=2026)
    a = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    os.makedirs(ART, exist_ok=True)

    df = pd.read_csv(a.data)
    df = df.dropna(subset=["brand", "model", "year", "km_driven", "fuel", "transmission", "owner", "price_lakh"])
    df = df[(df["price_lakh"] > 0) & (df["km_driven"] >= 0)].reset_index(drop=True)
    print(f"Loaded {len(df):,} rows from {a.data}")

    train_df, test_df = train_test_split(df, test_size=0.15, random_state=SEED)
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=SEED)

    pp = Preprocessor(a.ref_year).fit(train_df)
    pp_full_df = Preprocessor._fill_specs(df.copy())  # same imputation for every split
    train_df, val_df, test_df = (pp_full_df.loc[d.index] for d in (train_df, val_df, test_df))

    X_tr, X_va, X_te = (pp.transform(d) for d in (train_df, val_df, test_df))
    y_tr, y_va, y_te = (np.log(d["price_lakh"].values.astype("float32")) for d in (train_df, val_df, test_df))

    model = build_model(pp)
    model.summary(print_fn=lambda s: print("  " + s))

    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5),
    ]
    t0 = time.time()
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=a.epochs,
                     batch_size=a.batch, callbacks=cbs, verbose=2)
    train_secs = time.time() - t0
    epochs_run = len(hist.history["loss"])
    print(f"Trained {epochs_run} epochs in {train_secs:.0f}s")

    # ---------------------------------------------------------- evaluation
    y_true = test_df["price_lakh"].values
    dl_pred = np.exp(model.predict(X_te, verbose=0).ravel())
    results = {"Deep neural network": scores(y_true, dl_pred)}

    base_tr, base_te = baseline_frame(pp, train_df), baseline_frame(pp, test_df)
    cat = ["brand", "model", "fuel", "transmission", "owner"]
    num = ["age", "km", "cc", "bhp", "seats", "showroom"]
    ct = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")

    lin = make_pipeline(ct, LinearRegression()).fit(base_tr, train_df["price_lakh"])
    results["Linear regression"] = scores(y_true, np.clip(lin.predict(base_te), 0.3, None))

    rf = make_pipeline(ct, RandomForestRegressor(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=SEED))
    rf.fit(base_tr, np.log(train_df["price_lakh"]))
    results["Random forest"] = scores(y_true, np.exp(rf.predict(base_te)))

    for k, v in results.items():
        print(f"  {k:22s} R2={v['r2']:.4f}  MAE={v['mae_lakh']:.3f}L  MAPE={v['mape_pct']:.1f}%")

    # residual spread (log space) on validation set - used for the price range in the UI
    val_pred = model.predict(X_va, verbose=0).ravel()
    resid_std = float(np.std(val_pred - y_va))

    # --- calibrate the price range: validation residual spread, scaled by per-car MC-dropout uncertainty
    mc_val = mc_std(model, X_va)
    mc_median = float(np.median(mc_val))
    mc_te = mc_std(model, X_te)
    te_log_err = np.abs(np.log(dl_pred) - np.log(y_true))
    half = Z80 * resid_std * np.clip(mc_te / mc_median, 0.6, 1.8)
    coverage = float(np.mean(te_log_err <= half) * 100)
    print(f"  80% price range covers {coverage:.1f}% of test cars; median half-width +/-{(np.exp(np.median(half))-1)*100:.1f}%")

    importance = permutation_importance(model, pp, test_df, X_te, np.log(y_true))

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y_true), size=min(220, len(y_true)), replace=False)
    scatter = [[round(float(y_true[i]), 2), round(float(dl_pred[i]), 2)] for i in idx]

    arch = [
        {"name": "Inputs", "detail": "5 embeddings + 6 numeric features", "units": 4 + 10 + 3 + 2 + 2 + 6},
        {"name": "Dense 1", "detail": "ReLU · Dropout 0.15", "units": 256},
        {"name": "Dense 2", "detail": "ReLU · Dropout 0.15", "units": 128},
        {"name": "Dense 3", "detail": "ReLU · Dropout 0.10", "units": 64},
        {"name": "Output", "detail": "log(price)", "units": 1},
    ]

    metrics = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "dataset": {"rows": len(df), "train": len(train_df), "val": len(val_df), "test": len(test_df),
                    "source": os.path.basename(a.data)},
        "params": int(model.count_params()),
        "epochs_run": epochs_run,
        "train_seconds": round(train_secs),
        "resid_std_log": round(resid_std, 4),
        "mc_std_median": round(mc_median, 5),
        "interval_coverage": round(coverage, 1),
        "results": results,
        "history": {k: [round(float(x), 5) for x in v] for k, v in hist.history.items()
                    if k in ("loss", "val_loss")},
        "importance": importance,
        "scatter": scatter,
        "architecture": arch,
    }

    model.save(os.path.join(ART, "model.keras"))
    pp.save(os.path.join(ART, "preprocess.json"))
    with open(os.path.join(ART, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    print("Saved artifacts/ (model.keras, preprocess.json, metrics.json)")


if __name__ == "__main__":
    main()
