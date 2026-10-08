"""Portable JSON scaler state: reuse local preprocessing without shipping pickle files."""

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from taxi.data import FEATURES


def scaler_state(scaler):
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "variance": scaler.var_.tolist(),
        "samples_seen": int(scaler.n_samples_seen_),
    }


def restore_scaler(snapshot):
    features = snapshot["features"]
    if not features or len(set(features)) != len(features) or not set(features) <= set(FEATURES):
        raise ValueError("Invalid frozen feature schema")
    scaler = StandardScaler()
    state = snapshot["scaler"]
    scaler.mean_ = np.asarray(state["mean"], dtype="float64")
    scaler.scale_ = np.asarray(state["scale"], dtype="float64")
    scaler.var_ = np.asarray(state["variance"], dtype="float64")
    for values in [scaler.mean_, scaler.scale_, scaler.var_]:
        if values.shape != (len(features),) or not np.isfinite(values).all():
            raise ValueError("Invalid frozen scaler values")
    if (scaler.scale_ <= 0).any() or (scaler.var_ < 0).any():
        raise ValueError("Invalid frozen scaler scale/variance")
    scaler.n_features_in_ = len(features)
    scaler.n_samples_seen_ = int(state["samples_seen"])
    return scaler


def export_snapshot(scaler, features, k, rules, provenance, destination):
    snapshot = {
        "version": 1,
        "features": features,
        "k": k,
        "rules": rules,
        "seed": 42,
        "scaler": scaler_state(scaler),
        "source_sha256": provenance["sha256"],
        "cleaned_rows": provenance["accepted_rows"],
    }
    Path(destination).write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot
