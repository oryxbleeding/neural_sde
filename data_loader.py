"""
Lädt Sterbetafeldaten (Perioden- oder Kohortentafel) und erzeugt empirische
Todesalter-Stichproben für das Neural-SDE-Training.

Autor: Daniel Alija, 2025
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import yaml


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #
def load_config(path: str | Path = "config.yaml") -> dict:
    """Lädt die YAML-Konfigurationsdatei."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_life_table(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Erwartet CSV-Datei mit Spalten:
        Age, dx   (Sterbefälle von 100 000)
    Gibt zurück:
        ages  - Altersarray (int)
        probs - Sterbe-Wahrscheinlichkeiten (dx / Σdx)
    """
    df = pd.read_csv(csv_path, sep=";")
    ages: np.ndarray = df["Age"].to_numpy(dtype=int)
    deaths: np.ndarray = df["dx"].to_numpy(dtype=float)
    probs = deaths / deaths.sum()
    return ages, probs


def sample_lifespans(
    ages: np.ndarray, probs: np.ndarray, n_samples: int, seed: int = 42
) -> np.ndarray:
    """Zieht Todesalter gemäß gegebener diskreter Verteilung."""
    rng = np.random.default_rng(seed)
    return rng.choice(ages, size=n_samples, p=probs)


# --------------------------------------------------------------------------- #
# Convenience-Wrapper
# --------------------------------------------------------------------------- #
def get_empirical_sample(cfg: dict) -> np.ndarray:
    """Lädt Lebensdauern entsprechend den Angaben in config.yaml."""
    ages, probs = load_life_table(cfg["data"]["life_table_csv"])
    n = cfg["data"]["n_empirical"]
    return sample_lifespans(ages, probs, n)