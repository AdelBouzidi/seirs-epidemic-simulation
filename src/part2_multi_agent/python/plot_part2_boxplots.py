#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA = Path("data/part2_multi_agent")
FIG = Path("figures/part2")
FIG.mkdir(parents=True, exist_ok=True)

# Charger les données
py = pd.read_csv(DATA / "python_peaks.csv")
cpp = pd.read_csv(DATA / "cpp_peaks.csv")
c = pd.read_csv(DATA / "c_peaks.csv")

# Préparer les données
labels = ["Python", "C++", "C"]

peak_I_data = [
    py["peak_I"].values,
    cpp["peak_I"].values,
    c["peak_I"].values,
]

day_peak_data = [
    py["day_peak"].values,
    cpp["day_peak"].values,
    c["day_peak"].values,
]

# Figure
plt.figure(figsize=(10, 4))

# --- Boxplot peak_I ---
plt.subplot(1, 2, 1)
plt.boxplot(peak_I_data, labels=labels, showfliers=True)
plt.ylabel("Hauteur du premier pic infectieux")
plt.title("Distribution de peak_I")

# --- Boxplot day_peak ---
plt.subplot(1, 2, 2)
plt.boxplot(day_peak_data, labels=labels, showfliers=True)
plt.ylabel("Jour du premier pic infectieux")
plt.title("Distribution de day_peak")

plt.tight_layout()
out = FIG / "part2_boxplots_peaks.png"
plt.savefig(out, dpi=150)
plt.close()

print("OK ->", out)
