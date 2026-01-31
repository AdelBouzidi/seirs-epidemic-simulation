#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data/part2_multi_agent")
FIG_DIR = Path("figures/part2")
FIG_DIR.mkdir(parents=True, exist_ok=True)

py = pd.read_csv(DATA_DIR / "python_mean_3reps.csv")
cpp = pd.read_csv(DATA_DIR / "cpp_mean_3reps.csv")

# Vérifs légères
for df, name in [(py, "python"), (cpp, "cpp")]:
    expected = ["t", "S_mean", "E_mean", "I_mean", "R_mean"]
    if list(df.columns) != expected:
        raise ValueError(f"Colonnes inattendues dans {name}: {list(df.columns)}")

if not (py["t"].values == cpp["t"].values).all():
    raise ValueError("Les colonnes t ne coïncident pas entre python et cpp.")

# Figure principale : comparaison sur I (le plus pertinent épidémiologiquement)
plt.figure()
plt.plot(py["t"], py["I_mean"], label="I moyenne (Python, 3 rép.)")
plt.plot(cpp["t"], cpp["I_mean"], label="I moyenne (C++, 3 rép.)")
plt.xlabel("Temps (jours)")
plt.ylabel("Nombre d'agents infectieux (moyenne)")
plt.title("Partie 2 — Comparaison Python vs C++ (moyennes sur 3 réplications)")
plt.legend()
plt.tight_layout()

out_fig = FIG_DIR / "compare_python_cpp_Imean.png"
plt.savefig(out_fig, dpi=180)
plt.close()

print("OK ->", out_fig)
