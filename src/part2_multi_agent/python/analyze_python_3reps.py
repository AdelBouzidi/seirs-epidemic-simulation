#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data/part2_multi_agent")
FIG_DIR = Path("figures/part2")
FIG_DIR.mkdir(parents=True, exist_ok=True)

files = [
    DATA_DIR / "python_rep01.csv",
    DATA_DIR / "python_rep02.csv",
    DATA_DIR / "python_rep03.csv",
]

# Charger et vérifier
dfs = []
for f in files:
    df = pd.read_csv(f)
    if list(df.columns) != ["t", "S", "E", "I", "R"]:
        raise ValueError(f"Colonnes inattendues dans {f}: {list(df.columns)}")
    dfs.append(df)

# Fusionner par t et calculer moyenne
merged = dfs[0][["t"]].copy()
for k, df in enumerate(dfs, start=1):
    merged[f"S{k}"] = df["S"].values
    merged[f"E{k}"] = df["E"].values
    merged[f"I{k}"] = df["I"].values
    merged[f"R{k}"] = df["R"].values

mean_df = pd.DataFrame({
    "t": merged["t"],
    "S_mean": merged[[f"S{k}" for k in range(1, 4)]].mean(axis=1),
    "E_mean": merged[[f"E{k}" for k in range(1, 4)]].mean(axis=1),
    "I_mean": merged[[f"I{k}" for k in range(1, 4)]].mean(axis=1),
    "R_mean": merged[[f"R{k}" for k in range(1, 4)]].mean(axis=1),
})

# Sauvegarder le CSV de moyenne
out_csv = DATA_DIR / "python_mean_3reps.csv"
mean_df.to_csv(out_csv, index=False)

# Tracer la moyenne
plt.figure()
plt.plot(mean_df["t"], mean_df["S_mean"], label="S (moyenne)")
plt.plot(mean_df["t"], mean_df["E_mean"], label="E (moyenne)")
plt.plot(mean_df["t"], mean_df["I_mean"], label="I (moyenne)")
plt.plot(mean_df["t"], mean_df["R_mean"], label="R (moyenne)")
plt.xlabel("Temps (jours)")
plt.ylabel("Nombre d'agents")
plt.title("Partie 2 (Python) — Moyenne sur 3 réplications")
plt.legend()
plt.tight_layout()

out_fig = FIG_DIR / "python_mean_3reps.png"
plt.savefig(out_fig, dpi=180)
plt.close()

print("OK ->", out_csv)
print("OK ->", out_fig)
