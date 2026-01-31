#!/usr/bin/env python3
from pathlib import Path
import glob
import pandas as pd
import matplotlib.pyplot as plt

N = 20000
TMAX = 100  # 0..100 jours (comme l'exemple)
DATA = Path("data/part2_multi_agent")
FIG = Path("figures/part2")
FIG.mkdir(parents=True, exist_ok=True)

# Fichiers de moyennes déjà existants (3 reps)
py_mean_path = DATA / "python_mean_3reps.csv"
cpp_mean_path = DATA / "cpp_mean_3reps.csv"

# Dossier des 30 runs C
c_runs_glob = str(DATA / "c_runs" / "c_rep*.csv")

def load_mean_I_from_file(path: Path, col_I: str):
    df = pd.read_csv(path)
    df = df[df["t"] <= TMAX].copy()
    return df["t"].to_numpy(), (df[col_I].to_numpy() / N)

def compute_c_mean_I_from_runs():
    files = sorted(glob.glob(c_runs_glob))
    if len(files) == 0:
        raise FileNotFoundError("Aucun fichier trouvé dans data/part2_multi_agent/c_runs/")
    if len(files) != 30:
        print(f"[WARN] {len(files)} fichiers détectés (attendu: 30).")

    # Charger tous les runs et empiler I(t)
    Is = []
    t_ref = None
    for f in files:
        df = pd.read_csv(f)
        df = df[df["t"] <= TMAX].copy()
        if t_ref is None:
            t_ref = df["t"].to_numpy()
        else:
            # sécurité: s'assurer que t est aligné
            if not (df["t"].to_numpy() == t_ref).all():
                raise ValueError(f"Temps t non aligné dans {f}")
        Is.append(df["I"].to_numpy())

    I_mean = (pd.DataFrame(Is).mean(axis=0).to_numpy()) / N
    return t_ref, I_mean

def main():
    # Python mean (3 reps)
    t_py, I_py = load_mean_I_from_file(py_mean_path, "I_mean")

    # C++ mean (3 reps)
    t_cpp, I_cpp = load_mean_I_from_file(cpp_mean_path, "I_mean")

    # C mean (30 reps) calculé depuis les 30 CSV
    t_c, I_c = compute_c_mean_I_from_runs()

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(t_py, I_py, label="Python (moyenne 3)")
    plt.plot(t_cpp, I_cpp, label="C++ (moyenne 3)")
    plt.plot(t_c, I_c, label="C (moyenne 30)")

    plt.xlabel("Temps (jours)")
    plt.ylabel("Proportion infectieux I/N")
    plt.title("Modèle multi-agent — comparaison des langages (0–100 jours)")
    plt.legend()
    plt.tight_layout()

    out = FIG / "compare_languages_I_first100days.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print("OK ->", out)

if __name__ == "__main__":
    main()
