#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

DATA = Path("data/part2_multi_agent")

def first_local_peak(I_series):
    """
    Retourne (day_peak, peak_I) pour le *premier pic* de I(t).

    Définition robuste :
    - on cherche le premier maximum local : I[t-1] < I[t] >= I[t+1]
    - si aucun maximum local n'est trouvé (cas rare), on prend argmax global.
    """
    I = I_series.values
    n = len(I)
    # Chercher le premier pic local
    for t in range(1, n - 1):
        if I[t-1] < I[t] and I[t] >= I[t+1]:
            return t, int(I[t])
    # Fallback : maximum global
    tmax = int(I.argmax())
    return tmax, int(I[tmax])

def process_one(file_csv: Path, rep_name: str):
    df = pd.read_csv(file_csv)
    if list(df.columns) != ["t", "S", "E", "I", "R"]:
        raise ValueError(f"Colonnes inattendues dans {file_csv}: {list(df.columns)}")
    day_peak, peak_I = first_local_peak(df["I"])
    return {"rep": rep_name, "day_peak": day_peak, "peak_I": peak_I}

def main():
    # Python (3 reps)
    py_files = [
        ("python_rep01", DATA / "python_rep01.csv"),
        ("python_rep02", DATA / "python_rep02.csv"),
        ("python_rep03", DATA / "python_rep03.csv"),
    ]
    py_rows = [process_one(f, name) for (name, f) in py_files]
    pd.DataFrame(py_rows).to_csv(DATA / "python_peaks.csv", index=False)
    print("OK ->", DATA / "python_peaks.csv")

    # C++ (3 reps)
    cpp_files = [
        ("cpp_rep01", DATA / "cpp_rep01.csv"),
        ("cpp_rep02", DATA / "cpp_rep02.csv"),
        ("cpp_rep03", DATA / "cpp_rep03.csv"),
    ]
    cpp_rows = [process_one(f, name) for (name, f) in cpp_files]
    pd.DataFrame(cpp_rows).to_csv(DATA / "cpp_peaks.csv", index=False)
    print("OK ->", DATA / "cpp_peaks.csv")

    # C (30 reps) dans c_runs/
    c_dir = DATA / "c_runs"
    c_files = sorted(c_dir.glob("c_rep*.csv"))
    if len(c_files) != 30:
        raise ValueError(f"Attendu 30 fichiers dans {c_dir}, trouvé {len(c_files)}")
    c_rows = []
    for f in c_files:
        rep = f.stem  # ex: c_rep01
        c_rows.append(process_one(f, rep))
    pd.DataFrame(c_rows).to_csv(DATA / "c_peaks.csv", index=False)
    print("OK ->", DATA / "c_peaks.csv")

if __name__ == "__main__":
    main()
