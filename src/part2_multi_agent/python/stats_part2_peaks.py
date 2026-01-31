#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from scipy.stats import kruskal

DATA = Path("data/part2_multi_agent")

def load(name):
    return pd.read_csv(DATA / name)

def kw_test(metric, py, cpp, c):
    H, p = kruskal(py[metric].values, cpp[metric].values, c[metric].values)
    return H, p

def main():
    py = load("python_peaks.csv")
    cpp = load("cpp_peaks.csv")
    c = load("c_peaks.csv")

    print("=== Descriptif (moyenne ± std) ===")
    for label, df in [("Python (3)", py), ("C++ (3)", cpp), ("C (30)", c)]:
        print(f"\n[{label}]")
        for metric in ["peak_I", "day_peak"]:
            m = df[metric].mean()
            s = df[metric].std(ddof=1) if len(df) > 1 else 0.0
            print(f"  {metric}: {m:.3f} ± {s:.3f}")

    print("\n=== Kruskal–Wallis (3 groupes : Python, C++, C) ===")
    for metric in ["peak_I", "day_peak"]:
        H, pval = kw_test(metric, py, cpp, c)
        print(f"{metric}: H={H:.6g}, p-value={pval:.6g}")

    print("\nInterprétation:")
    print("- p-value < 0.05 : différence statistiquement significative entre au moins deux langages.")
    print("- p-value >= 0.05 : pas de différence significative détectée (au seuil 5%).")

if __name__ == "__main__":
    main()
