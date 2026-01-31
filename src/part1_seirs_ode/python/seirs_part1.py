#!/usr/bin/env python3
"""
Partie 1 — SEIRS ODE
Simulation sur 730 jours
Méthodes : Euler explicite et Runge–Kutta 4
Sorties : CSV + figures
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Paramètres du modèle
# =========================
@dataclass
class Params:
    rho: float = 1.0 / 365.0
    beta: float = 0.5
    sigma: float = 1.0 / 3.0
    gamma: float = 1.0 / 7.0


@dataclass
class Initial:
    S0: float = 0.999
    E0: float = 0.0
    I0: float = 0.001
    R0: float = 0.0


# =========================
# Modèle SEIRS
# =========================
def seirs_rhs(y, p: Params):
    S, E, I, R = y
    dS = p.rho * R - p.beta * S * I
    dE = p.beta * S * I - p.sigma * E
    dI = p.sigma * E - p.gamma * I
    dR = p.gamma * I - p.rho * R
    return np.array([dS, dE, dI, dR])


# =========================
# Méthodes numériques
# =========================
def step_euler(y, dt, p):
    return y + dt * seirs_rhs(y, p)


def step_rk4(y, dt, p):
    k1 = seirs_rhs(y, p)
    k2 = seirs_rhs(y + 0.5 * dt * k1, p)
    k3 = seirs_rhs(y + 0.5 * dt * k2, p)
    k4 = seirs_rhs(y + dt * k3, p)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# =========================
# Simulation
# =========================
def simulate(method, dt, days, p, init):
    n_steps = int(days / dt)
    t = np.linspace(0, days, n_steps + 1)

    Y = np.zeros((n_steps + 1, 4))
    Y[0] = np.array([init.S0, init.E0, init.I0, init.R0])

    for n in range(n_steps):
        if method == "euler":
            Y[n+1] = step_euler(Y[n], dt, p)
        else:
            Y[n+1] = step_rk4(Y[n], dt, p)

        Y[n+1] = np.clip(Y[n+1], 0.0, 1.0)

    return t, Y


# =========================
# Sauvegarde CSV
# =========================
def write_csv(path, t, Y):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "S", "E", "I", "R"])
        for ti, (S, E, I, R) in zip(t, Y):
            w.writerow([ti, S, E, I, R])


# =========================
# Tracé des figures
# =========================
def plot_curves(path, t, Y, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(t, Y[:, 0], label="S")
    plt.plot(t, Y[:, 1], label="E")
    plt.plot(t, Y[:, 2], label="I")
    plt.plot(t, Y[:, 3], label="R")
    plt.xlabel("Temps (jours)")
    plt.ylabel("Proportion")
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# =========================
# Programme principal
# =========================
def main():
    dt = 1.0
    days = 730

    p = Params()
    init = Initial()

    out_data = Path("data/part1_seirs_ode")
    out_fig = Path("figures/part1")

    for method in ["euler", "rk4"]:
        t, Y = simulate(method, dt, days, p, init)

        csv_path = out_data / f"python_{method}.csv"
        fig_path = out_fig / f"python_{method}.png"

        write_csv(csv_path, t, Y)
        plot_curves(fig_path, t, Y, f"SEIRS — Python — {method.upper()}")

        print(f"{method} terminé → {csv_path}")

    print("Simulation Python terminée.")


if __name__ == "__main__":
    main()
