#!/usr/bin/env python3
"""
Partie 2 — Modèle multi-agent SEIRS (CONFORME SUJET)
- N=20000 individus
- Grille 300x300 toroïdale
- Plusieurs agents par cellule
- Déplacement : à chaque pas, saut vers une cellule aléatoire (global)
- Voisinage : Moore (8 cases) + la case elle-même
- Infection : p = 1 - exp(-0.5 * N_I)
- Planification : ordre aléatoire, asynchrone, mise à jour immédiate
- Durées individuelles fixes tirées au début :
    dE ~ Exp(mean=3), dI ~ Exp(mean=7), dR ~ Exp(mean=365)
- 730 itérations (jours)
Sortie : CSV t,S,E,I,R
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import numpy as np


# États
SUS, EXP, INF, REM = 0, 1, 2, 3


@dataclass
class Params:
    L: int = 300          # grille LxL
    N: int = 20000        # nombre d'individus
    T: int = 730          # itérations (jours)
    seed: int = 12345     # graine principale

    # Initialisation imposée
    init_S: int = 19980
    init_I: int = 20
    init_E: int = 0
    init_R: int = 0

    # Moyennes exponentielles imposées
    mean_dE: float = 3.0
    mean_dI: float = 7.0
    mean_dR: float = 365.0

    # Coefficient infection imposé (0.5)
    inf_force: float = 0.5


# Offsets Moore (8 voisins)
MOORE = [(-1, -1), (-1, 0), (-1, 1),
         ( 0, -1),          ( 0, 1),
         ( 1, -1), ( 1, 0), ( 1, 1)]


def neg_exp(rng: np.random.Generator, mean: float) -> float:
    """
    Tirage exponentiel conforme à l'énoncé :
      -mean * log(1 - U), U ~ Uniform[0,1)
    """
    u = rng.random()
    return -mean * np.log(1.0 - u)


def init_population(p: Params):
    rng = np.random.default_rng(p.seed)

    # États init (exact)
    states = np.empty(p.N, dtype=np.int8)
    states[:p.init_S] = SUS
    states[p.init_S:p.init_S + p.init_E] = EXP
    states[p.init_S + p.init_E:p.init_S + p.init_E + p.init_I] = INF
    states[p.init_S + p.init_E + p.init_I:] = REM

    # Mélanger pour ne pas avoir les I tous regroupés
    rng.shuffle(states)

    # Durées individuelles (fixes pendant la simulation)
    dE = np.array([neg_exp(rng, p.mean_dE) for _ in range(p.N)], dtype=np.float64)
    dI = np.array([neg_exp(rng, p.mean_dI) for _ in range(p.N)], dtype=np.float64)
    dR = np.array([neg_exp(rng, p.mean_dR) for _ in range(p.N)], dtype=np.float64)

    # Temps écoulé dans l'état courant (initialement 0)
    t_in_state = np.zeros(p.N, dtype=np.int16)

    # Positions initiales aléatoires sur la grille (plusieurs agents par cellule autorisés)
    x = rng.integers(0, p.L, size=p.N, dtype=np.int16)
    y = rng.integers(0, p.L, size=p.N, dtype=np.int16)

    # Grille de comptage des infectieux par cellule (pour calculer N_I rapidement)
    Icount = np.zeros((p.L, p.L), dtype=np.int16)
    for i in range(p.N):
        if states[i] == INF:
            Icount[x[i], y[i]] += 1

    return rng, states, t_in_state, dE, dI, dR, x, y, Icount


def count_S_E_I_R(states: np.ndarray):
    S = int(np.sum(states == SUS))
    E = int(np.sum(states == EXP))
    I = int(np.sum(states == INF))
    R = int(np.sum(states == REM))
    return S, E, I, R


def neighborhood_I(Icount: np.ndarray, x: int, y: int, L: int) -> int:
    """
    N_I = infectieux dans Moore (8 voisins) + cellule centrale,
    sur une grille toroïdale.
    """
    total = int(Icount[x, y])  # cellule centrale incluse (exigé par le sujet)
    for dx, dy in MOORE:
        total += int(Icount[(x + dx) % L, (y + dy) % L])
    return total


def step_one_agent(i: int,
                   rng: np.random.Generator,
                   p: Params,
                   states: np.ndarray,
                   t_in_state: np.ndarray,
                   dE: np.ndarray, dI: np.ndarray, dR: np.ndarray,
                   x: np.ndarray, y: np.ndarray,
                   Icount: np.ndarray):
    """
    Mise à jour asynchrone d'un agent i :
    1) déplacement global aléatoire
    2) incrément temps dans l'état
    3) transitions (S->E probabiliste ; E->I ; I->R ; R->S)
    Mise à jour immédiate (Icount et états sont modifiés sur le champ).
    """
    L = p.L

    # ---------- 1) Déplacement global aléatoire (cellule choisie au hasard dans la grille)
    oldx, oldy = int(x[i]), int(y[i])
    nx = int(rng.integers(0, L))
    ny = int(rng.integers(0, L))

    # "vers une autre cellule" : on évite une fois le cas identique
    if nx == oldx and ny == oldy:
        nx = int(rng.integers(0, L))
        ny = int(rng.integers(0, L))

    # Mettre à jour Icount si l'agent est infectieux et change de cellule
    if states[i] == INF and (nx != oldx or ny != oldy):
        Icount[oldx, oldy] -= 1
        Icount[nx, ny] += 1

    x[i], y[i] = nx, ny

    # ---------- 2) temps écoulé dans l'état (discret, 1 pas = 1 jour)
    t_in_state[i] += 1

    # ---------- 3) transitions
    st = states[i]

    if st == SUS:
        NI = neighborhood_I(Icount, nx, ny, L)
        if NI > 0:
            # p = 1 - exp(-0.5 * N_I)
            prob = 1.0 - np.exp(-p.inf_force * NI)
            if rng.random() < prob:
                states[i] = EXP
                t_in_state[i] = 0  # reset temps dans l'état

    elif st == EXP:
        # devient I si temps écoulé > dE
        if float(t_in_state[i]) > float(dE[i]):
            states[i] = INF
            t_in_state[i] = 0
            # devient infectieux -> Icount +1 à la cellule actuelle
            Icount[nx, ny] += 1

    elif st == INF:
        if float(t_in_state[i]) > float(dI[i]):
            states[i] = REM
            t_in_state[i] = 0
            # quitte infectieux -> Icount -1
            Icount[nx, ny] -= 1

    elif st == REM:
        if float(t_in_state[i]) > float(dR[i]):
            states[i] = SUS
            t_in_state[i] = 0


def run_one_sim(p: Params, out_csv: Path):
    rng, states, t_in_state, dE, dI, dR, x, y, Icount = init_population(p)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    order = np.arange(p.N, dtype=np.int32)

    with out_csv.open("w", encoding="utf-8") as f:
        f.write("t,S,E,I,R\n")
        S, E, I, R = count_S_E_I_R(states)
        f.write(f"0,{S},{E},{I},{R}\n")

        for t in range(1, p.T + 1):
            rng.shuffle(order)  # planification aléatoire
            for i in order:
                step_one_agent(int(i), rng, p, states, t_in_state, dE, dI, dR, x, y, Icount)

            S, E, I, R = count_S_E_I_R(states)
            f.write(f"{t},{S},{E},{I},{R}\n")


def main():
    parser = argparse.ArgumentParser(description="SEIRS multi-agent (Partie 2)")
    parser.add_argument("--seed", type=int, default=12345, help="Graine RNG")
    parser.add_argument("--out", type=str, default="data/part2_multi_agent/python_rep01.csv",
                        help="Chemin du CSV de sortie")
    parser.add_argument("--T", type=int, default=730, help="Nombre d'itérations (jours)")

    args = parser.parse_args()

    p = Params(seed=args.seed, T=args.T)
    out = Path(args.out)
    run_one_sim(p, out)
    print("Terminé ->", out)


if __name__ == "__main__":
    main()
