# Projet SEIRS — Modèles ODE et Multi-Agent (Python / C / C++)

Ce dépôt contient l’implémentation complète d’un projet de simulation
épidémiologique basé sur le modèle **SEIRS**, réalisée dans le cadre d’un projet
de niveau Master.  
Le travail repose sur deux approches complémentaires :

- une approche **macroscopique** basée sur des **équations différentielles (ODE)** ;
- une approche **microscopique stochastique** basée sur un **modèle multi-agent**.

L’objectif est de comparer les dynamiques épidémiques obtenues, et
l’influence du **schéma numérique** et du **langage de programmation** sur les
résultats.

---


## Partie 1 — Modèle SEIRS par équations différentielles (ODE)

La première partie implémente un modèle SEIRS continu décrit par un système
d’équations différentielles ordinaires.

- Méthodes numériques :
  - Euler explicite
  - Runge–Kutta d’ordre 4 (RK4)
- Implémentations :
  - Python
  - C
- Comparaisons :
  - Python vs C
  - Euler vs RK4

Les simulations produisent des fichiers CSV et des figures illustrant l’évolution
temporelle des compartiments S, E, I et R.  
Une analyse comparative met en évidence les différences dues au schéma numérique,
ainsi que la cohérence des résultats entre langages à précision machine.

---

## Partie 2 — Modèle SEIRS multi-agent

La seconde partie repose sur un modèle multi-agent stochastique, où chaque
individu est représenté explicitement.

Caractéristiques principales :
- Population de 20 000 agents
- Grille bidimensionnelle toroïdale (300 × 300)
- Mise à jour **asynchrone** avec ordre aléatoire
- Déplacements aléatoires globaux
- Voisinage de Moore (incluant la cellule centrale)
- Probabilité d’infection :  
  \( p = 1 - \exp(-0.5\,N_I) \)
- Durées individuelles (E, I, R) tirées selon des lois exponentielles

Implémentations réalisées :
- Python
- C++
- C

Les simulations sont répétées sur plusieurs réplications indépendantes
(jusqu’à 30 pour l’implémentation C).  
Les résultats sont analysés à l’aide de moyennes, d’extraction de pics
infectieux et de tests statistiques non paramétriques
(Kruskal–Wallis).

---

## Organisation générale

- `src/` : codes sources (ODE et multi-agent)
- `data/` : résultats bruts des simulations (CSV)
- `figures/` : figures générées automatiquement
- `notebooks/` : analyses et visualisations complémentaires
- `rapport/` : rapport scientifique du projet

---

## Prérequis

- Linux (testé sous Ubuntu)
- Python ≥ 3.10  
  Bibliothèques : `numpy`, `pandas`, `matplotlib`, `scipy`
- Compilateurs :
  - `gcc` (C)
  - `g++` (C++)

---

## Exécution (vue d’ensemble)

- Les scripts Python permettent :
  - de lancer les simulations,
  - d’agréger les réplications,
  - de générer automatiquement les figures et statistiques.
- Les programmes C et C++ sont compilés avec optimisation (`-O2`) et exécutés
  via la ligne de commande avec une graine aléatoire explicite.

Les résultats sont enregistrés automatiquement dans les répertoires `data/`
et `figures/`.

---

## Auteur =>

Projet réalisé par **Adel Bouzidi**  
Master CHPS — Université de Perpignan Via Domitia
