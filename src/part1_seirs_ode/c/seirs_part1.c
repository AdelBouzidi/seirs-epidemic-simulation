#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* =======================
   Structures de données
   ========================= */

typedef struct {
    double rho;
    double beta;
    double sigma;
    double gamma;
} Params;

typedef struct {
    double S0;
    double E0;
    double I0;
    double R0;
} Initial;

/* =========================
   Modèle SEIRS (ODE)
   ================== */

void seirs_rhs(const double y[4], const Params *p, double dydt[4]) {
    double S = y[0];
    double E = y[1];
    double I = y[2];
    double R = y[3];

    dydt[0] = p->rho * R - p->beta * S * I;
    dydt[1] = p->beta * S * I - p->sigma * E;
    dydt[2] = p->sigma * E - p->gamma * I;
    dydt[3] = p->gamma * I - p->rho * R;
}

/*
   Méthodes numériques
   ========================= */

void step_euler(const double y[4], double dt, const Params *p, double y_next[4]) {
    double k[4];
    seirs_rhs(y, p, k);
    for (int i = 0; i < 4; i++) {
        y_next[i] = y[i] + dt * k[i];
    }
}

void step_rk4(const double y[4], double dt, const Params *p, double y_next[4]) {
    double k1[4], k2[4], k3[4], k4[4], tmp[4];

    seirs_rhs(y, p, k1);

    for (int i = 0; i < 4; i++)
        tmp[i] = y[i] + 0.5 * dt * k1[i];
    seirs_rhs(tmp, p, k2);

    for (int i = 0; i < 4; i++)
        tmp[i] = y[i] + 0.5 * dt * k2[i];
    seirs_rhs(tmp, p, k3);

    for (int i = 0; i < 4; i++)
        tmp[i] = y[i] + dt * k3[i];
    seirs_rhs(tmp, p, k4);

    for (int i = 0; i < 4; i++) {
        y_next[i] = y[i] + (dt / 6.0) *
            (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

/* =========================
   Utilitaires
   ========================= */

double clamp01(double x) {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

void write_header(FILE *f) {
    fprintf(f, "t,S,E,I,R\n");
}

void write_row(FILE *f, double t, const double y[4]) {
    fprintf(f, "%.6f,%.12f,%.12f,%.12f,%.12f\n",
            t, y[0], y[1], y[2], y[3]);
}

/* =========================
   Programme principal
   ========================= */

int main(void) {
    const double dt = 1.0;
    const int days = 730;
    const int n_steps = (int)(days / dt);

    Params p = {1.0/365.0, 0.5, 1.0/3.0, 1.0/7.0};
    Initial init = {0.999, 0.0, 0.001, 0.0};

    const char *methods[2] = {"euler", "rk4"};

    for (int m = 0; m < 2; m++) {
        const char *method = methods[m];

        char filename[256];
        snprintf(filename, sizeof(filename),
                 "data/part1_seirs_ode/c_%s.csv", method);

        FILE *f = fopen(filename, "w");
        if (!f) {
            perror("Erreur ouverture fichier");
            return 1;
        }

        double y[4] = {init.S0, init.E0, init.I0, init.R0};
        double y_next[4];

        write_header(f);
        write_row(f, 0.0, y);

        for (int n = 0; n < n_steps; n++) {
            if (method[0] == 'e') {
                step_euler(y, dt, &p, y_next);
            } else {
                step_rk4(y, dt, &p, y_next);
            }

            for (int i = 0; i < 4; i++)
                y_next[i] = clamp01(y_next[i]);

            for (int i = 0; i < 4; i++)
                y[i] = y_next[i];

            write_row(f, (n + 1) * dt, y);
        }

        fclose(f);
        printf("%s terminé → %s\n", method, filename);
    }

    printf("Simulation C terminée.\n");
    return 0;
}
