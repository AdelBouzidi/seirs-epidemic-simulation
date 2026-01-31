#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define SUS 0
#define EXP 1
#define INF 2
#define REM 3

typedef struct {
    int L;
    int N;
    int T;
    unsigned int seed;
    int init_S, init_E, init_I, init_R;
    double mean_dE, mean_dI, mean_dR;
    double inf_force;
} Params;

/* RNG simple et rapide */
static inline double urand(void) {
    return rand() / (RAND_MAX + 1.0);
}

static inline double neg_exp(double mean) {
    /* -mean * log(1 - U), U ~ Uniform[0,1) */
    return -mean * log(1.0 - urand());
}

static inline int wrap(int a, int L) {
    int r = a % L;
    return (r < 0) ? r + L : r;
}

static inline int idx(int x, int y, int L) {
    return x * L + y;
}

/* Offsets Moore (8 voisins) */
static const int dx8[8] = {-1,-1,-1, 0, 0, 1, 1, 1};
static const int dy8[8] = {-1, 0, 1,-1, 1,-1, 0, 1};

static inline int neighborhood_I(const int16_t *Icount, int x, int y, int L) {
    /* Moore (8 voisins) + cellule centrale (tore) */
    int total = Icount[idx(x, y, L)]; /* centre inclus */
    for (int k = 0; k < 8; k++) {
        int xx = wrap(x + dx8[k], L);
        int yy = wrap(y + dy8[k], L);
        total += Icount[idx(xx, yy, L)];
    }
    return total;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <seed> <output.csv>\n", argv[0]);
        return 1;
    }

    Params p = {
        .L = 300, .N = 20000, .T = 730,
        .seed = (unsigned int)atoi(argv[1]),
        .init_S = 19980, .init_E = 0, .init_I = 20, .init_R = 0,
        .mean_dE = 3.0, .mean_dI = 7.0, .mean_dR = 365.0,
        .inf_force = 0.5
    };

    srand(p.seed);

    int8_t  *state  = (int8_t*) malloc((size_t)p.N * sizeof(int8_t));
    int16_t *tstate = (int16_t*)calloc((size_t)p.N, sizeof(int16_t));
    double  *dE     = (double*)  malloc((size_t)p.N * sizeof(double));
    double  *dI     = (double*)  malloc((size_t)p.N * sizeof(double));
    double  *dR     = (double*)  malloc((size_t)p.N * sizeof(double));
    int16_t *x      = (int16_t*) malloc((size_t)p.N * sizeof(int16_t));
    int16_t *y      = (int16_t*) malloc((size_t)p.N * sizeof(int16_t));
    int16_t *Icount = (int16_t*)calloc((size_t)p.L * (size_t)p.L, sizeof(int16_t));
    int     *order  = (int*)     malloc((size_t)p.N * sizeof(int));

    if (!state || !tstate || !dE || !dI || !dR || !x || !y || !Icount || !order) {
        fprintf(stderr, "Erreur: allocation mémoire échouée.\n");
        free(state); free(tstate); free(dE); free(dI); free(dR);
        free(x); free(y); free(Icount); free(order);
        return 1;
    }

    /* Initialisation des états (exact) */
    int k = 0;
    for (int i = 0; i < p.init_S; i++) state[k++] = SUS;
    for (int i = 0; i < p.init_E; i++) state[k++] = EXP;
    for (int i = 0; i < p.init_I; i++) state[k++] = INF;
    for (int i = 0; i < p.init_R; i++) state[k++] = REM;

    /* Mélange des états */
    for (int i = p.N - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int8_t tmp = state[i];
        state[i] = state[j];
        state[j] = tmp;
    }

    /* Durées individuelles fixes + positions initiales + Icount */
    for (int i = 0; i < p.N; i++) {
        dE[i] = neg_exp(p.mean_dE);
        dI[i] = neg_exp(p.mean_dI);
        dR[i] = neg_exp(p.mean_dR);

        x[i] = (int16_t)(rand() % p.L);
        y[i] = (int16_t)(rand() % p.L);

        if (state[i] == INF) {
            Icount[idx((int)x[i], (int)y[i], p.L)]++;
        }

        order[i] = i;
    }

    FILE *f = fopen(argv[2], "w");
    if (!f) {
        fprintf(stderr, "Erreur: impossible d'ouvrir %s\n", argv[2]);
        free(state); free(tstate); free(dE); free(dI); free(dR);
        free(x); free(y); free(Icount); free(order);
        return 1;
    }

    fprintf(f, "t,S,E,I,R\n");

    for (int t = 0; t <= p.T; t++) {
        /* Comptage S/E/I/R (sortie) */
        int S=0,E=0,I=0,R=0;
        for (int i = 0; i < p.N; i++) {
            if (state[i] == SUS) S++;
            else if (state[i] == EXP) E++;
            else if (state[i] == INF) I++;
            else R++;
        }
        fprintf(f, "%d,%d,%d,%d,%d\n", t, S, E, I, R);

        if (t == p.T) break;

        /* Ordre aléatoire (asynchrone) */
        for (int i = p.N - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }

        /* Mise à jour agents */
        for (int ii = 0; ii < p.N; ii++) {
            int i = order[ii];

            int ox = (int)x[i], oy = (int)y[i];
            int nx = rand() % p.L;
            int ny = rand() % p.L;

            /* "autre cellule" : éviter une fois le cas identique */
            if (nx == ox && ny == oy) {
                nx = rand() % p.L;
                ny = rand() % p.L;
            }

            /* Mise à jour Icount si infectieux et déplacement */
            if (state[i] == INF && (nx != ox || ny != oy)) {
                Icount[idx(ox, oy, p.L)]--;
                Icount[idx(nx, ny, p.L)]++;
            }

            x[i] = (int16_t)nx;
            y[i] = (int16_t)ny;

            tstate[i]++;

            if (state[i] == SUS) {
                int NI = neighborhood_I(Icount, nx, ny, p.L); /* Moore + centre */
                if (NI > 0) {
                    double prob = 1.0 - exp(-p.inf_force * (double)NI);
                    if (urand() < prob) {
                        state[i] = EXP;
                        tstate[i] = 0;
                    }
                }
            } else if (state[i] == EXP) {
                if ((double)tstate[i] > dE[i]) {
                    state[i] = INF;
                    tstate[i] = 0;
                    Icount[idx(nx, ny, p.L)]++; /* devient infectieux */
                }
            } else if (state[i] == INF) {
                if ((double)tstate[i] > dI[i]) {
                    state[i] = REM;
                    tstate[i] = 0;
                    Icount[idx(nx, ny, p.L)]--; /* quitte infectieux */
                }
            } else { /* REM */
                if ((double)tstate[i] > dR[i]) {
                    state[i] = SUS;
                    tstate[i] = 0;
                }
            }
        }
    }

    fclose(f);

    free(state); free(tstate); free(dE); free(dI); free(dR);
    free(x); free(y); free(Icount); free(order);
    return 0;
}
