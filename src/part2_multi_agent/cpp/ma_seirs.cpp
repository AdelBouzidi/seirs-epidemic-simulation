#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static constexpr int SUS = 0;
static constexpr int EXP = 1;
static constexpr int INF = 2;
static constexpr int REM = 3;

struct Params {
    int L = 300;
    int N = 20000;
    int T = 730;
    uint32_t seed = 12345;

    int init_S = 19980;
    int init_E = 0;
    int init_I = 20;
    int init_R = 0;

    double mean_dE = 3.0;
    double mean_dI = 7.0;
    double mean_dR = 365.0;

    double inf_force = 0.5;
};

static inline double neg_exp(std::mt19937 &gen, double mean) {
    // -mean * log(1 - U), U ~ Uniform[0,1)
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double u = U(gen);
    return -mean * std::log(1.0 - u);
}

static inline int idx2d(int x, int y, int L) {
    return x * L + y;
}

static inline int wrap(int a, int L) {
    int r = a % L;
    return (r < 0) ? r + L : r;
}

static inline int neighborhood_I(const std::vector<int16_t> &Icount, int x, int y, int L) {
    // Moore (8 voisins) + cellule centrale, tore
    static const int dx[8] = {-1,-1,-1, 0, 0, 1, 1, 1};
    static const int dy[8] = {-1, 0, 1,-1, 1,-1, 0, 1};

    int total = Icount[idx2d(x,y,L)]; // centre inclus
    for (int k = 0; k < 8; ++k) {
        int xx = wrap(x + dx[k], L);
        int yy = wrap(y + dy[k], L);
        total += Icount[idx2d(xx,yy,L)];
    }
    return total;
}

static void run_one_sim(const Params &p, const std::string &out_csv) {
    std::mt19937 gen(p.seed);
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int> Upos(0, p.L - 1);

    // états
    std::vector<int8_t> state(p.N, SUS);
    // init exact
    int kS = p.init_S;
    int kE = p.init_E;
    int kI = p.init_I;
    int kR = p.init_R;

    int pos = 0;
    for (int i = 0; i < kS; ++i) state[pos++] = SUS;
    for (int i = 0; i < kE; ++i) state[pos++] = EXP;
    for (int i = 0; i < kI; ++i) state[pos++] = INF;
    for (int i = 0; i < kR; ++i) state[pos++] = REM;

    // mélanger
    std::shuffle(state.begin(), state.end(), gen);

    // durées individuelles fixes
    std::vector<double> dE(p.N), dI(p.N), dR(p.N);
    for (int i = 0; i < p.N; ++i) {
        dE[i] = neg_exp(gen, p.mean_dE);
        dI[i] = neg_exp(gen, p.mean_dI);
        dR[i] = neg_exp(gen, p.mean_dR);
    }

    // temps dans l'état
    std::vector<int16_t> t_in_state(p.N, 0);

    // positions
    std::vector<int16_t> x(p.N), y(p.N);
    for (int i = 0; i < p.N; ++i) {
        x[i] = static_cast<int16_t>(Upos(gen));
        y[i] = static_cast<int16_t>(Upos(gen));
    }

    // grille de comptage infectieux
    std::vector<int16_t> Icount(static_cast<size_t>(p.L) * p.L, 0);
    for (int i = 0; i < p.N; ++i) {
        if (state[i] == INF) {
            Icount[idx2d(x[i], y[i], p.L)]++;
        }
    }

    auto count_states = [&]() {
        int S=0,E=0,I=0,R=0;
        for (int i = 0; i < p.N; ++i) {
            switch (state[i]) {
                case SUS: S++; break;
                case EXP: E++; break;
                case INF: I++; break;
                case REM: R++; break;
            }
        }
        return std::array<int,4>{S,E,I,R};
    };

    // ordre aléatoire
    std::vector<int> order(p.N);
    std::iota(order.begin(), order.end(), 0);

    std::ofstream f(out_csv);
    if (!f) {
        throw std::runtime_error("Impossible d'ouvrir le fichier: " + out_csv);
    }
    f << "t,S,E,I,R\n";
    {
        auto c = count_states();
        f << 0 << "," << c[0] << "," << c[1] << "," << c[2] << "," << c[3] << "\n";
    }

    for (int t = 1; t <= p.T; ++t) {
        std::shuffle(order.begin(), order.end(), gen);

        for (int ii = 0; ii < p.N; ++ii) {
            int i = order[ii];

            int oldx = x[i], oldy = y[i];
            int nx = Upos(gen);
            int ny = Upos(gen);
            if (nx == oldx && ny == oldy) { // "autre cellule" (1 tentative)
                nx = Upos(gen);
                ny = Upos(gen);
            }

            // mise à jour Icount si infectieux et déplacement
            if (state[i] == INF && (nx != oldx || ny != oldy)) {
                Icount[idx2d(oldx, oldy, p.L)]--;
                Icount[idx2d(nx, ny, p.L)]++;
            }

            x[i] = static_cast<int16_t>(nx);
            y[i] = static_cast<int16_t>(ny);

            // temps discret (1 jour)
            t_in_state[i]++;

            int st = state[i];

            if (st == SUS) {
                int NI = neighborhood_I(Icount, nx, ny, p.L);
                if (NI > 0) {
                    double prob = 1.0 - std::exp(-p.inf_force * static_cast<double>(NI));
                    if (U01(gen) < prob) {
                        state[i] = EXP;
                        t_in_state[i] = 0;
                    }
                }
            } else if (st == EXP) {
                if (static_cast<double>(t_in_state[i]) > dE[i]) {
                    state[i] = INF;
                    t_in_state[i] = 0;
                    Icount[idx2d(nx, ny, p.L)]++; // devient infectieux
                }
            } else if (st == INF) {
                if (static_cast<double>(t_in_state[i]) > dI[i]) {
                    state[i] = REM;
                    t_in_state[i] = 0;
                    Icount[idx2d(nx, ny, p.L)]--; // quitte infectieux
                }
            } else if (st == REM) {
                if (static_cast<double>(t_in_state[i]) > dR[i]) {
                    state[i] = SUS;
                    t_in_state[i] = 0;
                }
            }
        }

        auto c = count_states();
        f << t << "," << c[0] << "," << c[1] << "," << c[2] << "," << c[3] << "\n";
    }

    f.close();
}

int main(int argc, char **argv) {
    Params p;
    std::string out = "data/part2_multi_agent/cpp_rep01.csv";

    // Arguments simples : --seed <int> --out <path> --T <int>
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--seed" && i + 1 < argc) {
            p.seed = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (a == "--out" && i + 1 < argc) {
            out = argv[++i];
        } else if (a == "--T" && i + 1 < argc) {
            p.T = std::stoi(argv[++i]);
        } else {
            std::cerr << "Option inconnue: " << a << "\n";
            std::cerr << "Usage: " << argv[0] << " [--seed N] [--out path] [--T N]\n";
            return 1;
        }
    }

    try {
        run_one_sim(p, out);
        std::cout << "Terminé -> " << out << "\n";
    } catch (const std::exception &e) {
        std::cerr << "Erreur: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
