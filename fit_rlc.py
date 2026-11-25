import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from odrpack import odr_fit
from scipy.stats import norm

# ==========================================================
# PARAMETRI CIRCUITO RLC (Valori misurati con RLC meter)
# ==========================================================
R = 100.30       # Resistenza [Ohm]
dR = 0.35        # Incertezza resistenza [Ohm]
L = 9.891e-3     # Induttanza [H] (9.891 mH)
dL = 0.025e-3    # Incertezza induttanza [H] (0.025 mH)
C = 219.0e-9     # Capacità [F] (219.0 nF)
dC = 0.6e-9      # Incertezza capacità [F] (0.6 nF)


def getData(filepath):
    df = pd.read_excel(filepath)
    x, y, dx, dy = df.to_numpy(dtype=float).T
    labels = list(df.columns)
    return x, y, dx, dy, labels


def pulisci(nome):
    return nome.split("=")[-1].strip()


def retta(x, params):
    A, B = params
    return A * x + B


def makePesi(dx, dy):
    wx = 1.0 / (np.where(dx > 0, dx, 1e-12) ** 2)
    wy = 1.0 / (np.where(dy > 0, dy, 1e-12) ** 2)
    return wx, wy


def test_consistenza(val_fit, err_fit, val_att, err_att, nome_param="Parametro", unita=""):
    diff = abs(val_fit - val_att)
    err_diff = np.sqrt(err_fit**2 + err_att**2)
    t_stat = diff / err_diff
    p_val = 2.0 * norm.sf(t_stat)

    if t_stat <= 2.0:
        esito = "COMPATIBILE (accordo entro 2 sigma)"
    elif t_stat <= 3.0:
        esito = "TENSIONE STATISTICA (discrepanza tra 2 e 3 sigma)"
    else:
        esito = "INCOMPATIBILE (discrepanza oltre 3 sigma)"

    print("=========================================")
    print(f"TEST DI CONSISTENZA STATISTICA ({nome_param})")
    print(f"Valore da fit:       {val_fit:.4e} ± {err_fit:.4e} {unita}")
    print(f"Valore atteso (R/L): {val_att:.4e} ± {err_att:.4e} {unita}")
    print(f"Discrepanza |Δ|:     {diff:.4e} ± {err_diff:.4e} {unita}")
    print(f"Compatibilità t:     {t_stat:.2f} sigma  (p-value = {p_val:.4f})")
    print(f"Esito:               {esito}")
    print("=========================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python fit_rlc.py <file_dati_rlc.xlsx>")
        sys.exit(1)

    x, y, dx, dy, nomi = getData(sys.argv[1])
    nomi = [pulisci(n) for n in nomi]
    wx, wy = makePesi(dx, dy)
    print(f"Caricati {len(x)} punti da {sys.argv[1]}")

    stima_iniziale = np.polyfit(x, y, 1)
    sol = odr_fit(retta, x, y, stima_iniziale, weight_x=wx, weight_y=wy)

    beta, sd_beta, cov_beta = sol.beta, sol.sd_beta, sol.cov_beta
    A, B = beta
    dA, dB = sd_beta

    # Residui efficaci e chi-quadro
    r = y - retta(x, beta)
    sr = np.sqrt(dy**2 + (A * dx)**2)
    chi2 = sol.sum_square
    gdl = len(x) - 2

    # Da V_c(t) ~ exp(-Gamma/2 * t) -> ln(inviluppo) = -(Gamma/2)*t + B
    Gamma = 2.0 * abs(A)
    dGamma = 2.0 * dA

    # Tempo caratteristico di smorzamento dell'inviluppo (tau_RLC = 2/Gamma = 1/|A|)
    tau_smorz = 1.0 / abs(A)
    dtau_smorz = (tau_smorz**2) * dA

    # Valori teorici attesi dai componenti R, L, C
    gamma_att = R / L
    dgamma_att = np.sqrt((dR / L)**2 + (R * dL / (L**2))**2)
    Q_att = (1.0 / R) * np.sqrt(L / C)
    omega0_att = 1.0 / np.sqrt(L * C)

    # Resistenza efficace totale del circuito stimata dal fit (R_eff = Gamma * L)
    R_eff = Gamma * L
    dR_eff = np.sqrt((L * dGamma)**2 + (Gamma * dL)**2)

    print("=========================================")
    print(f"FIT LINEARE RLC - INVILUPPO (Punti analizzati: {len(x)})")
    print("Equazione: y = ln(ΔV_picco) = A*t + B = -(Gamma/2)*t + B")
    print(f"A = {A:.4e} ± {dA:.4e} s^-1")
    print(f"B = {B:.4f} ± {dB:.4f}")
    print("Matrice di covarianza (A, B):")
    print(f"  [[{cov_beta[0, 0]:.4e}, {cov_beta[0, 1]:.4e}],")
    print(f"   [{cov_beta[1, 0]:.4e}, {cov_beta[1, 1]:.4e}]]")
    print("=========================================")
    print("Parametri Fisici estratti:")
    print(f"Gamma_fit = {Gamma:.4e} ± {dGamma:.4e} s^-1")
    print(f"tau_smorz = {tau_smorz*1e6:.2f} ± {dtau_smorz*1e6:.2f} us")
    print(f"R_eff (Gamma*L) = {R_eff:.2f} ± {dR_eff:.2f} Ohm  (vs R_nom = {R:.2f} ± {dR:.2f} Ohm)")
    print(f"Parametri teorici RLC: Q_att = {Q_att:.2f}, omega_0 = {omega0_att:.2f} rad/s")
    print(f"chi2 = {chi2:.3f}  gdl = {gdl}  chi2/gdl = {chi2 / gdl:.3f}")

    # Test di consistenza statistico tra Gamma_fit e R/L
    test_consistenza(Gamma, dGamma, gamma_att, dgamma_att, nome_param="Gamma", unita="s^-1")

    # ===== Finestra 1: Dati e retta di fit =====
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        x, y, xerr=dx, yerr=dy,
        fmt="o", color="blue", capsize=2, elinewidth=0.8, markersize=3,
        label="Picchi inviluppo (scala log)"
    )

    margin = 0.05 * (x.max() - x.min()) if x.max() != x.min() else 1.0
    x_fit = np.linspace(x.min() - margin, x.max() + margin, 200)
    ax.plot(
        x_fit, retta(x_fit, beta), "r-", lw=1.2,
        label=(
            f"Fit Lineare ($At + B$)\n"
            f"A = {A:.3e} ± {dA:.3e} s$^{{-1}}$\n"
            f"B = {B:.3f} ± {dB:.3f}\n"
            f"$\\Gamma$ = ({Gamma:.3e} ± {dGamma:.3e}) s$^{{-1}}$"
        )
    )

    ax.set_xlabel(f"{nomi[0]}")
    ax.set_ylabel(f"{nomi[1]}" if len(nomi) > 1 else "ln(ΔV_picco)")
    ax.set_title("Fit lineare inviluppo - Circuito RLC")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    # ===== Finestra 2: Residui normalizzati con istogramma =====
    z = r / sr
    edges = np.linspace(-3, 3, 7)
    y_g = np.linspace(-3.5, 3.5, 400)

    fig_res, axs_res = plt.subplots(
        1, 2, figsize=(10, 4.5), sharey=True,
        width_ratios=[4, 1], layout="constrained"
    )
    ax_s, ax_h = axs_res

    ax_s.axhspan(-1, 1, color="gray", alpha=0.15, lw=0)
    ax_s.axhspan(-2, 2, color="gray", alpha=0.08, lw=0)
    ax_s.axhline(0, color="black", linewidth=0.8)
    ax_s.plot(x, z, "o", color="blue", label=f"Residui (chi2/gdl = {chi2/gdl:.2f})")
    ax_s.set_xlabel(f"{nomi[0]}")
    ax_s.set_ylabel("r_i / $\\sigma_r$")
    ax_s.set_ylim(-3.5, 3.5)
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(loc="upper left")
    ax_s.set_title("Residui normalizzati - Circuito RLC")

    ax_h.hist(z, bins=edges, orientation="horizontal", color="blue", alpha=0.5, edgecolor="black")
    ax_h.plot(len(z) * (edges[1] - edges[0]) * norm.pdf(y_g), y_g, "k-", linewidth=1)
    ax_h.grid(True, alpha=0.3)
    ax_h.set_xlabel("conteggi")

    plt.show()