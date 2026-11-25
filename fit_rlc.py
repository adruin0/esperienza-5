import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from odrpack import odr_fit
from scipy.stats import norm

# ==========================================================
# PARAMETRI CIRCUITO RLC (Valori misurati con RLC meter)
# ==========================================================
R = 218.5         # Resistenza [Ohm]
dR = 0.6          # Incertezza resistenza [Ohm]
C = 11.15e-9      # Capacità [F]
dC = 0.03e-9      # Incertezza capacità [F]
L = 9.823e-3      # Induttanza [H]
dL = 0.025e-3     # Incertezza induttanza [H]


def getData(filepath):
    df = pd.read_excel(filepath)
    # Nota: se nel file Excel le colonne sono ordinate come (t, dt, y, dy),
    # usa: x, dx, y, dy = df.to_numpy(dtype=float).T
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

    # x e dx sono già in microsecondi [us]
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

    # Con t in [us], A e dA sono già in [us^-1]
    Gamma = 2.0 * abs(A)
    dGamma = 2.0 * dA

    # Tempo caratteristico di smorzamento dell'inviluppo: già in [us]
    tau_smorz = 1.0 / abs(A)
    dtau_smorz = (tau_smorz**2) * dA

    # Valori teorici attesi dai componenti R, L, C convertiti in [us^-1] e [us]
    # R/L è in [s^-1], quindi moltiplichiamo per 1e-6 per passare a [us^-1]
    gamma_att = (R / L) * 1e-6
    dgamma_att = np.sqrt((dR / L)**2 + (R * dL / (L**2))**2) * 1e-6

    tau_att = 2.0 / gamma_att
    dtau_att = (2.0 / (gamma_att**2)) * dgamma_att

    Q_att = (1.0 / R) * np.sqrt(L / C)
    omega0_att_us = (1.0 / np.sqrt(L * C)) * 1e-6  # [rad/us]

    # Resistenza efficace totale del circuito: R_eff = Gamma[s^-1] * L = (Gamma[us^-1] * 1e6) * L
    Gamma_s = Gamma * 1e6
    dGamma_s = dGamma * 1e6
    R_eff = Gamma_s * L
    dR_eff = np.sqrt((L * dGamma_s)**2 + (Gamma_s * dL)**2)

    print("=========================================")
    print(f"FIT LINEARE RLC - INVILUPPO (Punti analizzati: {len(x)})")
    print("Equazione: y = ln(ΔV_picco) = A*t + B = -(Gamma/2)*t + B")
    print(f"A = {A:.4e} ± {dA:.4e} us^-1")
    print(f"B = {B:.4f} ± {dB:.4f}")
    print("Matrice di covarianza (A, B):")
    print(f"  [[{cov_beta[0, 0]:.4e}, {cov_beta[0, 1]:.4e}],")
    print(f"   [{cov_beta[1, 0]:.4e}, {cov_beta[1, 1]:.4e}]]")
    print("=========================================")
    print("Parametri Fisici estratti:")
    print(f"Gamma_fit = {Gamma:.4e} ± {dGamma:.4e} us^-1")
    print(f"tau_smorz = {tau_smorz:.2f} ± {dtau_smorz:.2f} us  (atteso: {tau_att:.2f} ± {dtau_att:.2f} us)")
    print(f"R_eff (Gamma*L) = {R_eff:.2f} ± {dR_eff:.2f} Ohm  (vs R_nom = {R:.2f} ± {dR:.2f} Ohm)")
    print(f"Parametri teorici RLC: Q_att = {Q_att:.2f}, omega_0 = {omega0_att_us:.4e} rad/us")
    print(f"chi2 = {chi2:.3f}  gdl = {gdl}  chi2/gdl = {chi2 / gdl:.3f}")

    # Test di consistenza statistico tra Gamma_fit [us^-1] e R/L [us^-1]
    test_consistenza(Gamma, dGamma, gamma_att, dgamma_att, nome_param="Gamma", unita="us^-1")

    # ===== Finestra 1: Dati e retta di fit =====
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        x, y, xerr=dx, yerr=dy,
        fmt="o", color="blue", capsize=2, elinewidth=0.8, markersize=3,
        label="Picchi inviluppo (scala semilog)"
    )

    margin = 0.05 * (x.max() - x.min()) if x.max() != x.min() else 1.0
    x_fit = np.linspace(x.min() - margin, x.max() + margin, 200)
    ax.plot(
        x_fit, retta(x_fit, beta), "r-", lw=1.2,
        label=(
            f"Fit Lineare ($At + B$)\n"
            f"A = {A:.3e} ± {dA:.3e} $\\mu$s$^{{-1}}$\n"
            f"B = {B:.3f} ± {dB:.3f}\n"
            f"$\\Gamma$ = ({Gamma:.3e} ± {dGamma:.3e}) $\\mu$s$^{{-1}}$"
        )
    )

    ax.set_xlabel(f"{nomi[0]} [$\\mu$s]" if "s" not in nomi[0].lower() else f"{nomi[0]}")
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
    ax_s.set_xlabel(f"{nomi[0]} [$\\mu$s]" if "s" not in nomi[0].lower() else f"{nomi[0]}")
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