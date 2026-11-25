import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from odrpack import odr_fit 
#from scipy.odr import ODR, Model, RealData
#from scipy.stats import norm

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
    wx, wy = 1/np.where(dx > 0, dx, 1e-12)^2, 1/np.where(dy > 0, dy, 1e-12)^2
    return wx, wy




if __name__ == "__main__":
    if len(sys.argv) < 2:
       print("Uso: python fit_lineare.py <file_dati.xlsx>")
       sys.exit(1)

    x, y, dx, dy, nomi = getData(sys.argv[1])
    nomi = [pulisci(n) for n in nomi]
    wx, wy = makePesi(dx, dy)
    print(f"Caricati {len(x)} punti da {sys.argv[1]}")
    stima_iniziale = np.polyfit(x,y,1)

    sol = odr_fit(retta, x, y, stima_iniziale, weight_x=wx, weight_y=wy)

    beta, sd_beta, cov_beta = sol.beta, sol.sd_beta, sol.cov_beta
    A, B = beta
    dA, dB = sd_beta

    chi2, r, dr = sol.sum_square, sol.sum_square, 
    gdl = len(x) - 2

    tau = 1/abs(A)
    dtau = (tau**2) * dA

    print("=========================================")
    print(f"FIT LINEARE (Punti analizzati: {len(x)})")
    print("Equazione: y = A*x + B")
    print(f"A = {A:.4e} ± {dA:.4e}")
    print(f"B = {B:.4f} ± {dB:.4f}")
    print("Matrice di covarianza (A, B):")
    print(f"  [[{cov_beta[0, 0]:.4e}, {cov_beta[0, 1]:.4e}],")
    print(f"   [{cov_beta[1, 0]:.4e}, {cov_beta[1, 1]:.4e}]]")
    print("=========================================")
    print("Parametri Fisici estratti:")
    print(f"f_0 = {tau:.2f} ± {dtau:.2f} Hz")
    print(f"chi2 = {chi2:.3f}  gdl = {gdl}  chi2/gdl = {chi2 / gdl:.3f}  chi2_lin = {sol.res_var:.3f}") 
    print("=========================================")

    '''# ===== Finestra 1: Dati e retta di fit =====
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        x, y, xerr=dx, yerr=dy,
        fmt="o", color="blue", capsize=2, elinewidth=0.8, markersize=3,
        label="Dati"
    )

    margin = 0.05 * (x.max() - x.min()) if x.max() != x.min() else 1.0
    x_fit = np.linspace(x.min() - margin, x.max() + margin, 200)
    ax.plot(
        x_fit, retta(x_fit, beta), "r-", lw=1,
        label=(
            f"Fit Lineare ($Ax + B$)\n"
            f"A = {A:.3f} ± {dA:.3f}\n"
            f"B = {B:.3f} ± {dB:.3f}"
        )
    )

    ax.set_xlabel(f"{nomi[0]}")
    ax.set_ylabel("V0")
    ax.set_title("Fit lineare")
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

    # Scatter plot residui
    ax_s.axhspan(-1, 1, color="gray", alpha=0.15, lw=0)
    ax_s.axhspan(-2, 2, color="gray", alpha=0.08, lw=0)
    ax_s.axhline(0, color="black", linewidth=0.8)
    ax_s.plot(x, z, "o", color="blue", label=f"Residui $\\phi_R$ (chi2/gdl = {chi2/gdl:.2f})")
    ax_s.set_xlabel(f"{nomi[0]} [Hz]")
    ax_s.set_ylabel("r_i / $\\sigma_r$")
    ax_s.set_ylim(-3.5, 3.5)
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(loc="upper left")
    ax_s.set_title("Residui normalizzati del fit lineare")

    # Istogramma residui
    ax_h.hist(z, bins=edges, orientation="horizontal", color="blue", alpha=0.5, edgecolor="black")
    # Curva gaussiana sovrapposta (conteggi attesi)
    ax_h.plot(len(z) * (edges[1] - edges[0]) * norm.pdf(y_g), y_g, "k-", linewidth=1)
    ax_h.grid(True, alpha=0.3)
    ax_h.set_xlabel("conteggi")

    plt.show()'''




