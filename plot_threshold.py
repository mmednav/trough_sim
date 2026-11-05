# -*- coding: utf-8 -*-

import argparse
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required:\n    pip install matplotlib")

from results_roc import load as load_csv

CSV = "results_final/results_troughs_final.csv"

INK = "#1a1a1a"
INK2 = "#555555"
GRID = "#d9d9d4"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#9aa0a6"
GREEN = "#4b9b6e"


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False, "font.size": 10,
        "axes.labelsize": 10, "axes.titlesize": 10, "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
    })


def tidy(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, length=3)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fig_detection(out, alpha=0.05):
    d = load_csv(CSV)
    lab = d["label"].astype(int)
    rho = d["rho_oriented"] * 1000.0
    amp = np.round(d["defect_amplitude_m"] * 1000.0).astype(int)
    dt = d["defect_type"]
    sigma = 1000.0 * d["noise_sigma"][0]

    nom = rho[lab == 0]
    tau = float(np.quantile(nom, 1.0 - alpha))
    floor = nom.mean() / 2.0

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    amps = sorted(set(amp[lab == 1]))

    for kind, col, mk in (("tilt", ORANGE, "s"), ("dent", BLUE, "^")):
        xs, ys, lo, hi = [], [], [], []
        for a in amps:
            m = (lab == 1) & (dt == kind) & (amp == a)
            n = int(m.sum())
            if not n:
                continue
            k = int((rho[m] > tau).sum())
            l, u = wilson(k, n)
            xs.append(a)
            ys.append(k / n)
            lo.append(k / n - l)
            hi.append(u - k / n)
        ax.errorbar(xs, ys, yerr=[lo, hi], marker=mk, ms=5, lw=1.3,
                    color=col, capsize=2.5, elinewidth=0.9, label=kind)

    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("defect amplitude (mm)")
    ax.set_ylabel(r"fraction detected at $\tau_{%.2f}$" % alpha)
    ax.legend(loc="lower right")
    tidy(ax)

    # a second scale in units of the range noise, which is the form in
    # which the floor transfers to another instrument
    sec = ax.secondary_xaxis("top", functions=(lambda v: v / sigma,
                                               lambda v: v * sigma))
    sec.set_xlabel(r"amplitude in units of the range noise, $A/\sigma$")
    sec.tick_params(colors=INK2, length=3)

    fig.tight_layout()
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(out, "fig_th_detection.%s" % ext)
        fig.savefig(p)
        print("   wrote", p)
    plt.close(fig)

    print()
    print("   tau_%.2f = %.2f mm   floor rho_0/2 = %.1f mm = %.2f sigma"
          % (alpha, tau, floor, floor / sigma))


def fig_roc(out, alpha=0.05, reps=2000):
    from results_roc import auc_mw, auc_ci, roc_curve

    d = load_csv(CSV)
    lab = d["label"].astype(int)
    methods = [("oriented", "rho_oriented", BLUE, "-"),
               ("perturbation", "rho_pert", ORANGE, "--"),
               ("rotation search", "rho_grid", GREEN, ":")]

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], lw=0.9, ls=(0, (1, 3)), color=GREY,
            label="chance")

    for name, col, colour, ls in methods:
        if col not in d:
            continue
        a = auc_mw(d[col], lab)
        lo, hi = auc_ci(d[col], lab, reps=reps)
        fpr, tpr, _ = roc_curve(d[col], lab)
        ax.step(fpr, tpr, where="post", lw=1.4, color=colour, ls=ls,
                label=r"%s, %.3f $[%.3f,\,%.3f]$" % (name, a, lo, hi))

    # where the threshold of the rest of the section sits on the curve
    score = d["rho_oriented"]
    tau = float(np.quantile(score[lab == 0], 1.0 - alpha))
    tp = float(((score > tau) & (lab == 1)).sum()) / max(1, (lab == 1).sum())
    fp = float(((score > tau) & (lab == 0)).sum()) / max(1, (lab == 0).sum())
    ax.plot([fp], [tp], marker="o", ms=7, mfc="white", mec=INK, mew=1.4,
            zorder=5)
    ax.annotate(r"$\tau_{%.2f}$" % alpha, xy=(fp, tp), xytext=(10, -10),
                textcoords="offset points", fontsize=9, color=INK)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(loc="lower right", title="AUC, with 95% interval",
              title_fontsize=8.5)
    tidy(ax)
    fig.tight_layout()

    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(out, "fig_th_roc.%s" % ext)
        fig.savefig(p)
        print("   wrote", p)
    plt.close(fig)
    print()
    print("   operating point: FPR %.3f, TPR %.3f at tau = %.2f mm"
          % (fp, tp, 1000 * tau))


def fig_null(out, alpha=0.05):
    d = load_csv(CSV)
    lab = d["label"].astype(int)
    nom = d["rho_oriented"][lab == 0] * 1000.0
    sigma = 1000.0 * d["noise_sigma"][0]
    n = int(np.median(d["n_points"]))

    gamma = 0.5772156649
    ln = np.log(n)
    a_n = np.sqrt(2.0 * ln)
    b_n = a_n - (np.log(ln) + np.log(4.0 * np.pi)) / (2.0 * a_n)
    predicted = 2.0 * sigma * (b_n + gamma / a_n)

    tau = float(np.quantile(nom, 1.0 - alpha))

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.hist(nom, bins=18, color=BLUE, alpha=0.55, edgecolor="white",
            linewidth=0.6, label="the %d defect-free collectors" % len(nom))

    ax.axvline(nom.mean(), lw=1.3, color=BLUE,
               label=r"mean $\rho_{0} = %.2f$ mm" % nom.mean())
    ax.axvline(predicted, lw=1.3, ls="--", color=GREY,
               label=r"$d_{2}(n)\,\sigma = %.2f$ mm" % predicted)
    ax.axvline(tau, lw=1.3, ls=":", color=ORANGE,
               label=r"$\tau_{%.2f} = %.2f$ mm" % (alpha, tau))

    ax.set_xlabel(r"$\rho_{0}$, the width at the nominal axis (mm)")
    ax.set_ylabel("collectors")
    ax.legend(loc="upper left")
    tidy(ax)
    fig.tight_layout()

    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(out, "fig_th_null.%s" % ext)
        fig.savefig(p)
        print("   wrote", p)
    plt.close(fig)

    print()
    print("   measured %.2f +- %.2f mm   predicted %.2f mm   ratio %.3f"
          % (nom.mean(), nom.std(ddof=1), predicted, nom.mean() / predicted))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None,
                    choices=["detection", "roc", "null"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="results_final/figures")
    args = ap.parse_args()
    style()
    todo = {"null": fig_null, "detection": fig_detection,
            "roc": fig_roc}
    for name, f in todo.items():
        if args.only in (None, name):
            print(name)
            f(args.out, args.alpha)


if __name__ == "__main__":
    main()
