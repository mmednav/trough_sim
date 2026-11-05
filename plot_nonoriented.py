# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from pipeline_loader import load
from results_roc import load as load_csv

NS = load()
CSV = "results_final/results_troughs_final.csv"
CLOUDS = "results_final/clouds"

INK2 = "#555555"
GRID = "#d9d9d4"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#4b9b6e"
GREY = "#9aa0a6"


# ---------------------------------------------------------------- style
# & store the figures
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


def save(fig, name, out):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(out, "%s.%s" % (name, ext))
        fig.savefig(p)
        print("   wrote", p)
    plt.close(fig)


# ---------------------------------------------------------------- landscape
def fig_landscape(out, runs=(1, 7, 19), phi_max=0.14, n_grid=281):

    # can only run with data points
    missing = [k for k in runs
               if not os.path.exists(os.path.join(CLOUDS,
                                                  "trough_%04d.npz" % k))]
    if missing:
        print("   skipped: missing archived clouds")
        return


    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    d = load_csv(CSV)
    colours = (BLUE, ORANGE, GREEN)

    for k, col in zip(runs, colours):
        z = np.load(os.path.join(CLOUDS, "trough_%04d.npz" % k))
        P = NS["voxel_downsample_numpy"](z["points"].astype(float),
                                        float(z["voxel_size"]))
        xz = NS["project_to_cross_section"](P)
        i = int(np.where(d["run_id"] == k)[0][0])
        delta = float(d["delta_injected_rad"][i])
        rot = NS["rotate_points_about_vertex"](xz, delta)
        r = NS["rotation_search_parabolicity"](rot, phi_max=phi_max,
                                              n_grid=n_grid)
        phi = np.rad2deg(r["phis"])
        w = 1000.0 * r["widths"]
        ax.plot(phi, w, lw=1.3, color=col,
                label=r"collector %d,  $\delta=%+.2f^\circ$"
                      % (k, np.rad2deg(delta)))
        ax.plot([-np.rad2deg(delta)], [1000 * r["width"]], marker="v",
                ms=6, color=col, zorder=4)

    ax.set_xlabel(r"direction of the axis, $\varphi$ (degrees)")
    ax.set_ylabel(r"annulus width $\rho$ (mm)")
    ax.set_title("The width against the direction searched")
    ax.legend(loc="upper center")
    tidy(ax)
    fig.tight_layout()
    save(fig, "fig_no_landscape", out)


# ----------------------------------------------------------------- recovery
def fig_recovery(out):
    # Both series are measured against the SAME baseline, the width of the
    # very same cloud before any misalignment was injected.
    d = load_csv(CSV)
    dl = np.abs(d["delta_injected_deg"])
    base = d["rho_oriented"]
    infl = 100 * (d["rho_rotated_uncorrected"] - base) / base
    rec = 100 * (d["rho_grid"] - base) / base

    # only for the defect-free collectors
    keep = d["label"].astype(int) == 0
    dl, infl, rec = dl[keep], infl[keep], rec[keep]

    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    ax.scatter(dl, infl, s=14, color=ORANGE, alpha=0.7, linewidths=0,
               label=r"$\rho=\rho_{\delta}$, at the nominal axis")
    ax.scatter(dl, rec, s=14, color=BLUE, alpha=0.7, linewidths=0,
               label=r"$\rho=\rho_{\mathrm{rot}}$, after searching")
    ax.axhline(0, lw=1.0, color=INK2)
    ax.set_xlabel(r"misalignment injected, $|\delta|$ (degrees)")
    ax.set_ylabel(r"$(\rho-\rho_{0})/\rho_{0}$   (per cent)")
    ax.legend(loc="upper left")
    tidy(ax)
    fig.tight_layout()
    save(fig, "fig_no_recovery", out)


# ---------------------------------------------------------------- agreement
def fig_agreement(out):
    #  linearized program against the rotation search.

    d = load_csv(CSV)
    rot = d["rho_grid"] * 1000
    lin = d["rho_pert"] * 1000
    dt = d["defect_type"]
    kinds = (("none", GREY, "o", "defect-free"),
             ("tilt", ORANGE, "s", "tilt"),
             ("dent", BLUE, "^", "dent"))

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(7.4, 3.3))

    lim = [min(rot.min(), lin.min()) * 0.98, max(rot.max(), lin.max()) * 1.02] # select limit
    ax.plot(lim, lim, ls="--", lw=1.0, color=INK2, zorder=1,
            label=r"$\rho_{\mathrm{lin}}=\rho_{\mathrm{rot}}$")
    for k, col, mk, lab in kinds:
        m = dt == k
        ax.scatter(rot[m], lin[m], s=14, color=col, marker=mk, alpha=0.75,
                   linewidths=0, label=lab)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\rho_{\mathrm{rot}}$ (mm)")
    ax.set_ylabel(r"$\rho_{\mathrm{lin}}$ (mm)")
    tidy(ax)

    rel = 100 * (lin - rot) / rot
    axr.axhline(0, lw=1.0, color=INK2)
    for k, col, mk, _ in kinds:
        m = dt == k
        axr.scatter(rot[m], rel[m], s=14, color=col, marker=mk, alpha=0.75,
                    linewidths=0)
    axr.set_xlabel(r"$\rho_{\mathrm{rot}}$ (mm)")
    axr.set_ylabel(r"$(\rho_{\mathrm{lin}}-\rho_{\mathrm{rot}})"
                   r"/\rho_{\mathrm{rot}}$   (per cent)")
    tidy(axr)

    # legend
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               handletextpad=0.6, columnspacing=2.0, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_no_agreement", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None,
                    choices=["landscape", "recovery", "agreement"])
    ap.add_argument("--out", default="results_final/figures")
    args = ap.parse_args()
    style()
    todo = {"landscape": fig_landscape, "recovery": fig_recovery,
            "agreement": fig_agreement}
    for name, fn in todo.items():
        if args.only in (None, name):
            print(name)
            fn(args.out)


if __name__ == "__main__":
    main()
