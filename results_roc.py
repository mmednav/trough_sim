# -*- coding: utf-8 -*-

import argparse
import csv
import sys

import numpy as np


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        sys.exit("no rows in %s" % path)
    out = {}
    for key in rows[0]:
        try:
            out[key] = np.array([float(r[key]) for r in rows])
        except ValueError:
            out[key] = np.array([r[key] for r in rows])
    return out


def auc_mw(score, lab):

    pos, neg = score[lab == 1], score[lab == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average the ranks within each group of ties
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def auc_ci(score, lab, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(score)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        if len(np.unique(lab[idx])) < 2:
            continue
        vals.append(auc_mw(score[idx], lab[idx]))
    if not vals:
        return (float("nan"), float("nan"))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def roc_curve(score, lab):
    order = np.argsort(-score, kind="mergesort")
    s, y = score[order], lab[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    n_pos, n_neg = max(1, int((lab == 1).sum())), max(1, int((lab == 0).sum()))
    # keep only the last index of each run of equal scores
    keep = np.r_[np.diff(s) != 0, True]
    return (np.r_[0.0, fp[keep] / n_neg], np.r_[0.0, tp[keep] / n_pos],
            np.r_[np.inf, s[keep]])


def rate_at(score, lab, tau):
    tp = int(((score > tau) & (lab == 1)).sum())
    fn = int(((score <= tau) & (lab == 1)).sum())
    fp = int(((score > tau) & (lab == 0)).sum())
    tn = int(((score <= tau) & (lab == 0)).sum())
    return tp, fp, fn, tn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="?",
                    default="results_final/results_troughs_final.csv")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="false positive rate the operating threshold targets")
    ap.add_argument("--bands-mm", type=float, nargs="+",
                    default=[0, 5, 10, 20, 100])
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()

    d = load(args.csv)
    lab = d["label"].astype(int)
    amp = d["defect_amplitude_m"] * 1000.0
    dtype = d["defect_type"]

    methods = [("oriented", "rho_oriented")]
    for label, col in (("perturbation", "rho_pert"),
                       ("rotation search", "rho_grid")):
        if col in d:
            methods.append((label, col))

    n_nom, n_def = int((lab == 0).sum()), int((lab == 1).sum())
    print("=" * 76)
    print("%s   |   %d nominal, %d defective" % (args.csv, n_nom, n_def))
    print("=" * 76)

    # ---- 1. separation, over all thresholds ---------------------------
    print()
    print("1. AREA UNDER THE ROC  (separation, independent of any threshold)")
    print()
    print("   %-16s %7s   %-18s" % ("method", "AUC", "95% CI (bootstrap)"))
    for name, col in methods:
        a = auc_mw(d[col], lab)
        lo, hi = auc_ci(d[col], lab, reps=args.reps)
        print("   %-16s %7.4f   [%.4f, %.4f]" % (name, a, lo, hi))
    print()

    # ---- 2. operating threshold ---------------------------------------
    primary, pcol = methods[0]
    score = d[pcol]
    nom = score[lab == 0]
    tau_q = float(np.quantile(nom, 1 - args.alpha))
    print()
    print("2. OPERATING THRESHOLD  (on %s)" % primary)
    print()
    print("   from the nominal block, tau_%.2f = %.6f m" % (args.alpha, tau_q))
    tp, fp, fn, tn = rate_at(score, lab, tau_q)
    print("      -> TPR %.3f   FPR %.3f   (%d/%d, %d/%d)"
          % (tp / max(1, tp + fn), fp / max(1, fp + tn), tp, tp + fn, fp, fp + tn))


    # ---- 3. detection against amplitude -------------------------------
    print()
    print("3. DETECTION AGAINST AMPLITUDE  at tau_%.2f" % args.alpha)
    print()
    edges = args.bands_mm
    print("   %-14s %6s  %-16s" % ("band", "n", "detected"))
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (lab == 1) & (amp >= a) & (amp < b)
        n = int(sel.sum())
        if not n:
            continue
        hit = int((score[sel] > tau_q).sum())
        print("   %-14s %6d  %3d/%-3d  %5.0f%%"
              % ("%g-%g mm" % (a, b), n, hit, n, 100.0 * hit / n))

    # ---- 4. by defect type --------------------------------------------
    kinds = sorted(set(dtype[lab == 1]))
    if len(kinds) > 1:
        print()
        print("4. BY DEFECT TYPE  (AUC against the same nominal block)")
        print()
        print("   %-14s %6s %7s   %-18s" % ("type", "n", "AUC", "95% CI"))
        for k in kinds:
            sel = (lab == 0) | ((lab == 1) & (dtype == k))
            a = auc_mw(score[sel], lab[sel])
            lo, hi = auc_ci(score[sel], lab[sel], reps=args.reps)
            print("   %-14s %6d %7.4f   [%.4f, %.4f]"
                  % (k, int(((lab == 1) & (dtype == k)).sum()), a, lo, hi))


if __name__ == "__main__":
    main()
