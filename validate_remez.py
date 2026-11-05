# -*- coding: utf-8 -*-

import argparse
import sys
import time

import numpy as np

from pipeline_loader import load

NS = load()
remez_parabolicity = NS["remez_parabolicity"]
oriented_parabolicity = NS["oriented_parabolicity"]
CFG = dict(NS["cfg"])


def synthetic_section(n, sigma, rng, cfg=None):
   # create a simulated section with gaussian noise
    cfg = CFG if cfg is None else cfg
    x = rng.uniform(-cfg["W"] / 2.0, cfg["W"] / 2.0, n)
    z = x ** 2 / (4.0 * cfg["f"]) + rng.normal(0.0, sigma, n)
    return np.stack([x, z], axis=1)


def one_run(n, sigma, rng):
    P = synthetic_section(n, sigma, rng)

    t0 = time.perf_counter()
    lp = oriented_parabolicity(P)
    t_lp = time.perf_counter() - t0

    t0 = time.perf_counter()
    rz = remez_parabolicity(P)
    t_rz = time.perf_counter() - t0

    return {
        "n": n,
        "sigma": sigma,
        "rho_lp": float(lp.width),
        "rho_rz": float(rz["width"]),
        "rel": abs(rz["width"] - lp.width) / lp.width,
        "t_lp": t_lp,
        "t_rz": t_rz,
        "iters": rz["iterations"],
        "cond": rz["max_reference_condition"],
        "status": rz["status"],
        "gap": rz["rho_up"] - rz["rho_lo"],
    }


def summarise(runs):
    # Collapse replicates of one setting into a row.
    a = lambda k: np.array([r[k] for r in runs], dtype=float)
    return {
        "n": runs[0]["n"],
        "sigma": runs[0]["sigma"],
        "reps": len(runs),
        "rho": a("rho_lp").mean(),
        "rho_sd": a("rho_lp").std(ddof=1) if len(runs) > 1 else 0.0,
        "rel_max": a("rel").max(),
        "rel_mean": a("rel").mean(),
        "iters": a("iters").mean(),
        "cond": a("cond").max(),
        "t_lp": a("t_lp").mean(),
        "t_rz": a("t_rz").mean(),
        "speedup": a("t_lp").mean() / a("t_rz").mean(),
        "clean": all(r["status"] == "converged" for r in runs),
    }


def table(rows, title, varying):
    print()
    print(title)
    head = ("  %-11s %6s  %9s %9s   %9s %9s  %6s  %8s %8s %7s"
            % (varying, "reps", "rho (mm)", "sd (mm)", "max|rel|",
               "mean|rel|", "iters", "t_LP(s)", "t_Rz(s)", "speedup"))
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in rows:
        label = ("%.1f mm" % (1000 * r["sigma"]) if varying.startswith("noise")
                 else "%d" % r["n"])
        print("  %-11s %6d  %9.4f %9.4f   %9.2e %9.2e  %6.1f  %8.3f %8.3f %7.1fx%s"
              % (label, r["reps"], 1000 * r["rho"], 1000 * r["rho_sd"],
                 r["rel_max"], r["rel_mean"], r["iters"],
                 r["t_lp"], r["t_rz"], r["speedup"],
                 "" if r["clean"] else "  !!"))


def compare_on_clouds(cloud_dir, csv_path=None, limit=None):
    # run both solvers on the archived clouds of a completed batch.

    import glob
    import os

    voxel_downsample_numpy = NS["voxel_downsample_numpy"]
    project_to_cross_section = NS["project_to_cross_section"]

    recorded = {}
    if csv_path and os.path.exists(csv_path):
        import csv as _csv
        for r in _csv.DictReader(open(csv_path, encoding="utf-8")):
            recorded[int(r["run_id"])] = float(r["rho_oriented"])

    files = sorted(glob.glob(os.path.join(cloud_dir, "trough_*.npz")))
    if limit:
        files = files[:limit]
    if not files:
        sys.exit("no clouds found in %s" % cloud_dir)

    rows = []
    for k, path in enumerate(files):
        z = np.load(path, allow_pickle=False)
        P_ds = voxel_downsample_numpy(z["points"].astype(float),
                                      float(z["voxel_size"]))
        xz = project_to_cross_section(P_ds)

        t0 = time.perf_counter()
        lp = oriented_parabolicity(xz)
        t_lp = time.perf_counter() - t0

        t0 = time.perf_counter()
        rz = remez_parabolicity(xz)
        t_rz = time.perf_counter() - t0

        rid = int(z["run_id"])
        rows.append({
            "run_id": rid, "n": len(xz),
            "lp": float(lp.width), "rz": float(rz["width"]),
            "rel": abs(rz["width"] - lp.width) / lp.width,
            "t_lp": t_lp, "t_rz": t_rz,
            "nit_lp": int(lp.nit), "nit_rz": int(rz["iterations"]),
            "vs_csv": (abs(rz["width"] - recorded[rid]) / recorded[rid]
                       if rid in recorded else float("nan")),
        })
        if (k + 1) % 25 == 0:
            print("   %d/%d" % (k + 1, len(files)))

    a = lambda key: np.array([r[key] for r in rows], dtype=float)
    print()
    print("ARCHIVED CLOUDS  (%d troughs, median %d points)"
          % (len(rows), int(np.median(a("n")))))
    print()
    print("  %-34s %s" % ("agreement, worst |rel| difference",
                          "%.2e" % a("rel").max()))
    print("  %-34s %s" % ("agreement, mean |rel| difference",
                          "%.2e" % a("rel").mean()))
    if recorded:
        print("  %-34s %s" % ("recomputed vs the batch CSV, worst",
                              "%.2e" % np.nanmax(a("vs_csv"))))
    print()
    print("  %-34s %7.3f s   (%d iterations)"
          % ("linear program, mean", a("t_lp").mean(),
             int(round(a("nit_lp").mean()))))
    print("  %-34s %7.4f s   (%d iterations)"
          % ("exchange algorithm, mean", a("t_rz").mean(),
             int(round(a("nit_rz").mean()))))
    print("  %-34s %7.1f x" % ("speed-up",
                               a("t_lp").mean() / a("t_rz").mean()))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--clouds", default=None,
                    help="compare on a batch's archived clouds instead of "
                         "running the synthetic sweep")
    ap.add_argument("--batch-csv", default=None,
                    help="that batch's CSV, to check the archive round-trips")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.clouds:
        compare_on_clouds(args.clouds, args.batch_csv, args.limit)
        return

    rng = np.random.default_rng(args.seed)
    sigmas = (0.0005, 0.001, 0.002, 0.003, 0.005, 0.010)

    sizes = (5000, 25000, 83400, 250000)
    # the noise sweep runs at the median cloud size of the experiment itself,
    # so it is measured at the density the results were obtained at
    n_fixed, sigma_fixed = 83400, 0.003


    one_run(2000, 0.003, np.random.default_rng(0))

    total = len(sigmas) * args.reps + len(sizes) * args.reps
    print("validating over %d runs (%d settings x %d replicates)"
          % (total, len(sigmas) + len(sizes), args.reps))

    rows_a = []
    for s in sigmas:
        runs = [one_run(n_fixed, s, rng) for _ in range(args.reps)]
        rows_a.append(summarise(runs))
        print("   sigma = %5.1f mm done" % (1000 * s))

    rows_b = []
    for n in sizes:
        runs = [one_run(n, sigma_fixed, rng) for _ in range(args.reps)]
        rows_b.append(summarise(runs))
        print("   n = %6d done" % n)

    table(rows_a, "A. NOISE SWEPT  (n = %d points)" % n_fixed, "noise sigma")
    table(rows_b, "B. CLOUD SIZE SWEPT  (sigma = %.1f mm)"
          % (1000 * sigma_fixed), "n")

    every = rows_a + rows_b
    worst = max(r["rel_max"] for r in every)
    print()
    print("  worst relative departure over all %d runs : %.2e" % (total, worst))
    print("  double-precision epsilon                  : %.2e" % np.finfo(float).eps)
    print("  all references converged                  : %s"
          % all(r["clean"] for r in every))
    print("  worst 4x4 condition number                : %.1f"
          % max(r["cond"] for r in every))



if __name__ == "__main__":
    main()
