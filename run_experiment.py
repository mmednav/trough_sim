# -*- coding: utf-8 -*-
"""Batch experiment: simulate many parabolic troughs, evaluate all three
parabolicity algorithms on each, and write one CSV row per trough.

For each trough
    1. a defect scenario is drawn at random and recorded (ground truth),
    2. the mesh is built and scanned by the simulated UAV LiDAR,
    3. the ORIENTED parabolicity is computed on the cloud as generated,
       i.e. before any rotation is injected,
    4. a rotation delta ~ U(-D, D) is injected about the parabola vertex,
    5. the PERTURBATION model and the ROTATION SEARCH are run on the
       rotated cloud and compared against the oriented value.

Because the rotation is injected by us, the true orientation is known and the
oriented value plays the role of ground truth for the two non-oriented
algorithms.  The defect label makes the file usable for a confusion matrix.

Usage:
    python run_experiment.py --stratified --out results_new --csv results.csv
    python run_experiment.py --n 6 --out smoke --csv smoke.csv
"""
import csv
import os
import time

import numpy as np

from pipeline_loader import load

NS = load()

build_trough_panels = NS["build_trough_panels"]
build_receiver = NS["build_receiver"]
build_scene = NS["build_scene"]
run_uav_lidar_simulation = NS["run_uav_lidar_simulation"]
voxel_downsample_numpy = NS["voxel_downsample_numpy"]
project_to_cross_section = NS["project_to_cross_section"]
remez_parabolicity = NS["remez_parabolicity"]
rotate_points_about_vertex = NS["rotate_points_about_vertex"]
perturbation_parabolicity = NS["perturbation_parabolicity"]
rotation_search_parabolicity = NS["rotation_search_parabolicity"]


BASE_CFG = dict(NS["cfg"])


# ---------------------------------------------------------------------------
# experiment settings
# ---------------------------------------------------------------------------

SETTINGS = {
    "n_steps": 60,             # UAV stations per lane
    "dropout": 0.15,
    "voxel_size": 0.015,

    "p_defect": 0.5,           # unused, proportion of troughs with defects
    "defect_types": ("tilt", "dent"),
    "defect_block_max": (1, 1),   # panels along the trough, across aperture
    "dent_radius_m": 0.15,     # length scale of the Gaussian depression
    "dent_cutoff": 3.0,        # truncated at this many radii
    "amp_range_m": (0.002, 0.040),   # unsued, for random design only, log-uniform amplitude of errors
    "n_nominal": 150,
    "amp_grid_mm": (8, 12, 14, 16, 18, 21, 25, 32), # amplitudes used for defects
    "reps_per_amp": 10, #  number of troughs generated per error


    "base_tilt_std_deg": (0.05, 0.05, 0.01),     # baseline installation error present on every trough


    # non-oriented evaluation
    "delta_max_deg": 1.5,      # injected rotation ~ U(-delta_max, delta_max)
    "eps_deg": 2.0,            # trust region of the perturbation model
    "C_a": 0.5,
    "C_b": 2.0,
    "phi_max_rad": 0.14,      # +/- 8 deg; a deformed trough can pull well past
    "grid_points": 561,       # for rot search
}


def tilt_lever_arm(panel):
    """Largest distance from the rotation centre, in the plane of rotation.

    A panel tilted by beta about the trough axis displaces a vertex by
    beta times its distance from the centre measured in the xz plane.  The
    panel is a curved strip, so that distance is not the half-width: the
    departure of the surface in z contributes as well, by about four per
    cent at the aperture centre and more towards the rim.  Taking the true
    lever arm makes the realised peak displacement equal to the requested
    amplitude on every panel, so a single amplitude axis means the same
    thing across the collector.
    """
    v = np.asarray(panel["mesh"].vertices)
    c = panel.get("center", v.mean(axis=0))
    return float(np.max(np.hypot(v[:, 0] - c[0], v[:, 2] - c[2])))


def random_block(rng, Ny, Nx, block_max=(1, 1)):
    """A contiguous block of panels, as (i0, i1, j0, j1).

    `block_max` caps the extent along the trough and across the aperture.
    The default (1, 1) puts the defect on a single panel, which keeps the
    amplitude the only thing that varies between defective troughs: with a
    block of random size the realised amplitude drifts with the extent as
    well, since it is the largest of several independently jittered panels,
    and the detection curve then mixes "how deep" with "how many".
    """
    di = int(rng.integers(1, max(1, min(block_max[0], Ny)) + 1))
    dj = int(rng.integers(1, max(1, min(block_max[1], Nx)) + 1))
    i0 = int(rng.integers(0, Ny - di + 1))
    j0 = int(rng.integers(0, Nx - dj + 1))
    return i0, i0 + di, j0, j0 + dj


def in_block(panel, block):
    i0, i1, j0, j1 = block
    return i0 <= panel["i"] < i1 and j0 <= panel["j"] < j1


def apply_tilt_defect(panel, amp, rng):
    """Rotate the panel about the trough axis to a given peak displacement."""
    jitter = float(rng.uniform(0.85, 1.15))
    beta = amp * jitter / tilt_lever_arm(panel)
    panel["errors"]["tilt"]["enabled"] = True
    # compose with the baseline tilt already present rather than replacing it
    panel["errors"]["tilt"]["angles"] = (
        panel["errors"]["tilt"]["angles"] + np.array([0.0, beta, 0.0]))
    return amp * jitter


def apply_dent_defect(panel, amp, rng, settings):
    """A localised depression, placed so its support lies within the panel."""
    jitter = float(rng.uniform(0.85, 1.15))
    radius = float(settings["dent_radius_m"])
    cutoff = float(settings["dent_cutoff"])
    reach = cutoff * radius
    umax = max(0.0, 0.5 * panel["dx"] - reach)
    smax = max(0.0, 0.5 * panel["dy"] - reach)
    panel["errors"]["dent"]["enabled"] = True
    panel["errors"]["dent"]["center"] = np.array([
        rng.uniform(-umax, umax), rng.uniform(-smax, smax)])
    panel["errors"]["dent"]["amplitude"] = -amp * jitter
    panel["errors"]["dent"]["radius"] = radius
    panel["errors"]["dent"]["cutoff"] = cutoff
    return amp * jitter


def make_scenario(panels, rng, settings, forced=None):
    # to run the simulation
    tilt_std = np.array(settings["base_tilt_std_deg"], dtype=float)

    for p in panels:
        for key in p["errors"]:
            p["errors"][key]["enabled"] = False
        p["errors"]["tilt"]["enabled"] = True
        p["errors"]["tilt"]["angles"] = np.deg2rad(rng.normal(0.0, tilt_std))

    if forced is None:
        label = int(rng.random() < settings["p_defect"])
    else:
        label = int(forced["label"])

    scenario = {
        "label": label,
        "defect_type": "none",
        "defect_amplitude_m": 0.0,
        "defect_amplitude_realised_m": 0.0,
        "n_defective_panels": 0,
        "defect_i0": -1, "defect_i1": -1,
        "defect_j0": -1, "defect_j1": -1,
        "defect2_i0": -1, "defect2_i1": -1,
        "defect2_j0": -1, "defect2_j1": -1,
    }

    if not label:
        return scenario


    if forced is not None and forced.get("defect_type"):
        dtype = forced["defect_type"]
    else:
        dtype = settings["defect_types"][
            rng.integers(len(settings["defect_types"]))]

    if forced is None:
        lo, hi = settings["amp_range_m"]
        amp = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    else:
        amp = float(forced["amplitude_m"])

    Ny, Nx = panels[0]["Ny"], panels[0]["Nx"]
    bmax = tuple(settings.get("defect_block_max", (1, 1)))

    if dtype == "tilt":
        block_tilt, block_dent = random_block(rng, Ny, Nx, bmax), None
    elif dtype == "dent":
        block_tilt, block_dent = None, random_block(rng, Ny, Nx, bmax)
    else:
        raise ValueError("unknown defect type: %s" % dtype)

    n_hit = 0
    realised = 0.0
    for p in panels:
        if block_tilt is not None and in_block(p, block_tilt):
            realised = max(realised, apply_tilt_defect(p, amp, rng))
            n_hit += 1
        elif block_dent is not None and in_block(p, block_dent):
            realised = max(realised, apply_dent_defect(p, amp, rng, settings))
            n_hit += 1

    primary = block_tilt if block_tilt is not None else block_dent

    scenario.update({
        "defect_type": dtype,
        "defect_amplitude_m": amp,
        "defect_amplitude_realised_m": realised,
        "n_defective_panels": n_hit,
        "defect_i0": primary[0], "defect_i1": primary[1],
        "defect_j0": primary[2], "defect_j1": primary[3],
    })
    return scenario


def _panel_lookup(geometry_id_to_panel, field):
    # panel row or column, as a lookup array.
    size = max(geometry_id_to_panel) + 1
    out = np.full(size, -1, dtype=np.int16)
    for gid, meta in geometry_id_to_panel.items():
        out[int(gid)] = int(meta[field])
    return out


def panel_points_noisy(lidar, geometry_id_to_panel):
    # points that hit a mirror panel, taken from the noisy cloud.

    ids = lidar["geometry_ids"].astype(int)
    panel_ids = set(int(k) for k in geometry_id_to_panel.keys())
    keep = np.fromiter((g in panel_ids for g in ids), dtype=bool, count=len(ids))
    return lidar["noisy_points"][keep], lidar["raw_points"][keep], ids[keep]


 # build csv row
def run_one(run_id, seed, settings, cfg, forced=None):
    rng = np.random.default_rng(seed)
    t_start = time.perf_counter()

    row = {"run_id": run_id, "seed": seed}
    row.update({k: cfg[k] for k in ("W", "L", "f", "Nx", "Ny", "noise_sigma")})
    row["voxel_size"] = settings["voxel_size"]
    row["n_steps"] = settings["n_steps"]

    # ---- geometry and defects ------------------------------------------
    panels = build_trough_panels(cfg["W"], cfg["L"], cfg["f"],
                                 cfg["Nx"], cfg["Ny"],
                                 cfg["sub_x"], cfg["sub_y"])
    scenario = make_scenario(panels, rng, settings, forced=forced)
    row.update(scenario)

    receiver = build_receiver(cfg["L"], cfg["f"], cfg["receiver_radius"])
    scene, _, _, _, g2p = build_scene(cfg, panels, receiver, False, True)

    # ---- scan ----------------------------------------------------------
    lidar = run_uav_lidar_simulation(
        scene, cfg,
        n_steps=settings["n_steps"],
        voxel_size=None,
        noise_sigma=cfg["noise_sigma"],
        dropout=settings["dropout"],
        plot=False,
        seed=int(rng.integers(2 ** 31)),
    )

    noisy, clean, hit_ids = panel_points_noisy(lidar, g2p)
    row["n_raw"] = int(len(noisy))

    P = voxel_downsample_numpy(noisy, settings["voxel_size"])
    row["n_points"] = int(len(P))

    # for saving the cloud
    if settings.get("cloud_dir"):
        os.makedirs(settings["cloud_dir"], exist_ok=True)
        np.savez_compressed(
            os.path.join(settings["cloud_dir"], "trough_%04d.npz" % run_id),
            points=noisy.astype(np.float32),
            clean=clean.astype(np.float32),
            geometry_id=hit_ids.astype(np.int32),
            panel_i=_panel_lookup(g2p, "i")[hit_ids],
            panel_j=_panel_lookup(g2p, "j")[hit_ids],
            run_id=run_id, seed=seed,
            label=scenario["label"],
            defect_type=str(scenario["defect_type"]),
            amplitude_m=scenario["defect_amplitude_m"],
            amplitude_realised_m=scenario["defect_amplitude_realised_m"],
            voxel_size=settings["voxel_size"],
            noise_sigma=cfg["noise_sigma"],
        )

    xz = project_to_cross_section(P)

    # effective noise after voxel averaging, measured against the fit
    # (filled in below once we have the oriented fit)

    # ---- 1. ORIENTED, before any rotation is injected -------------------
    t = time.perf_counter()
    ori = remez_parabolicity(xz)
    row["t_oriented"] = time.perf_counter() - t
    row["rho_oriented"] = ori["width"]
    row["rho_lo"] = ori["rho_lo"]
    row["rho_up"] = ori["rho_up"]
    row["remez_iters"] = ori["iterations"]
    row["remez_converged"] = int(ori["converged"])

    row["remez_status"] = ori["status"]
    row["remez_ties"] = ori["n_tied_abscissae"]
    row["remez_min_gap"] = ori["min_abscissa_gap"]
    row["remez_cond"] = ori["max_reference_condition"]

    a_f, b_f, c_f = ori["coeffs"]
    resid = xz[:, 1] - (a_f * xz[:, 0] ** 2 + b_f * xz[:, 0] + c_f)
    row["sigma_eff"] = float(np.std(resid))
    row["fitted_a"] = float(a_f)
    row["fitted_b"] = float(b_f)

    # ---- 2. inject a random rotation about the vertex -------------------
    dmax = np.deg2rad(settings["delta_max_deg"])
    delta = float(rng.uniform(-dmax, dmax))
    row["delta_injected_rad"] = delta
    row["delta_injected_deg"] = float(np.rad2deg(delta))

    obs = rotate_points_about_vertex(xz, delta)

    a_nom = 1.0 / (4.0 * cfg["f"])
    row["a_nominal"] = a_nom

    # parabolicity of the rotated cloud without any correction, for reference
    row["rho_rotated_uncorrected"] = remez_parabolicity(obs)["width"]

    # ---- 3. PERTURBATION model -----------------------------------------
    eps = np.deg2rad(settings["eps_deg"])
    row["eps_deg"] = settings["eps_deg"]
    row["C_a"] = settings["C_a"]
    row["C_b"] = settings["C_b"]

    t = time.perf_counter()
    pert = perturbation_parabolicity(obs, a_nom, b_nom=0.0, eps=eps,
                                     C_a=settings["C_a"], C_b=settings["C_b"])
    row["t_pert"] = time.perf_counter() - t
    if pert["success"]:
        row["rho_pert"] = pert["width"]
        row["pert_fitted_delta_deg"] = float(np.rad2deg(pert["delta"]))
        row["pert_delta_err_deg"] = row["pert_fitted_delta_deg"] + row["delta_injected_deg"]
        row["pert_box_delta"] = int(pert["box_active"]["delta"])
        row["pert_box_da"] = int(pert["box_active"]["da"])
        row["pert_box_db"] = int(pert["box_active"]["db"])
    else:
        for k in ("rho_pert", "pert_fitted_delta_deg", "pert_delta_err_deg"):
            row[k] = float("nan")
        for k in ("pert_box_delta", "pert_box_da", "pert_box_db"):
            row[k] = -1

    # ---- 4. ROTATION SEARCH ---------------------------------------------
    t = time.perf_counter()
    grid = rotation_search_parabolicity(obs,
                                        phi_max=settings["phi_max_rad"],
                                        n_grid=settings["grid_points"])
    row["t_grid"] = time.perf_counter() - t
    row["rho_grid"] = grid["width"]
    row["phi_recovered_deg"] = float(np.rad2deg(grid["phi"]))
    row["phi_error_deg"] = row["phi_recovered_deg"] + row["delta_injected_deg"]
    row["grid_points"] = grid["n_grid"]
    row["grid_spacing_rad"] = grid["spacing"]
    row["grid_at_boundary"] = int(grid["at_boundary"])


    kappa = 2.0 * a_nom

    # ---- derived comparisons -------------------------------------------
    row["pert_minus_oriented"] = row["rho_pert"] - row["rho_oriented"]
    row["grid_minus_oriented"] = row["rho_grid"] - row["rho_oriented"]
    row["pert_ratio"] = row["rho_pert"] / row["rho_oriented"]
    row["grid_ratio"] = row["rho_grid"] / row["rho_oriented"]
    # the non-oriented width is a minimum over directions and the nominal axis
    # is one of them, so the search must not exceed the oriented value
    row["grid_le_oriented"] = int(row["rho_grid"] <= row["rho_oriented"] + 1e-12)

    row["t_total"] = time.perf_counter() - t_start
    return row


def build_schedule(settings, seed=0):

    rng = np.random.default_rng(seed)
    sched = [{"label": 0, "amplitude_m": 0.0} for _ in range(settings["n_nominal"])]

    for kind in settings["defect_types"]:
        for amp_mm in settings["amp_grid_mm"]:
            for _ in range(settings["reps_per_amp"]):
                sched.append({"label": 1, "amplitude_m": amp_mm / 1000.0,
                              "defect_type": kind})
    rng.shuffle(sched)
    return sched





def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stratified", action="store_true",
                    help="dedicated nominal block + amplitude grid with replicates")
    ap.add_argument("--n", type=int, default=30,
                    help="number of troughs (random design only)")
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--csv", default="results_troughs.csv", help="file name inside --out")
    ap.add_argument("--save-clouds", action="store_true",
                    help="archive each trough's point cloud under <out>/clouds "
                         "so the analysis can be redone without re-simulating")
    ap.add_argument("--seed0", type=int, default=20260916,
                    help="base seed; trough k uses seed0 + k")
    ap.add_argument("--force", action="store_true",
                    help="allow writing over an existing CSV or cloud archive")
    args = ap.parse_args()

    cfg = dict(BASE_CFG)
    os.makedirs(args.out, exist_ok=True)
    if args.save_clouds:
        SETTINGS["cloud_dir"] = os.path.join(args.out, "clouds")
    out_path = os.path.join(args.out, args.csv)

    # Refuse to destroy a previous batch.
    if not args.force:
        clash = []
        if os.path.exists(out_path):
            clash.append(out_path)
        if args.save_clouds:
            cdir = SETTINGS["cloud_dir"]
            if os.path.isdir(cdir) and os.listdir(cdir):
                clash.append("%s (%d files)" % (cdir, len(os.listdir(cdir))))
        if clash:
            raise SystemExit(
                "refusing to overwrite:\n"
                + "".join("    %s\n" % c for c in clash)
                + "re-run with --force, or choose another --out/--csv.\n"
                "note that a repeat run does not reproduce an earlier batch\n"
                "byte for byte unless the code and the seed are both unchanged.")

    if args.stratified:
        schedule = build_schedule(SETTINGS)
        n_troughs = len(schedule)
        design = "stratified: %d nominal + %d amplitudes x %d reps" % (
            SETTINGS["n_nominal"], len(SETTINGS["amp_grid_mm"]),
            SETTINGS["reps_per_amp"])
    else:
        schedule = [None] * args.n
        n_troughs = args.n
        design = "random: p_defect = %.2f, amplitude log-uniform %g-%g m" % (
            SETTINGS["p_defect"], *SETTINGS["amp_range_m"])

    print("running %d troughs -> %s" % (n_troughs, out_path))
    print("design: %s" % design)
    print("cfg: W=%.2f L=%.1f f=%.2f  %dx%d panels  sigma=%.4f  voxel=%.3f"
          % (cfg["W"], cfg["L"], cfg["f"], cfg["Ny"], cfg["Nx"],
             cfg["noise_sigma"], SETTINGS["voxel_size"]))
    print()
    hdr = ("  run  label  defect        amp(mm)  n_pts   rho_ori   rho_pert  "
           "rho_grid   delta(deg)  phi_err(deg)   t(s)")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    t0 = time.perf_counter()
    for k, forced in enumerate(schedule):
        try:
            row = run_one(k, args.seed0 + k, SETTINGS, cfg, forced=forced)
        except Exception as exc:                    # keep the batch alive
            print("  %3d  FAILED: %s" % (k, exc))
            continue
        rows.append(row)
        if k % 5 == 0 or k == n_troughs - 1:
            print("  %3d    %d    %-12s %6.1f  %6d  %.6f  %.6f  %.6f  %+8.3f   %+8.3f  %6.2f"
                  % (row["run_id"], row["label"], row["defect_type"],
                     1000 * row["defect_amplitude_m"], row["n_points"],
                     row["rho_oriented"], row["rho_pert"], row["rho_grid"],
                     row["delta_injected_deg"], row["phi_error_deg"],
                     row["t_total"]))

    if not rows:
        print("no successful runs")
        return

    fields = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)

    with open(out_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print()
    print("wrote %d rows x %d columns to %s  (%.1f s total)"
          % (len(rows), len(fields), os.path.abspath(out_path),
             time.perf_counter() - t0))


if __name__ == "__main__":
    main()
