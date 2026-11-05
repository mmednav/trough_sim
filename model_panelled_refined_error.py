import os
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def build_trough_panels(W, L, f, Nx, Ny, sub_x=6, sub_y=6):
    """
    Build ideal parabolic trough panels as subdivided meshes.

    W, L  : total trough width and length
    f     : focal length
    Nx    : number of panels across aperture
    Ny    : number of panels along length
    sub_x : subdivisions per panel across aperture
    sub_y : subdivisions per panel along length
    """

    dx = W / Nx
    dy = L / Ny

    panels = []

    for i in range(Ny):
        for j in range(Nx):

            x0 = -W / 2 + j * dx
            x1 = x0 + dx

            y0 = i * dy
            y1 = y0 + dy

            xs = np.linspace(x0, x1, sub_x + 1)
            ys = np.linspace(y0, y1, sub_y + 1)

            vertices = []

            for y in ys:
                for x in xs:
                    z = x**2 / (4 * f)
                    vertices.append([x, y, z])

            vertices = np.array(vertices)

            triangles = []

            for iy in range(sub_y):
                for ix in range(sub_x):

                    v00 = iy * (sub_x + 1) + ix
                    v10 = v00 + 1
                    v01 = (iy + 1) * (sub_x + 1) + ix
                    v11 = v01 + 1

                    triangles.append([v00, v10, v11])
                    triangles.append([v00, v11, v01])

            triangles = np.array(triangles)

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.compute_vertex_normals()

            center = np.mean(vertices, axis=0)

            panel = {
                "mesh": mesh,

                # panel ID
                "i": i,
                "j": j,

                # global trough geometry
                "W": W,
                "L": L,
                "f": f,
                "Nx": Nx,
                "Ny": Ny,

                # local panel geometry
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "dx": dx,
                "dy": dy,
                "sub_x": sub_x,
                "sub_y": sub_y,
                "center": center,

                # cleanly separated error model
                "errors": {
                    "tilt": {
                        "enabled": False,
                        "angles": np.zeros(3)
                    },
                    "translation": {
                        "enabled": False,
                        "vector": np.zeros(3)
                    },
                    "gravity_sag": {
                        "enabled": False,
                        "amplitude": 0.0
                    },
                    "bow": {
                        "enabled": False,
                        "amplitude": 0.0
                    },
                    "dent": {
                        "enabled": False,
                        "center": np.zeros(2),
                        "amplitude": 0.0,
                        "radius": 0.0,
                        "cutoff": 3.0
                    }
                }
            }

            panels.append(panel)

    return panels


def build_receiver(L=12.0, f=2.0, r=0.04):
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=r,
        height=L
    )

    ## rotate so cylinder axis aligns with Y
    #R = cylinder.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
    #cylinder.rotate(R, center=(0, 0, 0))

    ## move it to focal line (centered in y)
    #cylinder.translate((0, L/2, f))

    cylinder.compute_vertex_normals()

    return cylinder


def apply_panel_errors(panel, copy=True, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    mesh = panel["mesh"]
    if copy:
        mesh = o3d.geometry.TriangleMesh(mesh)

    errors = panel.get("errors", {})
    center = panel.get("center", np.mean(np.asarray(mesh.vertices), axis=0))

    # 1. Rigid tilt
    if errors.get("tilt", {}).get("enabled", False):
        angles = errors["tilt"]["angles"]
        R = mesh.get_rotation_matrix_from_xyz(angles)
        mesh.rotate(R, center=center)

    # 2. Rigid translation
    if errors.get("translation", {}).get("enabled", False):
        vector = errors["translation"]["vector"]
        mesh.translate(vector)

    # Recompute normals before shape errors
    mesh.compute_vertex_normals()

    # 3. Shape errors
    if errors.get("gravity_sag", {}).get("enabled", False):
        amp = errors["gravity_sag"]["amplitude"]
        mesh = apply_gravity_sag(mesh, panel, amp)

    if errors.get("dent", {}).get("enabled", False):
        dent = errors["dent"]
        mesh = apply_dent(
            mesh,
            panel,
            center_uv=dent["center"],
            amplitude=dent["amplitude"],
            radius=dent["radius"],
            cutoff=dent.get("cutoff", 3.0)
        )

    mesh.compute_vertex_normals()
    return mesh

# Supporting functions for errors
def local_panel_coordinates(mesh, panel):
    v = np.asarray(mesh.vertices)

    x_mid = 0.5 * (panel["x0"] + panel["x1"])
    y_mid = 0.5 * (panel["y0"] + panel["y1"])

    u = v[:, 0] - x_mid
    s = v[:, 1] - y_mid

    return u, s


def apply_gravity_sag(mesh, panel, amplitude):
    mesh.compute_vertex_normals()

    v = np.asarray(mesh.vertices)
    n = np.asarray(mesh.vertex_normals)

    u, s = local_panel_coordinates(mesh, panel)

    dx = panel["dx"]
    dy = panel["dy"]

    # normalized coordinates: -1 to 1
    U = 2.0 * u / dx
    S = 2.0 * s / dy

    # hammock sag: zero at all edges, max at center
    shape = (1.0 - U**2) * (1.0 - S**2)
    shape = np.clip(shape, 0.0, None)

    v_new = v + amplitude * shape[:, None] * n

    mesh.vertices = o3d.utility.Vector3dVector(v_new)
    mesh.compute_vertex_normals()

    return mesh



def apply_dent(mesh, panel, center_uv, amplitude, radius, cutoff=3.0):
    """Localised Gaussian depression, truncated at `cutoff` radii.

    The Gaussian has unbounded support, so without truncation a dent of any
    appreciable radius is felt across the whole panel: at radius 0.43 m on a
    1.44 m panel the profile is still at a quarter of full depth at the edge,
    which is a dish rather than a dent.

    Truncating alone would leave a step of exp(-cutoff^2/2) of the amplitude
    at the boundary - 1.1% at the default cutoff of three radii, a fifth of a
    millimetre on a 20 mm dent - so the profile is shifted by that value and
    renormalised. The result vanishes continuously at `cutoff` radii while
    still reaching exactly 1 at the centre, so the stated amplitude remains
    the true peak departure from the nominal surface.
    """
    mesh.compute_vertex_normals()

    v = np.asarray(mesh.vertices)
    n = np.asarray(mesh.vertex_normals)

    u, s = local_panel_coordinates(mesh, panel)

    uc, sc = center_uv

    r2 = (u - uc) ** 2 + (s - sc) ** 2

    floor = np.exp(-0.5 * cutoff ** 2)          # value at the truncation radius
    shape = (np.exp(-r2 / (2.0 * radius ** 2)) - floor) / (1.0 - floor)
    shape[r2 > (cutoff * radius) ** 2] = 0.0
    np.clip(shape, 0.0, None, out=shape)

    v_new = v + amplitude * shape[:, None] * n

    mesh.vertices = o3d.utility.Vector3dVector(v_new)
    mesh.compute_vertex_normals()

    return mesh


def build_support_legs(W, L, height=0.8):

    legs = []
    xs = [-W/2, W/2]
    ys = [0, L]

    for x in xs:
        for y in ys:
            leg = o3d.geometry.TriangleMesh.create_cylinder(radius=0.05, height=height)
            leg.translate([x, y, height/2])
            legs.append(leg)

    return legs


def build_receiver_support(L, f):

    beam = o3d.geometry.TriangleMesh.create_box(width=0.1, height=L, depth=0.1)
    beam.translate([-0.05, 0, f - 0.2])
    return beam


def build_receiver_radial_supports(cfg, Nx_s=4, Ny_s=6):

    W, L, f = cfg["W"], cfg["L"], cfg["f"]

    supports = []

    xs = np.linspace(-W/2, W/2, Nx_s)
    ys = np.linspace(0, L, Ny_s)

    for y in ys:
        for x in xs:

            # trough surface
            z = (x**2) / (4*f)

            p0 = np.array([x, y, z])
            p1 = np.array([0.0, y, f])

            vec = p1 - p0
            length = np.linalg.norm(vec)

            if length < 1e-6:
                continue

            cyl = o3d.geometry.TriangleMesh.create_cylinder(
                radius=0.015,
                height=length
            )

            direction = vec / length
            z_axis = np.array([0, 0, 1])

            axis = np.cross(z_axis, direction)
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))

            if np.linalg.norm(axis) > 1e-6:
                axis = axis / np.linalg.norm(axis)
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
                cyl.rotate(R, center=(0, 0, 0))

            cyl.translate((p0 + p1) / 2)

            supports.append(cyl)

    return supports


# Optional for receiver rings, not included yet.


# Plot scene


# Set custom directories depending on parameters
def make_scene_dir(cfg):
    return f"simulations/f_{cfg['f']}_W_{cfg['W']}_L_{cfg['L']}_error"


def build_scene(cfg, panels, receiver, plot=False, add_support=False):

    # -------------------------
    # output folder
    # -------------------------
    out_dir = make_scene_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # scene
    # -------------------------
    scene = o3d.t.geometry.RaycastingScene()

    geometry_id_to_panel = {}
    trough_meshes = []

    for idx, p in enumerate(panels):
        mesh = apply_panel_errors(p, copy=True)
        mesh.compute_vertex_normals()

        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        geom_id = scene.add_triangles(tmesh)

        geometry_id_to_panel[int(geom_id)] = {
            "panel_index": idx,
            "i": p["i"],
            "j": p["j"],
            "errors": p["errors"],
        }

        trough_meshes.append(mesh)

    # -------------------------
    # merge trough for export/plot
    # -------------------------
    trough = trough_meshes[0]
    for m in trough_meshes[1:]:
        trough += m  # Open3D mesh merge

    trough.compute_vertex_normals()

    # -------------------------
    # receiver placement (WORLD SPACE ONLY)
    # -------------------------
    R = receiver.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
    receiver.rotate(R, center=(0, 0, 0))
    receiver.translate((0, cfg["L"]/2, cfg["f"]))
    receiver.compute_vertex_normals()

    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(receiver))

    # -------------------------
    # add support to scene
    # -------------------------
    if add_support:
        supports = build_support_legs(cfg["W"], cfg["L"])
        receiver_support = build_receiver_support(cfg["L"], cfg["f"])
        radial_supports = build_receiver_radial_supports(cfg)

        # plotting
        all_visual_meshes = []
        all_visual_meshes += supports
        all_visual_meshes.append(receiver_support)
        all_visual_meshes += radial_supports

        # add to scene
        for s in supports:
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(s))

        for s in radial_supports:
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(s))

        scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(receiver_support)
        )

    # -------------------------
    # export
    # -------------------------
    o3d.io.write_triangle_mesh(
        os.path.join(out_dir, "trough.ply"),
        trough
    )

    o3d.io.write_triangle_mesh(
        os.path.join(out_dir, "receiver.ply"),
        receiver
    )


    with open(os.path.join(out_dir, "config.json"), "w") as fp:
        json.dump(cfg, fp, indent=4)

    # -------------------------
    # plot (optional)
    # -------------------------
    if plot:
        if add_support:
            o3d.visualization.draw_geometries([trough, receiver] + all_visual_meshes)
        else:
            o3d.visualization.draw_geometries([trough, receiver])

    return scene, trough, receiver, out_dir, geometry_id_to_panel

cfg = {

    # -----------------------
    # GEOMETRY (trough model)
    # -----------------------
    # EuroTrough ET-150 (Luepfert et al., ISES 2003)
    "W": 5.76,         # aperture width
    "L": 12.0,         # module length
    "f": 1.71,         # focal length

    "Nx": 4,           # panels across the aperture
    "Ny": 7,           # panels along the length

    "sub_x": 30,         # inner vertices in each panel for error deformations
    "sub_y": 30,

    # -----------------------
    # STRUCTURE (supports)
    # -----------------------
    "Nx_support": 4,   # radial supports across width
    "Ny_support": 6,   # supports along length
    "receiver_radius": 0.04,

    # -----------------------
    # SCENE POSITIONING
    # -----------------------
    "origin": [0.0, 0.0, 0.0],  # optional global shift

    # -----------------------
    # LIDAR FLIGHT PATH
    # -----------------------
    "x_min": -4.0,
    "x_max": 4.0,
    "y_min": -1.5,
    "y_max": 1.5,

    "nx_path": 9,
    "ny_path": 40,

    "height": 3.0,   # drone altitude (IMPORTANT: absolute now)

    # -----------------------
    # SENSOR MODEL
    # -----------------------
    "azimuth_res": 120,
    "elevation_res": 48,
    "noise_sigma": 0.003,
    "voxel_size": 0.01
}

# Another simulation
def run_uav_lidar_simulation(
    scene,
    cfg,
    n_steps=150,
    voxel_size=None,
    noise_sigma=0.003,
    dropout=0.15,
    plot=False,
    seed=None,
):
    rng = np.random.default_rng(seed)

    # ----------------------------
    # LiDAR stochastic cone
    # ----------------------------
    n_rays = 800

    # Nadir-centred fan, following the configuration survey-grade UAV
    # scanners use: a wide across-track field of view swept about straight
    # down, together with a small forward and backward oblique.  RIEGL's
    # VUX-120 and VUX-160 quote 100 degrees across track and alternate the
    # scan plane between nadir and +/-10 degrees fore and aft.
    #
    # An earlier version fanned ALONG the track and gave every ray a
    # positive x component, so each lane illuminated only the ground ahead
    # of itself and the aperture was sampled asymmetrically by better than
    # a factor of ten.  Nothing physical motivated that, and it made
    # detection depend on where across the aperture a defect happened to
    # fall rather than on its size alone.
    fov_across = np.deg2rad(108.0)
    fov_along = np.deg2rad(20.0)

    a_x = (rng.random(n_rays) - 0.5) * fov_across
    a_y = (rng.random(n_rays) - 0.5) * fov_along

    dirs = np.stack([
        np.tan(a_x),
        np.tan(a_y),
        -np.ones(n_rays),
    ], axis=1).astype(np.float32)

    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # small angular jitter
    dirs += rng.normal(0, 0.003, dirs.shape)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    rng.shuffle(dirs)

    # ----------------------------
    # UAV lanes
    # ----------------------------
    n_lanes = 4
    lane_offsets = np.linspace(-cfg["W"] / 2, cfg["W"] / 2, n_lanes)

    all_points_raw = []
    all_points_noisy = []
    all_geometry_ids = []

    for x_lane in lane_offsets:
        ys = np.linspace(0, cfg["L"], n_steps)

        for y in ys:
            sensor_origin = np.array(
                [x_lane, y, cfg["height"]],
                dtype=np.float32
            )

            origins = np.repeat(sensor_origin[None, :], len(dirs), axis=0)
            rays = np.hstack((origins, dirs)).astype(np.float32)

            rays_tensor = o3d.core.Tensor(
                rays,
                dtype=o3d.core.Dtype.Float32
            )

            ans = scene.cast_rays(rays_tensor)
            geometry_ids = ans["geometry_ids"].numpy()

            t = ans["t_hit"].numpy()
            valid = np.isfinite(t)
            hit_geom_ids = geometry_ids[valid]

            if np.sum(valid) == 0:
                continue

            ray_dirs = dirs[valid]
            ray_dirs = ray_dirs / np.linalg.norm(
                ray_dirs, axis=1, keepdims=True
            )

            hit_raw = origins[valid] + ray_dirs * t[valid, None]

            # ----------------------------
            # Dropout
            # ----------------------------
            verticality = -ray_dirs[:, 2]
            keep_prob = np.clip(verticality, 0.1, 1.0)

            mask_angle = rng.random(len(hit_raw)) < keep_prob
            mask_sensor = rng.random(len(hit_raw)) > dropout

            mask = mask_angle & mask_sensor

            hit_raw = hit_raw[mask]
            ray_dirs = ray_dirs[mask]
            hit_geom_ids = hit_geom_ids[mask]
            all_geometry_ids.append(hit_geom_ids) #Store them

            if len(hit_raw) == 0:
                continue


            # ----------------------------
            # Range noise
            # ----------------------------
            range_noise = rng.normal(
                0.0,
                noise_sigma,
                size=(len(hit_raw), 1)
            )

            hit_noisy = hit_raw + range_noise * ray_dirs

            all_points_raw.append(hit_raw)
            all_points_noisy.append(hit_noisy)

    if len(all_points_raw) == 0:
        raise ValueError("No points generated")

    points_raw = np.vstack(all_points_raw)
    points_noisy = np.vstack(all_points_noisy)
    geometry_ids_raw = np.concatenate(all_geometry_ids)

    pcd_raw = o3d.geometry.PointCloud()
    pcd_raw.points = o3d.utility.Vector3dVector(points_raw)

    pcd_noisy = o3d.geometry.PointCloud()
    pcd_noisy.points = o3d.utility.Vector3dVector(points_noisy)

    if voxel_size is not None:
        pcd_down = pcd_noisy.voxel_down_sample(voxel_size)
    else:
        pcd_down = pcd_noisy

    if plot:
        o3d.visualization.draw_geometries(
            [pcd_down],
            window_name="Simulated LiDAR Point Cloud"
        )

    return {
        "raw": pcd_raw,
        "noisy": pcd_noisy,
        "downsampled": pcd_down,
        "raw_points": points_raw,
        "noisy_points": points_noisy,
        "geometry_ids": geometry_ids_raw
    }

# Not used for now
error_cfg = {
    "tilt": {
        "enabled": True,
        "probability": 0.7,
        "std_deg": [0.1, 0.1, 0.02]
    },
    "translation": {
        "enabled": True,
        "probability": 0.4,
        "std": [0.001, 0.001, 0.0005]
    },
    "gravity_sag": {
        "enabled": True,
        "probability": 1.0,
        "amplitude_mean": -0.0003,
        "amplitude_std": 0.0001
    },
    "bow": {
        "enabled": True,
        "probability": 0.3,
        "amplitude_mean": 0.0,
        "amplitude_std": 0.0002
    },
    "dent": {
        "enabled": True,
        "probability": 0.1,
        "amplitude_mean": -0.0005,
        "amplitude_std": 0.0002,
        "radius_range": [0.05, 0.20]
    }
}


# For debugging


def plot_panel(panel, apply_errors=True, show_edges=True):
    if apply_errors:
        mesh = apply_panel_errors(panel, copy=True)
    else:
        mesh = o3d.geometry.TriangleMesh(panel["mesh"])

    mesh.compute_vertex_normals()

    geometries = [mesh]

    if show_edges:
        edges = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        geometries.append(edges)

    o3d.visualization.draw_geometries(geometries)


# For plotting errors


# Running

#Debugging only
#for p in panels:
#    if p["errors"]["gravity_sag"]["enabled"]:
#        plot_panel(p, apply_errors=True)
#        break




### PROJECTION AND ALGORITHM IMPLEMENTATION

def project_to_cross_section(points):
    return points[:, [0, 2]]


def voxel_downsample_numpy(points, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd_ds = pcd.voxel_down_sample(voxel_size)

    return np.asarray(pcd_ds.points)


## Model application (oriented case)


def oriented_parabolicity(points, plot=False, convexHull=True,
                          center=True, method="highs", verbose=False):
    """Minimum-width parabolic corridor for a fixed (vertical) axis.

    Solves   min t   s.t.   a x^2 + b x + c <= y_k <= a x^2 + b x + c + t
    with a, b, c free and t >= 0, which is the linear program of the text.

    Returns the SciPy result, with the following extra fields attached:
        width            optimal corridor width t
        coeffs           (a, b, c) in the ORIGINAL abscissa
        rho_lo, rho_up   certified bounds on the parabolicity of the full
                         cloud; rho_lo <= rho* <= rho_up.  The programme is
                         solved exactly, so these coincide; a gap between
                         them means the convex-hull reduction dropped an
                         active constraint, which is checked for below.
        support_points   dict with the active lower/upper constraints
    """
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull
    import time

    xz = np.asarray(points, dtype=float)
    if xz.shape[1] == 3:
        xz = project_to_cross_section(xz)

    # --- lower boundary candidates ------------------------------------

    if convexHull and len(xz) > 3:
        hull = ConvexHull(xz)
        lower = xz[hull.vertices]
    else:
        lower = xz

    # --- upper boundary candidates ------------------------------------

    upper = xz

    # --- centring and scaling of the abscissa -------------------------

    if center:
        x0 = float(np.mean(xz[:, 0]))
        sx = float(np.max(np.abs(xz[:, 0] - x0)))
        if sx == 0.0:
            sx = 1.0
    else:
        x0, sx = 0.0, 1.0

    xu = (upper[:, 0] - x0) / sx
    yu = upper[:, 1]
    n = len(xu)

    for attempt in (0, 1):
        xl = (lower[:, 0] - x0) / sx
        yl = lower[:, 1]
        m = len(xl)

        A_ub = np.zeros((m + n, 4))
        b_ub = np.zeros(m + n)

        # lower boundary constraints: a x^2 + b x + c <= y
        A_ub[:m, 0] = xl ** 2
        A_ub[:m, 1] = xl
        A_ub[:m, 2] = 1.0
        A_ub[:m, 3] = 0.0
        b_ub[:m] = yl

        # upper boundary constraints: y <= a x^2 + b x + c + t
        A_ub[m:, 0] = -xu ** 2
        A_ub[m:, 1] = -xu
        A_ub[m:, 2] = -1.0
        A_ub[m:, 3] = -1.0
        b_ub[m:] = -yu

        obj = np.array([0.0, 0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (None, None), (0, None)]

        start = time.perf_counter()
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
        res.solve_time = time.perf_counter() - start

        if not res.success or not convexHull or attempt == 1:
            break
        # the hull reduction assumed a > 0; if the fit came out concave the
        # dropped lower constraints may have been active, so solve again
        # against every point.  This has never triggered on trough data,
        # where a ~ 1/(4f) > 0, but the programme does not forbid it.
        if res.x[0] <= 0.0:
            if verbose:
                print("fitted a = %.3e <= 0; re-solving without the hull "
                      "reduction" % res.x[0])
            lower = xz
        else:
            break

    if verbose:
        print("constraints: %d lower + %d upper (from %d points)"
              % (m, n, len(xz)))
        print("A abs max: %.3e" % np.max(np.abs(A_ub)))
        print("solve time: %.3f s" % res.solve_time)

    if not res.success:
        print("LP failed:", res.message)
        return res

    a_s, b_s, c_s, t = res.x

    # map the fitted coefficients back to the original abscissa
    a_f = a_s / sx ** 2
    b_f = -2.0 * a_s * x0 / sx ** 2 + b_s / sx
    c_f = a_s * x0 ** 2 / sx ** 2 - b_s * x0 / sx + c_s

    res.width = float(t)
    res.coeffs = (a_f, b_f, c_f)
    res.abscissa_transform = (x0, sx)

    # --- two-sided certificate ----------------------------------------
    # rho_lo: optimum of the reduced problem            -> lower bound
    # rho_up: the same parabola evaluated on ALL points -> upper bound
    r_full = xz[:, 1] - a_f * xz[:, 0] ** 2 - b_f * xz[:, 0]
    res.rho_lo = float(t)
    res.rho_up = float(r_full.max() - r_full.min())

    # --- support points from the active constraints --------------------
    res.lower_candidates = lower
    res.upper_candidates = upper
    res.n_lower = m
    res.support_points = support_points_from_result(res, lower, upper, m)

    if plot:
        x_plot = np.linspace(xz[:, 0].min(), xz[:, 0].max(), 300)
        y_base = a_f * x_plot ** 2 + b_f * x_plot + c_f
        y_top = y_base + t

        plt.figure(figsize=(8, 5))
        plt.scatter(xz[:, 0], xz[:, 1], s=1, color="black", label="Points")
        plt.plot(x_plot, y_base, "b--", label="Lower parabola")
        plt.plot(x_plot, y_top, "r--", label="Upper parabola")
        plt.fill_between(x_plot, y_base, y_top, alpha=0.2)

        sup = res.support_points["all"]
        if len(sup):
            plt.scatter(sup[:, 0], sup[:, 1], s=40, facecolors="none",
                        edgecolors="green", label="Support points")

        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.title("Oriented parabolicity band (width = %.5f)" % t)
        plt.show()

    return res


def support_points_from_result(res, lower, upper, m, tol=1e-9):
    """Active constraints of the LP, split at the block boundary m.

    """
    marginals = None
    ineqlin = getattr(res, "ineqlin", None)
    if ineqlin is not None:
        marginals = getattr(ineqlin, "marginals", None)

    if marginals is not None:
        active = np.where(np.abs(marginals) > tol)[0]
    else:
        active = np.where(np.abs(res.slack) < 1e-7)[0]

    lo = active[active < m]
    up = active[active >= m] - m

    pts_lo = lower[lo]
    pts_up = upper[up]

    if len(pts_lo) + len(pts_up):
        allpts = np.vstack([p for p in (pts_lo, pts_up) if len(p)])
    else:
        allpts = np.empty((0, 2))

    return {"lower": pts_lo, "upper": pts_up, "all": allpts}


def remez_parabolicity(points, tol=1e-10, max_iter=200, tau=None,
                       verbose=False, strict=False):
    # Exact oriented parabolicity by Remez exchange.

    import time
    import warnings

    xz = np.asarray(points, dtype=float)
    if xz.shape[1] == 3:
        xz = project_to_cross_section(xz)

    order = np.argsort(xz[:, 0], kind="stable")
    xz = xz[order]

    x = xz[:, 0]
    y = xz[:, 1]
    n = len(x)

    if n < 4:
        raise ValueError("need at least 4 points")

    # x is sorted, so ties and the closest approach are one pass
    gaps = np.diff(x)
    n_tied = int(np.count_nonzero(gaps == 0.0))
    positive = gaps[gaps > 0.0]
    min_gap = float(positive.min()) if positive.size else 0.0

    # include warning
    if n_tied:
        message = ("remez_parabolicity: %d repeated abscissa(e) among %d "
                   "points. The alternation characterisation assumes "
                   "distinct abscissae; perturb one of each tied pair."
                   % (n_tied, n))
        if strict:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    #s centre and scale the abscissa for conditionings
    x0 = float(x.mean())
    sx = float(np.max(np.abs(x - x0)))
    if sx == 0.0:
        sx = 1.0
    u = (x - x0) / sx

    ref = np.unique(np.linspace(0, n - 1, 4).astype(int))
    if len(ref) < 4:
        ref = np.arange(4)

    signs = np.array([(-1.0) ** j for j in range(4)])

    start = time.perf_counter()
    history = []
    converged = False
    decision = None
    status = "max_iter"
    cond_max = 0.0
    a = b = c = h = 0.0

    for it in range(max_iter):
        ur = u[ref]
        M = np.column_stack([ur ** 2, ur, np.ones(4), signs])
        cond_max = max(cond_max, float(np.linalg.cond(M)))

        try:
            a, b, c, h = np.linalg.solve(M, y[ref])
        except np.linalg.LinAlgError:
            status = "singular_reference"
            warnings.warn("remez_parabolicity: singular 4x4 system at "
                          "iteration %d (reference abscissae %s)"
                          % (it, np.array2string(x[ref], precision=6)),
                          RuntimeWarning, stacklevel=2)
            break

        r = y - (a * u ** 2 + b * u + c)

        kstar = int(np.argmax(np.abs(r)))
        rmax = float(abs(r[kstar]))

        rho_lo = 2.0 * abs(h)                      # de la Vallee Poussin
        rho_up = float(r.max() - r.min())          # feasible corridor width
        history.append((it, rho_lo, rho_up))

        if verbose:
            print("  it %2d  bracket [%.8f, %.8f]  gap %.2e"
                  % (it, rho_lo, rho_up, rho_up - rho_lo))

        # early exit for the classification problem
        if tau is not None:
            if rho_lo > tau:
                decision = "reject"
                status = "decided"
                break
            if rho_up <= tau:
                decision = "accept"
                status = "decided"
                break

        if rmax <= abs(h) * (1.0 + tol) + tol:
            converged = True
            status = "converged"
            break

        # --- exchange: insert kstar preserving the alternation of signs ----
        refsign = signs * np.sign(h)
        rs = np.sign(r[kstar])

        if u[kstar] < u[ref[0]]:
            if rs == refsign[0]:
                ref[0] = kstar
            else:
                ref = np.array([kstar, ref[0], ref[1], ref[2]])
        elif u[kstar] > u[ref[3]]:
            if rs == refsign[3]:
                ref[3] = kstar
            else:
                ref = np.array([ref[1], ref[2], ref[3], kstar])
        else:
            j = int(np.searchsorted(u[ref], u[kstar]) - 1)
            j = max(0, min(j, 2))
            if rs == refsign[j]:
                ref[j] = kstar
            else:
                ref[j + 1] = kstar

        ref = np.unique(ref)
        if len(ref) < 4:                      # degenerate exchange, give up
            status = "degenerate_exchange"
            warnings.warn("remez_parabolicity: exchange collapsed the "
                          "reference below four points at iteration %d" % it,
                          RuntimeWarning, stacklevel=2)
            break

    elapsed = time.perf_counter() - start

    # map the coefficients back to the original abscissa
    a_f = a / sx ** 2
    b_f = -2.0 * a * x0 / sx ** 2 + b / sx
    c_f = a * x0 ** 2 / sx ** 2 - b * x0 / sx + c

    r = y - (a * u ** 2 + b * u + c)

    return {
        "width": 2.0 * abs(h),
        "coeffs": (a_f, b_f, c_f - abs(h)),     # lower boundary of the corridor
        "rho_lo": 2.0 * abs(h),
        "rho_up": float(r.max() - r.min()),
        "support_points": xz[ref],
        "support_signs": np.sign(r[ref]),
        "iterations": len(history),
        "converged": converged,
        "decision": decision,
        "status": status,
        "solve_time": elapsed,
        "history": history,
        "n_tied_abscissae": n_tied,
        "min_abscissa_gap": min_gap,
        "max_reference_condition": cond_max,
    }


# res = oriented_parabolicity(panel_points_ds)


### NON-ORIENTED EVALUATION


def nominal_vertex():
    # Cross-section coordinates of the parabola vertex for the generated mesh.

    return np.array([0.0, 0.0])


def rotate_points_about_vertex(points, delta, pivot=None):
    #Rotate a cloud by `delta` radians in the cross-section plane.

    P = np.asarray(points, dtype=float)
    pivot = nominal_vertex() if pivot is None else np.asarray(pivot, dtype=float)

    cd = np.cos(delta)
    sd = np.sin(delta)

    out = P.copy()

    if P.shape[1] == 3:
        ix, iz = 0, 2
    elif P.shape[1] == 2:
        ix, iz = 0, 1
    else:
        raise ValueError("points must have 2 or 3 columns")

    x = P[:, ix] - pivot[0]
    z = P[:, iz] - pivot[1]

    out[:, ix] = pivot[0] + x * cd - z * sd
    out[:, iz] = pivot[1] + x * sd + z * cd

    return out


def perturbation_parabolicity(points, a_nom, b_nom=0.0, eps=0.02,
                              C_a=0.0, C_b=0.0, method="highs"):
    from scipy.optimize import linprog
    import time

    xz = np.asarray(points, dtype=float)
    if xz.shape[1] == 3:
        xz = project_to_cross_section(xz)

    x = xz[:, 0]
    z = xz[:, 1]
    n = len(x)

    r0 = z - a_nom * x ** 2 - b_nom * x
    u = x + 2.0 * a_nom * x * z + b_nom * z

    # variables: (delta, da, db, c, t)
    A = np.zeros((2 * n, 5))
    b = np.zeros(2 * n)

    # lower:  c - r_k <= 0
    A[:n, 0] = -u
    A[:n, 1] = x ** 2
    A[:n, 2] = x
    A[:n, 3] = 1.0
    b[:n] = r0

    # upper:  r_k - c - t <= 0
    A[n:, 0] = u
    A[n:, 1] = -x ** 2
    A[n:, 2] = -x
    A[n:, 3] = -1.0
    A[n:, 4] = -1.0
    b[n:] = -r0

    obj = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    bounds = [(-eps, eps),
              (-C_a * eps, C_a * eps),
              (-C_b * eps, C_b * eps),
              (None, None),
              (0.0, None)]

    start = time.perf_counter()
    res = linprog(obj, A_ub=A, b_ub=b, bounds=bounds, method=method)
    elapsed = time.perf_counter() - start

    if not res.success:
        return {"success": False, "message": res.message, "solve_time": elapsed}

    delta, da, db, c, t = res.x

    tol = 1e-9
    box_active = {
        "delta": abs(abs(delta) - eps) < tol * max(1.0, eps),
        "da": C_a > 0 and abs(abs(da) - C_a * eps) < tol * max(1.0, C_a * eps),
        "db": C_b > 0 and abs(abs(db) - C_b * eps) < tol * max(1.0, C_b * eps),
    }


    return {
        "success": True,
        "width": float(t),
        "delta": float(delta),
        "da": float(da),
        "db": float(db),
        "c": float(c),
        "box_active": box_active,
        "any_box_active": any(box_active.values()),
        "solve_time": elapsed,
        "n": n,
    }


def rotation_search_parabolicity(points, phi_max=0.10, n_grid=None, h=None,
                                 pivot=None, solver="remez", verbose=False):
    # search over the axis direction, using Remez at each angle.

    import time

    P = np.asarray(points, dtype=float)

    if h is not None:
        n_grid = int(np.ceil(2.0 * phi_max / h)) + 1
    if n_grid is None:
        n_grid = 201

    phis = np.linspace(-phi_max, phi_max, n_grid)

    widths = np.empty(n_grid)
    start = time.perf_counter()

    for i, phi in enumerate(phis):
        rot = rotate_points_about_vertex(P, phi, pivot=pivot)
        if solver == "remez":
            widths[i] = remez_parabolicity(rot)["width"]
        else:
            r = oriented_parabolicity(rot, plot=False)
            widths[i] = r.width if r.success else np.inf

    elapsed = time.perf_counter() - start

    j = int(np.argmin(widths))

    if verbose:
        print("rotation search: %d angles over [%.4f, %.4f] rad in %.3f s"
              % (n_grid, -phi_max, phi_max, elapsed))
        print("  best width %.6f at phi = %.6f rad (%.4f deg)"
              % (widths[j], phis[j], np.rad2deg(phis[j])))

    return {
        "width": float(widths[j]),
        "phi": float(phis[j]),
        "phis": phis,
        "widths": widths,
        "n_grid": n_grid,
        "spacing": float(phis[1] - phis[0]) if n_grid > 1 else 0.0,
        "solve_time": elapsed,
        "at_boundary": j in (0, n_grid - 1),
    }
