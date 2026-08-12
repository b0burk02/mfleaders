"""Diagnose the 3D LUNG (folded-surface) control volume from
examples/make_3d_controls.py BEFORE trusting any conclusion drawn from its 2D
column-density projection.

This script does NOT change anything about how the fields are built -- it
imports make_3d_controls.py and calls its functions verbatim, with the exact
same seeds/parameters, to reproduce the identical smooth-fill and lung
volumes that script already reported on. Everything below is diagnosis and
visualization of those volumes: slice renders, an isosurface render, a
thin-slab max-projection (to isolate whether the weak 2D projection contrast
is a real projection-washout effect or a symptom of a weak 3D control), and
two morphology checks (void/wall structure, surface-area-vs-scale).

Run from the repo root or from examples/ -- it adds this file's directory to
sys.path so `import make_3d_controls` works regardless of cwd.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_3d_controls as gen  # noqa: E402  (see sys.path.insert above)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N = gen.N


# --- reproduce the exact volumes make_3d_controls.py builds -----------------

def build_volumes():
    H_target = 3 - gen.D_TARGET

    rng_smooth = np.random.default_rng(gen.SEED_SMOOTH)
    smooth = gen.fbm3d_spectral(gen.N, H_target, rng_smooth)
    smooth = smooth - smooth.min()

    rng_lung = np.random.default_rng(gen.SEED_LUNG)
    base = gen.fbm3d_spectral(gen.N, gen.BASE_H_LUNG, rng_lung)
    level = np.median(base)
    thickness, D_lung, converged = gen.tune_shell_thickness(
        base, level, gen.D_TARGET)
    lung = gen.make_shell(base, level, thickness * base.std())

    return dict(smooth=smooth, base=base, level=level, thickness=thickness,
                D_lung=D_lung, converged=converged, lung=lung)


# --- construction diagnostics: how thick is the shell, really? --------------

def radial_profile_from_center(volume_3d):
    """Radially average a 3D array (already centered, e.g. an fftshifted
    autocorrelation) by distance-from-center in voxels. Reuses
    gen._radial_k_index -- it builds exactly this "integer distance from the
    array center" index (it's used for frequency magnitude in the generator,
    but fftfreq(n)*n after fftshift is just centered integer coordinates,
    equally valid as a spatial-lag radius index here).
    """
    n = volume_3d.shape[0]
    r_idx = gen._radial_k_index(n)
    rmax = n // 2
    valid = (r_idx >= 0) & (r_idx < rmax)
    sums = np.bincount(r_idx[valid], weights=volume_3d[valid], minlength=rmax)
    counts = np.bincount(r_idx[valid], minlength=rmax)
    return np.arange(rmax), sums / counts


def correlation_length(field, drop_to=1.0 / np.e):
    """Isotropic correlation length in voxels: radius at which the
    normalized autocorrelation first drops below `drop_to`.
    """
    f = field - field.mean()
    F = np.fft.fftn(f)
    ac = np.fft.ifftn(np.abs(F) ** 2).real
    ac = np.fft.fftshift(ac)
    ac /= ac.max()
    r, prof = radial_profile_from_center(ac)
    below = np.where(prof < drop_to)[0]
    return float(r[below[0]]) if len(below) else float(r[-1])


def shell_spatial_thickness(base, level, shell, sigma_value):
    """Convert the shell's VALUE-space Gaussian sigma into a real-space
    (voxel) thickness using the local field gradient near the level set:
    spatial_sigma ~= sigma_value / |grad(base)|, gradient-weighted by the
    shell itself so it's evaluated where the shell actually has mass.
    """
    gx, gy, gz = np.gradient(base)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    weighted_grad = np.sum(grad_mag * shell) / np.sum(shell)
    sigma_spatial = sigma_value / weighted_grad
    fwhm_spatial = sigma_spatial * 2 * np.sqrt(2 * np.log(2))
    return weighted_grad, sigma_spatial, fwhm_spatial


# --- morphology check 1: void/wall structure ---------------------------------

def void_wall_structure(field, threshold):
    from scipy.ndimage import distance_transform_edt

    wall = field > threshold
    void = ~wall
    vol_frac = wall.mean()

    dist_to_wall = distance_transform_edt(void)   # for void voxels
    dist_to_void = distance_transform_edt(wall)    # for wall voxels

    void_radius = np.median(dist_to_wall[void]) if void.any() else 0.0
    wall_halfthick = np.median(dist_to_void[wall]) if wall.any() else 0.0
    void_max = dist_to_wall[void].max() if void.any() else 0.0

    return dict(vol_frac=vol_frac, void_radius=void_radius,
                wall_halfthick=wall_halfthick, void_max=void_max)


# --- morphology check 2: surface area vs. measurement scale ------------------

def block_reduce_mean(field, factor):
    n = field.shape[0]
    m = n // factor
    return field[:m * factor, :m * factor, :m * factor].reshape(
        m, factor, m, factor, m, factor).mean(axis=(1, 3, 5))


def surface_area_vs_scale(field, level, scales):
    from skimage.measure import marching_cubes, mesh_surface_area

    results = []
    for scale in scales:
        vol = block_reduce_mean(field, scale) if scale > 1 else field
        try:
            verts, faces, _, _ = marching_cubes(
                vol, level=level, spacing=(scale, scale, scale))
            area = mesh_surface_area(verts, faces)
        except (ValueError, RuntimeError):
            area = np.nan
        results.append(area)
    return np.array(results)


# --- rendering ----------------------------------------------------------------

def render_orthogonal_slices(smooth, lung, path):
    import matplotlib.pyplot as plt

    mid = N // 2
    fields = {"smooth-fill": smooth, "lung": lung}
    slices = {
        "xy (z=mid)": lambda f: f[:, :, mid],
        "xz (y=mid)": lambda f: f[:, mid, :],
        "yz (x=mid)": lambda f: f[mid, :, :],
    }
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for row, (fname, field) in enumerate(fields.items()):
        for col, (sname, sl) in enumerate(slices.items()):
            ax = axes[row, col]
            ax.imshow(sl(field), cmap="viridis")
            ax.set_title(f"{fname}: {sname}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_depth_series(smooth, lung, path, n_slices=6):
    import matplotlib.pyplot as plt

    zs = np.linspace(0, N - 1, n_slices).astype(int)
    fig, axes = plt.subplots(2, n_slices, figsize=(2.2 * n_slices, 4.6))
    for row, (fname, field) in enumerate([("smooth-fill", smooth), ("lung", lung)]):
        for col, z in enumerate(zs):
            ax = axes[row, col]
            ax.imshow(field[:, :, z], cmap="viridis")
            ax.set_title(f"{fname} z={z}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_thin_slab_vs_full(smooth, lung, path, slab=16):
    import matplotlib.pyplot as plt

    mid = N // 2
    lo, hi = mid - slab // 2, mid + slab // 2
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.5))
    for row, (fname, field) in enumerate([("smooth-fill", smooth), ("lung", lung)]):
        full_sum = field.sum(axis=2)
        thin_max = field[:, :, lo:hi].max(axis=2)
        axes[row, 0].imshow(full_sum, cmap="viridis")
        axes[row, 0].set_title(f"{fname}: full column SUM (all {N} voxels)")
        axes[row, 1].imshow(thin_max, cmap="viridis")
        axes[row, 1].set_title(f"{fname}: thin-slab MAX ({slab} voxels)")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_isosurfaces(smooth, lung, smooth_level, lung_level, path, scale=4):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    fig = plt.figure(figsize=(11, 5.5))
    for i, (fname, field, level) in enumerate(
            [("smooth-fill", smooth, smooth_level), ("lung", lung, lung_level)]):
        vol = block_reduce_mean(field, scale)
        verts, faces, normals, _ = marching_cubes(vol, level=level)
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        mesh = Poly3DCollection(verts[faces], alpha=0.6)
        mesh.set_facecolor((0.2, 0.5, 0.7, 0.6))
        mesh.set_edgecolor((0, 0, 0, 0.05))
        ax.add_collection3d(mesh)
        n = vol.shape[0]
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_zlim(0, n)
        ax.set_title(f"{fname} isosurface\n(downsampled x{scale}, level={level:.3g})")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print(f"reproducing {N}^3 volumes from make_3d_controls.py (seeds "
          f"{gen.SEED_SMOOTH}/{gen.SEED_LUNG})...\n")
    v = build_volumes()
    smooth, base, level, thickness, D_lung, converged, lung = (
        v["smooth"], v["base"], v["level"], v["thickness"], v["D_lung"],
        v["converged"], v["lung"])

    # ---------------------------------------------------------------------
    # 1. Restate the construction, with real numbers.
    # ---------------------------------------------------------------------
    print("=" * 78)
    print("LUNG CONSTRUCTION (from make_3d_controls.py, unchanged):")
    print("=" * 78)
    print("""
    base  = fbm3d_spectral(N, BASE_H_LUNG, rng_lung)   # underlying fBm volume
    level = np.median(base)                            # the isocontour value
    thickness, D_lung, converged = tune_shell_thickness(base, level, D_TARGET)
    lung  = make_shell(base, level, thickness * base.std())

    def make_shell(field, level, thickness):
        return np.exp(-0.5 * ((field - level) / thickness) ** 2)
    """)
    sigma_value = thickness * base.std()
    weighted_grad, sigma_spatial, fwhm_spatial = shell_spatial_thickness(
        base, level, lung, sigma_value)
    corr_len_base = correlation_length(base)

    print(f"BASE_H_LUNG = {gen.BASE_H_LUNG}, base.std() = {base.std():.4f}, "
          f"level = median(base) = {level:.4f}")
    print(f"bisection converged: {converged}, D_lung achieved = {D_lung:.4f} "
          f"(target {gen.D_TARGET})")
    print(f"tuned thickness fraction (of base.std()): {thickness:.4f}  ->  "
          f"value-space sigma = {sigma_value:.4f}")
    print(f"gradient-weighted |grad(base)| near the level set: "
          f"{weighted_grad:.5f} per voxel")
    print(f"  => shell spatial sigma  ~= {sigma_spatial:.2f} voxels")
    print(f"  => shell spatial FWHM   ~= {fwhm_spatial:.2f} voxels")
    print(f"base field correlation length (1/e radius): {corr_len_base:.2f} voxels")
    print(f"  => shell FWHM / correlation length = "
          f"{fwhm_spatial / corr_len_base:.3f}")
    print(f"  => shell FWHM / voxel size = {fwhm_spatial:.2f} "
          f"(1 voxel = 1 by definition)")

    # ---------------------------------------------------------------------
    # 2/3. Slice renders.
    # ---------------------------------------------------------------------
    print("\nrendering orthogonal slices, depth series, thin-slab vs full "
          "projection, isosurfaces...")
    render_orthogonal_slices(
        smooth, lung, os.path.join(OUT_DIR, "lung3d_orthogonal_slices.png"))
    render_depth_series(
        smooth, lung, os.path.join(OUT_DIR, "lung3d_depth_series.png"))
    render_thin_slab_vs_full(
        smooth, lung, os.path.join(OUT_DIR, "lung3d_thin_slab_vs_full_projection.png"))
    smooth_level = np.median(smooth)
    render_isosurfaces(
        smooth, lung, smooth_level, 0.5,
        os.path.join(OUT_DIR, "lung3d_isosurfaces.png"))
    print("saved: lung3d_orthogonal_slices.png, lung3d_depth_series.png, "
          "lung3d_thin_slab_vs_full_projection.png, lung3d_isosurfaces.png")

    # ---------------------------------------------------------------------
    # 4. Void structure.
    # ---------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("VOID/WALL STRUCTURE (threshold = 0.5 for lung, median for smooth):")
    print("=" * 78)
    lung_struct = void_wall_structure(lung, 0.5)
    smooth_struct = void_wall_structure(smooth, smooth_level)
    for name, s in [("lung", lung_struct), ("smooth-fill (reference)", smooth_struct)]:
        ratio = s["void_radius"] / s["wall_halfthick"] if s["wall_halfthick"] > 0 else float("inf")
        print(f"{name}: volume_fraction_above_thresh={s['vol_frac']:.4f}  "
              f"median_void_radius={s['void_radius']:.2f}vox  "
              f"median_wall_halfthickness={s['wall_halfthick']:.2f}vox  "
              f"max_void_radius={s['void_max']:.2f}vox  "
              f"void/wall ratio={ratio:.2f}")

    # ---------------------------------------------------------------------
    # 5. Surface-area-vs-scale.
    # ---------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SURFACE AREA vs. MEASUREMENT SCALE (block-averaged then re-isosurfaced):")
    print("=" * 78)
    scales = [1, 2, 4, 8, 16]
    lung_areas = surface_area_vs_scale(lung, 0.5, scales)
    smooth_areas = surface_area_vs_scale(smooth, smooth_level, scales)
    volume_native = float(N ** 3)
    print(f"{'scale':>6}  {'lung area':>14}  {'lung area/vol':>15}  "
          f"{'smooth area':>14}  {'smooth area/vol':>16}")
    for sc, la, sa in zip(scales, lung_areas, smooth_areas):
        print(f"{sc:6d}  {la:14.1f}  {la / volume_native:15.6f}  "
              f"{sa:14.1f}  {sa / volume_native:16.6f}")

    lung_growth = lung_areas[0] / lung_areas[-1] if lung_areas[-1] > 0 else float("nan")
    smooth_growth = smooth_areas[0] / smooth_areas[-1] if smooth_areas[-1] > 0 else float("nan")
    print(f"\nlung area(scale=1)/area(scale={scales[-1]})   = {lung_growth:.3f}")
    print(f"smooth area(scale=1)/area(scale={scales[-1]}) = {smooth_growth:.3f}")

    print("\ndone.")
