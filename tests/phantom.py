"""Physical measurements on an image of the example frame. The example shows a CIRS 054GS phantom (speed of sound
1540 m/s, 0.5 dB/cm/MHz). From its user guide, the targets inside the imaged 44 x 52 mm:

    near-field group    6 wires, 1 to 6 mm deep in steps of 1 mm, 6 mm apart sideways
    vertical group      wires every 20 mm in depth: the two at nominal 20 and 40 mm are in the image
    grey-scale target   a hyperechoic cylinder of 8 mm diameter at 40 mm depth

A wire is a peak along x. Before it is located, the median of every row over 8 mm is taken off: that removes what
all columns of a row share, above all the echo of the probe surface, which is in the image as soon as the first
samples of the record are kept and is 20 times brighter than the shallowest wire next to it.

measure() returns numbers that have a nominal value (scales, offsets), or a direction that is better (widths,
contrast). The values of the reference frame are in tests/golden/metrics.json.
"""
import numpy as np

NEAR_WIRES_MM = [(-16, 2.0), (-10, 3.2), (-4, 4.2), (2, 5.1), (8, 5.9), (14, 6.9)]     # rough (x, z), read off the image
DEPTH_WIRES_MM = [(14, 20.1), (14, 39.3)]
NOMINAL = dict(lateral_spacing_mm=6.0, depth_wire_spacing_mm=20.0, near_depth_step_mm=1.0, speckle_snr=1.91, cylinder_diameter_mm=8.0)


def _width(profile, step):
    """Full width at half maximum of the peak of a profile."""
    p = np.asarray(profile, float)
    k = int(p.argmax())
    half = p[k] / 2
    left, right = k, k
    while left > 0 and p[left] > half:
        left -= 1
    while right < len(p) - 1 and p[right] > half:
        right += 1
    lo = left + (half - p[left]) / (p[left + 1] - p[left]) if p[left] < half else left
    hi = right - (half - p[right]) / (p[right - 1] - p[right]) if p[right] < half else right
    return (hi - lo) * step


def _wire(env, x, z, guess, half=(1.6, 0.9), band=4.0):
    """Position (centroid of the upper half of the peak) and widths of the wire near `guess`, all in mm."""
    ix = np.flatnonzero(np.abs(x - guess[0]) <= half[0])
    iz = np.flatnonzero(np.abs(z - guess[1]) <= half[1])
    floor = np.median(env[np.ix_(iz, np.flatnonzero(np.abs(x - guess[0]) <= band))], axis=1)[:, None]
    patch = np.clip(env[np.ix_(iz, ix)] - floor, 0, None)
    kz, kx = np.unravel_index(patch.argmax(), patch.shape)
    w = patch ** 2 * (patch > 0.5 * patch.max())
    cx = (w.sum(axis=0) * x[ix]).sum() / w.sum()
    cz = (w.sum(axis=1) * z[iz]).sum() / w.sum()
    return cx, cz, _width(patch[kz], x[1] - x[0]), _width(patch[:, kx], z[1] - z[0])


def measure(env, grid_x, grid_z):
    """env: envelope [depth, lateral]; grid_x, grid_z: pixel coordinates [m]. Returns a dict of the phantom metrics."""
    x, z = 1e3 * np.asarray(grid_x, float), 1e3 * np.asarray(grid_z, float)
    near = np.array([_wire(env, x, z, g) for g in NEAR_WIRES_MM])
    deep = np.array([_wire(env, x, z, g) for g in DEPTH_WIRES_MM])
    lateral_spacing = np.polyfit(np.arange(6), near[:, 0], 1)[0]
    depth_step, depth_offset = np.polyfit(np.arange(1, 7), near[:, 1], 1)

    X, Z = np.meshgrid(x, z)
    centre = max(((env[(X - cx) ** 2 + (Z - cz) ** 2 < 9].mean(), cx, cz) for cx in np.arange(0, 6, 0.25) for cz in np.arange(36, 42, 0.25)))
    r = np.hypot(X - centre[1], Z - centre[2])
    inside = env[r < 3.0]
    ring = env[(r > 6.0) & (r < 8.5) & (np.hypot(X - deep[1, 0], Z - deep[1, 1]) > 2)]
    speckle = env[(Z > 24) & (Z < 32) & (np.abs(X) < 10)]
    return dict(
        lateral_spacing_mm=float(lateral_spacing),                         # nominal 6
        lateral_scale_percent=float(100 * (lateral_spacing / 6 - 1)),
        depth_wire_spacing_mm=float(deep[1, 1] - deep[0, 1]),              # nominal 20
        depth_scale_percent=float(100 * ((deep[1, 1] - deep[0, 1]) / 20 - 1)),
        near_depth_step_mm=float(depth_step),                              # nominal 1
        depth_offset_mm=float(depth_offset),                               # nominal 0
        wire_width_lateral_mm=[float(near[:, 2].mean()), float(deep[0, 2]), float(deep[1, 2])],    # near field, 20 mm, 40 mm
        wire_width_axial_mm=[float(near[:, 3].mean()), float(deep[0, 3]), float(deep[1, 3])],
        cylinder_contrast_db=float(20 * np.log10(inside.mean() / ring.mean())),
        cylinder_cnr=float((inside.mean() - ring.mean()) / np.sqrt(inside.var() + ring.var())),
        speckle_snr=float(speckle.mean() / speckle.std()),                 # 1.91 for fully developed speckle
        brightness_40_over_10_mm_db=float(20 * np.log10(env[(Z > 44) & (Z < 50)].mean() / env[(Z > 9) & (Z < 15)].mean())),
    )
