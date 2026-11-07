"""Reference-field vortex-ring geometry for diagnostic evaluation plots."""
import numpy as np


def detect_ring_geometry(coords, omega, config=None):
    config = config or {}
    from gino.data.spatial_sampling import analyze_vorticity_cores
    values = np.zeros((len(coords), 15), dtype=np.float32); values[:, 12:15] = omega
    core, diagnostics = analyze_vorticity_cores(coords, values, config)
    if not core.any():
        return {"valid": False, "reason": "no_significant_vorticity_core", **diagnostics}
    magnitude = np.linalg.norm(omega, axis=1)
    weight = np.where(core, magnitude ** float(config.get("axis_weight_power", 1.0)), 0.0)
    center = np.sum(coords * weight[:, None], axis=0) / max(float(weight.sum()), 1e-30)
    axis = int(config.get("propagation_axis", 2))
    trans = [i for i in range(3) if i != axis]
    axes = [np.unique(coords[:, i]) for i in range(3)]
    spacing = [float(np.median(np.diff(a))) if len(a) > 1 else 1.0 for a in axes]
    shape = tuple(map(len, axes))
    profile = np.zeros(shape[axis], dtype=float)
    ijk = tuple(np.searchsorted(axes[i], coords[:, i]) for i in range(3))
    mag2 = magnitude ** 2
    for index in range(len(coords)):
        profile[ijk[axis][index]] += mag2[index] * np.prod([spacing[i] for i in trans])
    peaks = np.array([], dtype=int)
    try:
        from scipy.signal import find_peaks
        distance = max(1, int(np.ceil(float(config.get("minimum_peak_separation_physical", 0.0)) / spacing[axis])))
        prominence = float(config.get("peak_prominence_fraction", .1)) * max(float(profile.max()), 0.0)
        peaks, _ = find_peaks(profile, distance=distance, prominence=prominence)
        if profile.size and (not len(peaks)) and profile.max() > 0:
            peaks = np.array([int(np.argmax(profile))])
    except ImportError:
        if profile.size and profile.max() > 0: peaks = np.array([int(np.argmax(profile))])
    max_peaks = int(config.get("maximum_ring_planes", 8))
    peaks = peaks[np.argsort(profile[peaks])[::-1][:max_peaks]] if len(peaks) else peaks
    peaks.sort()
    return {"valid": True, "axis_center_instantaneous": center.tolist(),
            "transverse_axes": trans, "propagation_axis": axis,
            "ring_axial_locations": axes[axis][peaks].astype(float).tolist(),
            "components": diagnostics.get("components", 0), "core_diagnostics": diagnostics}


def smooth_axis(previous, current, alpha):
    current = np.asarray(current, dtype=float)
    return current if previous is None else float(alpha) * current + (1.0 - float(alpha)) * np.asarray(previous)


def track_axial_peaks(previous, current, alpha=1.0):
    """Nearest-neighbour peak association; unmatched detections start new tracks."""
    current = np.asarray(current, dtype=float)
    if previous is None or not len(current):
        return current.tolist()
    previous = np.asarray(previous, dtype=float)
    remaining = set(range(len(current)))
    tracked = []
    for old in previous:
        if not remaining:
            break
        chosen = min(remaining, key=lambda i: abs(current[i] - old))
        remaining.remove(chosen)
        tracked.append(float(alpha) * current[chosen] + (1.0 - float(alpha)) * old)
    tracked.extend(float(current[i]) for i in sorted(remaining))
    return sorted(tracked)
