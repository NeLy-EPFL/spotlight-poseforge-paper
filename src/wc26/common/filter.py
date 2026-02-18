import numpy as _np
from scipy.ndimage import median_filter as _median_filter


def boolean_majority_filter(x: _np.ndarray, k: int, *, pad_mode="edge") -> _np.ndarray:
    """Denoise a 1D boolean array with a centered majority (median) filter.

    Args:
        k: Kernel/window size in samples (steps). Must be odd.
        pad_mode: Padding mode for _np.pad (e.g., "edge", "reflect", "constant").
    """
    x = _np.asarray(x, dtype=bool)
    if k <= 1:
        return x.copy()
    if k % 2 == 0:
        raise ValueError(f"Kernel size k must be odd, got {k}")

    r = k // 2
    xp = _np.pad(x.astype(_np.int16), (r, r), mode=pad_mode)
    w = _np.lib.stride_tricks.sliding_window_view(xp, k)
    return w.mean(axis=-1) >= 0.5


def boolean_true_runs(mask: _np.ndarray):
    """Find the start and end indices of contiguous True periods in a 1D boolean array.

    Args:
        mask: mask to analyze.

    Returns:
        A list of (start, end) index pairs for each contiguous True run.
    """
    m = _np.asarray(mask, dtype=bool)

    padded = _np.r_[False, m, False].astype(_np.int8)
    d = _np.diff(padded)

    starts = _np.flatnonzero(d == 1)
    ends = _np.flatnonzero(d == -1)
    return list(zip(starts, ends))


def median_filter_over_time(x: _np.ndarray, k: int) -> _np.ndarray:
    """Apply median filter along axis 0 only.

    Args:
        x: Input array of shape (n, ...).
        k: Window size along axis 0. Must be odd.

    Returns:
        Median-filtered array with same shape as x.
    """
    x = _np.asarray(x)
    if k <= 1:
        return x.copy()
    if k % 2 == 0:
        raise ValueError(f"Kernel size k must be odd, got {k}")

    filter_size = (k,) + (1,) * (x.ndim - 1)
    return _median_filter(x, size=filter_size)