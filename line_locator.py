import numpy as np
from scipy.signal import medfilt
from typing import List
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


@dataclass
class Line:
    f_start: float
    f_end: float
    f0: float
    bandwidth: float
    ratio: float


@dataclass
class LineLocatorResult:
    running_median_psd: np.ndarray
    is_line_bin: np.ndarray
    psd_model: np.ndarray
    lines: List[Line]
    ratio: np.ndarray

    # post init properties
    n_lines: int = 0
    max_ratio: float = 0.0
    ratios: List[float] = None
    f0s: List[float] = None
    bws: List[float] = None

    def __post_init__(self):
        self.n_lines = len(self.lines)
        if self.lines:
            self.max_ratio = max(line.ratio for line in self.lines)
            self.ratios = [line.ratio for line in self.lines]
            self.f0s = [line.f0 for line in self.lines]
            self.bws = [line.bandwidth for line in self.lines]
        else:
            self.ratios = []
            self.f0s = []
            self.bws = []



def _create_lines(is_line: np.ndarray, freq: np.ndarray, ratio: np.ndarray,
                  threshold_low: float = 1.25) -> List[Line]:
    line_details = []

    padded = np.concatenate([[False], is_line, [False]])
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == +1)[0]
    ends = np.where(diffs == -1)[0]
    N = len(freq)

    for s, e in zip(starts, ends):
        # Initial bounds of the "core" region
        i0 = max(0, s)
        i1 = min(N - 1, e - 1)

        buffer_bins = 2
        min_bandwidth_hz = 0.5

        # Expand edge
        while i0 > 0 and ratio[i0 - 1] > threshold_low:
            i0 -= 1
        while i1 < N - 1 and ratio[i1 + 1] > threshold_low:
            i1 += 1

        # Add buffer bins
        i0 = max(0, i0 - buffer_bins)
        i1 = min(N - 1, i1 + buffer_bins)

        # Enforce minimum bandwidth
        while (freq[i1] - freq[i0]) < min_bandwidth_hz:
            if i0 > 0:
                i0 -= 1
            if i1 < N - 1:
                i1 += 1
            if i0 == 0 and i1 == N - 1:
                break


        f_start = float(freq[i0])
        f_end = float(freq[i1])
        f0 = 0.5 * (f_start + f_end)
        bandwidth = f_end - f_start
        max_r = float(np.max(ratio[i0:i1 + 1]))

        line_details.append(Line(
            f_start=f_start,
            f_end=f_end,
            f0=f0,
            bandwidth=bandwidth,
            ratio=max_r
        ))

    return line_details


def line_locator(
    freq: np.ndarray,
    Pxx: np.ndarray,
    window_width_hz: float = 8,
    threshold: float = 10,
    fmin: float = 20,
    fmax: float = 2048,
) -> LineLocatorResult:
    """
    Locate the frequency bins that correspond to narrow lines in a periodogram.

    Based on method from [Gupta+Cornish 2024](https://arxiv.org/html/2312.11808v2)
    """

    freq = np.asarray(freq)
    Pxx = np.asarray(Pxx)

    if len(freq) != len(Pxx):
        raise ValueError("`freq` and `Pxx` must have the same length.")

    N = len(freq)
    df = np.median(np.diff(freq))  # assume roughly uniform spacing

    # Determine kernel size for running median
    half_bins = int(round((window_width_hz / df) / 2))
    kernel_size = max(1, 2 * half_bins + 1)

    running_median_psd = medfilt(Pxx, kernel_size=kernel_size)

    # Avoid division by zero
    eps = np.finfo(float).tiny
    ratio = Pxx / (running_median_psd + eps)

    # Build a mask for the specified frequency range
    in_range = np.ones(N, dtype=bool)
    if fmin is not None:
        in_range &= freq >= fmin
    if fmax is not None:
        in_range &= freq <= fmax

    # Identify "line" bins
    is_line_bin = (ratio > threshold) & in_range

    # PSD model: use Pxx where there's a line, running median elsewhere
    psd_model = np.where(is_line_bin, Pxx, running_median_psd)

    lines = _create_lines(is_line_bin, freq, ratio) if np.any(is_line_bin) else []

    return LineLocatorResult(
        running_median_psd=running_median_psd,
        is_line_bin=is_line_bin,
        psd_model=psd_model,
        lines=lines,
        ratio=ratio
    )




def plot_line_locator(freqs, pdgrm, running_median, lines, xlim=None, xscale='log'):
    fig, ax = plt.subplots()

    # Main PSD and median
    ax.loglog(freqs[1:-1], pdgrm[1:-1], label='Raw', alpha=0.5)
    ax.loglog(freqs[1:-1], running_median[1:-1], label='Running Median', color='tab:orange', lw=2.5, alpha=0.7)
    ax.set_xlabel(r'$f\ \mathrm{[Hz]}$')
    ax.set_ylabel(r'$P(f)\ \mathrm{[strain^2/Hz]}$')
    ax.legend(loc='upper right')
    ax.set_xscale(xscale)

    # Apply x-limits if provided
    if xlim:
        ax.set_xlim(xlim)
        fmin, fmax = xlim
        visible_lines = [l for l in lines if l.f_end >= fmin and l.f_start <= fmax]
    else:
        visible_lines = lines

    # Annotate number of lines in view
    ax.text(0.05, 0.9, f'Lines visible: {len(visible_lines)}', transform=ax.transAxes)

    # Prepare color map
    ratios = [l.ratio for l in visible_lines]
    cmap = plt.get_cmap('autumn')
    max_ratio = np.quantile(ratios, 0.8) if ratios else 1.0
    min_ratio = min(ratios) if ratios else 1.0

    # Highlight lines
    for line in visible_lines:
        color = cmap(min(line.ratio / max_ratio, 1.0))
        ax.axvspan(line.f_start, line.f_end, color=color, alpha=0.3, zorder=-10)
        ax.axvline(line.f0, color=color, linestyle='--', alpha=0.6, zorder=-10)

    # Colorbar
    if ratios:
        sm = ScalarMappable(norm=Normalize(vmin=min_ratio, vmax=max_ratio), cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label(r'$\mathrm{Pxx} / \mathrm{Median}$')

    fig.tight_layout()
    return fig, ax
