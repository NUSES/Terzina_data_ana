#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TERZINA FPA — publication-style finger plots.

Features
--------
- Reads TERZINA binary data.
- Scans all channels and identifies photoelectron peaks.
- Ranks channels by number of peaks and ADC ladder span.
- Saves plots WITH and WITHOUT Gaussian peak fits.
- Linear scale is the default.
- Logarithmic scale remains available through --yscale log.
- Wider X and Y axis ranges for clearer plots.

Examples
--------
# Default: linear scale, both with-fit and no-fit plots
python fingerplot_brightest_pixels.py RUN.bin --gain BOTH --top 32

# Only plots with Gaussian fits
python fingerplot_brightest_pixels.py RUN.bin --fit with

# Only plots without Gaussian fits
python fingerplot_brightest_pixels.py RUN.bin --fit without

# Logarithmic scale, both with-fit and no-fit
python fingerplot_brightest_pixels.py RUN.bin --yscale log

# HG gain, linear scale, top 20 channels
python fingerplot_brightest_pixels.py RUN.bin --gain HG --top 20

# Save plots in a specific directory
python fingerplot_brightest_pixels.py RUN.bin --out figures/

Requires
--------
numpy, scipy, matplotlib
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


__all__ = [
    "parse_file",
    "channel_values",
    "histogram",
    "analyse_channel",
    "rank_channels",
    "plot_grid",
    "plot_single",
    "run",
    "main",
]


# ==========================================================================
# TERZINA BINARY FORMAT
# ==========================================================================

DAQ_IDS = {
    0x00AAAA00: 1,
    0x00BBBB00: 2,
    0x00CCCC00: 3,
    0x00DDDD00: 4,
}

CONC_HEADER = 0xCA00FE11DD651983
VALID_PACK_ID = 0xA77A

DAQS = (1, 2, 3, 4)
ASICS = ("A", "B", "C", "D", "E")

CHANNELS_PER_ASIC = 32
N_CHANNELS = len(DAQS) * len(ASICS) * CHANNELS_PER_ASIC


# ==========================================================================
# ANALYSIS PARAMETERS
# ==========================================================================

NBINS = 16384

REBIN_FACTOR = 2

SMOOTH_SIGMA = 2.0

PEAK_PROMINENCE = 0.005
PEAK_DISTANCE = 10

GAUSS_FIT_WINDOW = 6

DEFAULT_GAIN = "HG"
DEFAULT_TOP = 32
DEFAULT_MIN_PEAKS = 3
DEFAULT_N_INDIVIDUAL = 3
DEFAULT_NCOLS = 4

DEFAULT_BAD_ASICS = ((3, "D"),)


# ==========================================================================
# PLOT DESIGN
# ==========================================================================

C = {
    "ink": "#172B3A",
    "muted": "#627586",

    "blue": "#176B87",
    "blue_light": "#D9EAF0",

    "teal": "#159A91",

    "coral": "#E4775E",
    "coral_light": "#F4C5B9",

    "grid": "#E5EBEF",
    "spine": "#B9C7D0",

    "white": "#FFFFFF",
    "panel": "#F7FAFB",
}


def set_plot_style():

    plt.rcParams.update({

        "figure.facecolor": C["white"],
        "axes.facecolor": C["white"],
        "savefig.facecolor": C["white"],
        "savefig.edgecolor": C["white"],

        "font.family": "DejaVu Sans",
        "font.size": 9,

        "text.color": C["ink"],
        "axes.labelcolor": C["ink"],
        "axes.titlecolor": C["ink"],

        "xtick.color": C["muted"],
        "ytick.color": C["muted"],

        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",

        "xtick.labelsize": 8,
        "ytick.labelsize": 8,

        "axes.linewidth": 0.8,

        "lines.solid_capstyle": "round",

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


set_plot_style()


# ==========================================================================
# AXIS STYLE
# ==========================================================================

def _style_axis(ax, yscale="linear"):

    ax.set_yscale(yscale)

    ax.set_axisbelow(True)

    ax.grid(
        True,
        axis="y",
        which="major",
        color=C["grid"],
        linewidth=0.8,
        linestyle="-",
    )

    if yscale == "log":

        ax.grid(
            True,
            axis="y",
            which="minor",
            color=C["grid"],
            linewidth=0.45,
            alpha=0.65,
            linestyle=":",
        )

    ax.grid(False, axis="x")

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for side in ("left", "bottom"):

        ax.spines[side].set_color(C["spine"])
        ax.spines[side].set_linewidth(0.8)

    ax.tick_params(
        axis="both",
        which="major",
        length=3.5,
        width=0.75,
    )

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def _positive_for_log(counts):

    return np.ma.masked_less_equal(counts, 0)


# ==========================================================================
# GAUSSIAN FUNCTION AND PEAK FITTING
# ==========================================================================

def gaussian(x, amplitude, mean, sigma):

    return amplitude * np.exp(
        -(x - mean) ** 2 / (2 * sigma ** 2)
    )


def _fit_peak_parameters(
    x: np.ndarray,
    y: np.ndarray,
    index: int,
) -> Optional[Tuple[float, float, float]]:
    """
    Fit a Gaussian around one detected peak.

    Returns
    -------
    amplitude, mean, sigma

    Returns None if the fit fails.
    """

    lo = max(0, index - GAUSS_FIT_WINDOW)

    hi = min(
        len(x),
        index + GAUSS_FIT_WINDOW + 1,
    )

    xw = x[lo:hi]
    yw = y[lo:hi]

    if len(xw) < 4:
        return None

    if not np.any(yw > 0):
        return None

    try:

        p0 = [
            float(yw.max()),
            float(x[index]),
            max(float(xw[-1] - xw[0]) / 4, 1.0),
        ]

        popt, _ = curve_fit(
            gaussian,
            xw,
            yw,
            p0=p0,
            bounds=(
                [0, xw[0], 0.1],
                [np.inf, xw[-1], np.inf],
            ),
            maxfev=5000,
        )

        return tuple(float(v) for v in popt)

    except (RuntimeError, ValueError):

        return None


def _fit_peak(
    x: np.ndarray,
    y: np.ndarray,
    index: int,
) -> float:
    """
    Return fitted peak position.
    If the fit fails, use the original bin centre.
    """

    params = _fit_peak_parameters(x, y, index)

    if params is None:
        return float(x[index])

    return params[1]


def _peak_fit_curves(
    centres: np.ndarray,
    counts: np.ndarray,
    peaks: np.ndarray,
):
    """
    Generate smooth Gaussian curves for all detected peaks.

    The fits are performed on the smoothed spectrum, while the
    original histogram remains visible in the plot.
    """

    smoothed = gaussian_filter1d(
        counts.astype(float),
        sigma=SMOOTH_SIGMA,
    )

    curves = []

    for index in peaks:

        params = _fit_peak_parameters(
            centres,
            smoothed,
            int(index),
        )

        if params is None:
            continue

        amplitude, mean, sigma = params

        half_width = max(
            3.5 * sigma,
            float(REBIN_FACTOR) * 2,
        )

        xmin = max(
            float(centres[0]),
            mean - half_width,
        )

        xmax = min(
            float(centres[-1]),
            mean + half_width,
        )

        xfit = np.linspace(
            xmin,
            xmax,
            180,
        )

        yfit = gaussian(
            xfit,
            amplitude,
            mean,
            sigma,
        )

        curves.append((xfit, yfit))

    return curves


# ==========================================================================
# SPECTRUM PLOTTING
# ==========================================================================

def _plot_spectrum(
    ax,
    centres,
    counts,
    peaks=None,
    positions=None,
    yscale="linear",
    compact=False,
    show_fit=False,
):
    """
    Draw the spectrum.

    show_fit=False:
        Histogram and detected peak markers.

    show_fit=True:
        Histogram, detected peak markers and Gaussian curves.
    """

    if yscale == "log":

        y = _positive_for_log(counts)
        baseline = 1

    else:

        y = counts
        baseline = 0

    # Histogram fill
    ax.fill_between(
        centres,
        y,
        baseline,
        color=C["blue_light"],
        alpha=0.95,
        linewidth=0,
        step="mid",
        zorder=1,
    )

    # Measured spectrum
    ax.step(
        centres,
        y,
        where="mid",
        color=C["blue"],
        linewidth=1.15 if compact else 1.45,
        zorder=3,
        label="Measured spectrum",
    )

    if peaks is not None and len(peaks):

        # --------------------------------------------------------------
        # Gaussian fits
        # --------------------------------------------------------------

        if show_fit:

            fit_curves = _peak_fit_curves(
                centres,
                counts,
                peaks,
            )

            for i, (xfit, yfit) in enumerate(fit_curves):

                if yscale == "log":
                    yfit_plot = np.maximum(yfit, 0.01)
                else:
                    yfit_plot = yfit

                ax.plot(
                    xfit,
                    yfit_plot,
                    color=C["coral"],
                    linewidth=1.25 if compact else 1.8,
                    linestyle="--",
                    alpha=0.95,
                    zorder=4,
                    label=(
                        "Gaussian peak fits"
                        if i == 0
                        else "_nolegend_"
                    ),
                )

        # --------------------------------------------------------------
        # Peak markers
        # --------------------------------------------------------------

        xp = (
            np.asarray(positions)
            if positions is not None
            else centres[peaks]
        )

        yp = counts[peaks]

        ax.scatter(
            xp,
            yp,
            s=22 if compact else 48,
            marker="o",
            facecolor=C["teal"],
            edgecolor=C["white"],
            linewidth=0.9,
            zorder=5,
            label="Detected p.e. peaks",
        )


# ==========================================================================
# X RANGE
# ==========================================================================

def _set_x_window(ax, centres, counts):

    # Consider only bins containing a significant fraction
    # of the maximum number of events.
    threshold = counts.max() * 0.005
    relevant = np.flatnonzero(counts >= threshold)

    if not relevant.size:
        return

    lo = centres[relevant[0]]
    hi = centres[relevant[-1]]

    span = max(
        float(hi - lo),
        float(REBIN_FACTOR),
    )

    pad = max(4.0, 0.03 * span)

    ax.set_xlim(
        max(0,lo - pad),
        min(NBINS, hi + pad),
    )
# ==========================================================================
# SAVE FIGURE
# ==========================================================================

def _finish(fig, save_path=None, show=False):

    if save_path:

        fig.savefig(
            save_path,
            dpi=320,
            bbox_inches="tight",
            facecolor=C["white"],
            edgecolor="none",
            pil_kwargs={"compress_level": 6},
        )

        print(f"  Saved: {save_path}")

    if show:

        plt.show()

    else:

        plt.close(fig)


# ==========================================================================
# BINARY PARSER
# ==========================================================================

def parse_file(
    path,
    max_packets=None,
    verbose=True,
):
    """
    Read TERZINA binary file.

    Each ASIC list stores 32 words per event.
    """

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"Binary file not found: {path}"
        )

    channels = {
        f"D{d}_{a}": []
        for d in DAQS
        for a in ASICS
    }

    events_per_daq = {
        d: 0
        for d in DAQS
    }

    n_packets = 0

    if verbose:

        print(f"File: {path}")

        print(
            f"Size: {os.path.getsize(path) / 1e6:.1f} MB"
        )

        print("Parsing", end="", flush=True)

    with open(path, "rb") as handle:

        while True:

            header = handle.read(12)

            if len(header) < 12:
                break

            id_pack = struct.unpack(
                "<I",
                header[0:4],
            )[0]

            len_pack = struct.unpack(
                "<I",
                header[4:8],
            )[0]

            if id_pack != VALID_PACK_ID:

                handle.read(len_pack)
                continue

            payload = handle.read(len_pack)

            if len(payload) < len_pack:
                break

            _parse_payload(
                payload,
                channels,
                events_per_daq,
            )

            n_packets += 1

            if verbose and n_packets % 10000 == 0:

                print(".", end="", flush=True)

            if (
                max_packets is not None
                and n_packets >= max_packets
            ):

                break

    if verbose:

        print(f"\nParsed {n_packets} packets.")

        for daq in DAQS:

            print(
                f"  DAQ{daq}: "
                f"{events_per_daq[daq]} events"
            )

    return channels, events_per_daq


def _parse_payload(
    payload,
    channels,
    events_per_daq,
):
    """Walk one payload's DAQ blocks."""

    offset = 0

    def u32():

        nonlocal offset

        if offset + 4 > len(payload):
            raise ValueError("Truncated payload")

        value = int.from_bytes(
            payload[offset:offset + 4],
            "little",
        )

        offset += 4

        return value

    def u64():

        nonlocal offset

        if offset + 8 > len(payload):
            raise ValueError("Truncated payload")

        value = int.from_bytes(
            payload[offset:offset + 8],
            "little",
        )

        offset += 8

        return value

    try:

        u64()

        if u64() != CONC_HEADER:
            return

        daq_id = u32()

        id_prog = 0

        while True:

            if offset >= len(payload) - 4:
                break

            if daq_id in DAQ_IDS:

                daq = DAQ_IDS[daq_id]

                # Trigger counter, data1, data2
                u32()
                u32()
                u32()

                for asic in ASICS:

                    channels[f"D{daq}_{asic}"].extend(
                        u32()
                        for _ in range(CHANNELS_PER_ASIC)
                    )

                # LOST, REAL
                u64()
                u64()

                events_per_daq[daq] += 1

                if daq == 4:
                    break

                daq_id = u32()

                id_prog += 1

            elif daq_id == 0xDEADBEEF:

                if id_prog == 3:
                    break

                id_prog += 1

                u32()

                daq_id = u32()

            else:

                id_prog += 1

                offset += 796

                if offset >= len(payload):
                    break

                daq_id = u32()

    except ValueError:

        # Ignore malformed or truncated payloads.
        return


# ==========================================================================
# SPECTRUM DECODING
# ==========================================================================

def channel_values(
    channels,
    daq,
    asic,
    channel,
    gain="HG",
):
    """Decode HG or LG field and invert the 14-bit ADC."""

    words = channels[f"D{daq}_{asic}"]

    if not words:

        return np.array(
            [],
            dtype=np.int64,
        )

    selected = np.asarray(
        words[channel::CHANNELS_PER_ASIC],
        dtype=np.int64,
    )

    gain = gain.upper()

    if gain == "HG":

        field = selected & 0x3FFF

    elif gain == "LG":

        field = (selected >> 14) & 0x3FFF

    else:

        raise ValueError(
            f"gain must be HG or LG, got {gain!r}"
        )

    return (0x3FFF - field) & 0x3FFF


# ==========================================================================
# HISTOGRAM
# ==========================================================================

def histogram(
    channels,
    daq,
    asic,
    channel,
    gain="HG",
):
    """Return rebinned counts and ADC bin centres."""

    values = channel_values(
        channels,
        daq,
        asic,
        channel,
        gain,
    )

    n_out = NBINS // REBIN_FACTOR

    if values.size == 0:

        centres = (
            np.arange(n_out) * REBIN_FACTOR
            + REBIN_FACTOR / 2
        )

        return np.zeros(n_out), centres

    counts, edges = np.histogram(
        values,
        bins=NBINS,
        range=(0, NBINS),
    )

    if REBIN_FACTOR > 1:

        usable = n_out * REBIN_FACTOR

        counts = (
            counts[:usable]
            .reshape(n_out, REBIN_FACTOR)
            .sum(axis=1)
        )

        centres = (
            edges[:usable]
            .reshape(n_out, REBIN_FACTOR)
            .mean(axis=1)
        )

    else:

        centres = (
            edges[:-1] + edges[1:]
        ) / 2

    return counts, centres


# ==========================================================================
# PEAK FINDING
# ==========================================================================

def analyse_channel(
    counts,
    centres,
):
    """
    Find p.e. peaks and estimate gain from consecutive peak spacings.
    """

    if counts.size == 0 or counts.max() == 0:

        return (
            np.array([], dtype=int),
            [],
            None,
        )

    smoothed = gaussian_filter1d(
        counts.astype(float),
        sigma=SMOOTH_SIGMA,
    )

    peaks, _ = find_peaks(
        smoothed,
        prominence=PEAK_PROMINENCE * max(
            smoothed.max(),
            1,
        ),
        distance=PEAK_DISTANCE,
    )

    if peaks.size == 0:

        return peaks, [], None

    positions = [
        _fit_peak(
            centres,
            smoothed,
            int(p),
        )
        for p in peaks
    ]

    gain = None

    spacings = np.diff(positions)

    if spacings.size:

        median = np.median(spacings)

        keep = (
            (spacings > 0.5 * median)
            & (spacings < 2.0 * median)
        )

        if keep.any():

            gain = float(
                np.mean(spacings[keep])
            )

    return peaks, positions, gain


# ==========================================================================
# CHANNEL SCAN AND RANKING
# ==========================================================================

def rank_channels(
    channels,
    events_per_daq,
    gain=DEFAULT_GAIN,
    min_peaks=DEFAULT_MIN_PEAKS,
    bad_asics=(),
    verbose=True,
):
    """
    Rank channels by peak count and ADC span of the detected ladder.
    """

    gain = gain.upper()

    if gain not in ("HG", "LG"):

        raise ValueError(
            f"gain must be HG or LG, got {gain!r}"
        )

    skip = set(bad_asics)

    if verbose:

        print(
            f"\nScanning {N_CHANNELS} channels ({gain})..."
        )

        if skip:

            print(
                "  Excluded: "
                + ", ".join(
                    f"DAQ{d} ASIC-{a}"
                    for d, a in sorted(skip)
                )
            )

    ranked = []

    n_skipped = 0

    for daq in DAQS:

        if events_per_daq.get(daq, 0) == 0:
            continue

        for asic in ASICS:

            if (daq, asic) in skip:

                n_skipped += CHANNELS_PER_ASIC
                continue

            for channel in range(CHANNELS_PER_ASIC):

                counts, centres = histogram(
                    channels,
                    daq,
                    asic,
                    channel,
                    gain,
                )

                if counts.max() == 0:
                    continue

                peaks, positions, chan_gain = analyse_channel(
                    counts,
                    centres,
                )

                if peaks.size < min_peaks:
                    continue

                nonzero = np.flatnonzero(counts)

                ranked.append({

                    "DAQ": daq,
                    "ASIC": asic,
                    "CH": channel,

                    "N_PEAKS": int(peaks.size),

                    "LADDER": float(
                        positions[-1] - positions[0]
                    ),

                    "GAIN": chan_gain,

                    "ADC_MIN": float(
                        centres[nonzero.min()]
                    ),

                    "ADC_MAX": float(
                        centres[nonzero.max()]
                    ),
                })

    ranked.sort(
        key=lambda row: (
            row["N_PEAKS"],
            row["LADDER"],
        ),
        reverse=True,
    )

    if verbose:

        pool = N_CHANNELS - n_skipped

        note = (
            f" (skipped {n_skipped})"
            if n_skipped
            else ""
        )

        print(
            f"  Resolving at least {min_peaks} peaks: "
            f"{len(ranked)} / {pool}{note}"
        )

    return ranked


# ==========================================================================
# Y AXIS RANGE
# ==========================================================================

def _set_y_range(
    ax,
    counts,
    yscale="linear",
    compact=False,
):
    """Set wider Y limits, leaving extra room above the peaks."""

    ymax = float(counts.max()) if counts.size else 0

    if yscale == "log":

        positive = counts[counts > 0]

        ymin = (
            max(
                0.35,
                float(positive.min()) * 0.45,
            )
            if positive.size
            else 0.35
        )

        # More vertical room in logarithmic mode.
        top_factor = 2.8 if compact else 3.0

        ax.set_ylim(
            ymin,
            max(
                ymin * 14,
                ymax * top_factor,
            ),
        )

    else:

        # More space above the tallest peak.
        top_factor = 1.32 if compact else 1.42

        ax.set_ylim(
            0,
            ymax * top_factor if ymax else 1,
        )


# ==========================================================================
# SINGLE CHANNEL FIGURE
# ==========================================================================

def plot_single(
    channels,
    daq,
    asic,
    channel,
    gain,
    save_path=None,
    show=False,
    yscale="linear",
    show_fit=False,
):
    """
    Create a detailed finger plot.

    show_fit=True:
        Add Gaussian curves.

    show_fit=False:
        Show only spectrum and peak markers.
    """

    if yscale not in ("linear", "log"):

        raise ValueError(
            "yscale must be 'linear' or 'log'"
        )

    counts, centres = histogram(
        channels,
        daq,
        asic,
        channel,
        gain,
    )

    peaks, positions, chan_gain = analyse_channel(
        counts,
        centres,
    )

    fig, ax = plt.subplots(
        figsize=(11.8, 6.2)
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.97,
        bottom=0.13,
        top=0.82,
    )

    _plot_spectrum(
        ax,
        centres,
        counts,
        peaks,
        positions,
        yscale=yscale,
        show_fit=show_fit,
    )

    _set_x_window(
        ax,
        centres,
        counts,
    )

    _style_axis(
        ax,
        yscale,
    )

    _set_y_range(
        ax,
        counts,
        yscale,
    )

    # --------------------------------------------------------------
    # Peak labels
    # --------------------------------------------------------------

    if peaks.size:

        for i, (p, xpos) in enumerate(
            zip(peaks, positions)
        ):

            ypos = max(
                float(counts[p]),
                1.0,
            )

            offset = (
                12 if i % 2 == 0
                else 25
            )

            ax.annotate(

                f"{i + 1} p.e.",

                xy=(xpos, ypos),

                xytext=(0, offset),

                textcoords="offset points",

                ha="center",
                va="bottom",

                fontsize=8.5,
                fontweight="semibold",

                color=C["ink"],

                arrowprops=dict(
                    arrowstyle="-",
                    color=C["coral"],
                    lw=0.85,
                    shrinkA=2,
                    shrinkB=4,
                ),

                bbox=dict(
                    boxstyle="round,pad=0.22",
                    facecolor=C["white"],
                    edgecolor="none",
                    alpha=0.88,
                ),

                zorder=8,
            )

    # --------------------------------------------------------------
    # Labels and metadata
    # --------------------------------------------------------------

    gain_label = (
        f"{chan_gain:.1f} ADC/p.e."
        if chan_gain is not None
        else "not resolved"
    )

    fit_label = (
        "WITH GAUSSIAN FIT"
        if show_fit
        else "WITHOUT FIT"
    )

    ax.set_xlabel(
        "ADC channel",
        labelpad=10,
        fontweight="medium",
    )

    ax.set_ylabel(
        "Events / 2 ADC",
        labelpad=10,
        fontweight="medium",
    )

    fig.text(
        0.09,
        0.955,
        "TERZINA  /  FPA CHARACTERISATION",
        fontsize=8,
        fontweight="bold",
        color=C["teal"],
        ha="left",
        va="top",
    )

    fig.text(
        0.09,
        0.905,
        f"DAQ {daq}   ·   ASIC {asic}   ·   Channel {channel}",
        fontsize=18,
        fontweight="bold",
        color=C["ink"],
        ha="left",
        va="top",
    )

    fig.text(
        0.97,
        0.91,
        f"{gain.upper()} GAIN\n"
        f"{yscale.upper()} SCALE\n"
        f"{fit_label}",
        fontsize=8,
        fontweight="bold",
        color=C["muted"],
        ha="right",
        va="top",
        linespacing=1.6,
    )

    fig.text(
        0.09,
        0.855,
        f"{peaks.size} detected p.e. peaks     |     "
        f"Estimated gain: {gain_label}     |     "
        f"Entries: {int(counts.sum()):,}",
        fontsize=9,
        color=C["muted"],
        ha="left",
        va="top",
    )

    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor=C["white"],
        edgecolor=C["grid"],
        framealpha=0.95,
        fontsize=8,
        borderpad=0.7,
    )

    fig.text(
        0.97,
        0.035,
        f"Y axis: {yscale}  ·  "
        f"Histogram rebinned ×{REBIN_FACTOR}",
        fontsize=7.5,
        color=C["muted"],
        ha="right",
    )

    _finish(
        fig,
        save_path,
        show,
    )

    return fig


# ==========================================================================
# RANKED GRID
# ==========================================================================

def plot_grid(
    channels,
    ranked,
    gain,
    top=DEFAULT_TOP,
    ncols=DEFAULT_NCOLS,
    save_path=None,
    show=False,
    yscale="linear",
    show_fit=False,
):
    """Create a multi-panel grid for the best channels."""

    if yscale not in ("linear", "log"):

        raise ValueError(
            "yscale must be 'linear' or 'log'"
        )

    if ncols < 1:
        raise ValueError("ncols must be at least 1")

    selection = list(ranked[:top])

    if not selection:

        print("  Nothing to plot.")

        return None, []

    fit_label = (
        "WITH FIT"
        if show_fit
        else "WITHOUT FIT"
    )

    print()
    print("=" * 78)

    print(
        f"  TOP {len(selection)} CHANNELS — "
        f"{gain} — {yscale.upper()} — {fit_label}"
    )

    print("=" * 78)

    print(
        f"  {'RANK':>4}  "
        f"{'CHANNEL':<19} "
        f"{'PEAKS':>5} "
        f"{'LADDER':>8} "
        f"{'GAIN':>9}  "
        f"{'ADC RANGE':>17}"
    )

    print("  " + "-" * 73)

    nrows = int(
        np.ceil(len(selection) / ncols)
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            4.8 * ncols,
            3.25 * nrows,
        ),
        squeeze=False,
    )

    fig.patch.set_facecolor(C["white"])

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.09,
        top=0.87,
        hspace=0.53,
        wspace=0.27,
    )

    for i, entry in enumerate(selection):

        ax = axes[i // ncols][i % ncols]

        daq = entry["DAQ"]
        asic = entry["ASIC"]
        channel = entry["CH"]

        counts, centres = histogram(
            channels,
            daq,
            asic,
            channel,
            gain,
        )

        peaks, positions, _ = analyse_channel(
            counts,
            centres,
        )

        _plot_spectrum(
            ax,
            centres,
            counts,
            peaks,
            positions,
            yscale=yscale,
            compact=True,
            show_fit=show_fit,
        )

        _set_x_window(
            ax,
            centres,
            counts,
        )

        _style_axis(
            ax,
            yscale,
        )

        _set_y_range(
            ax,
            counts,
            yscale,
            compact=True,
        )

        gain_text = (
            f"{entry['GAIN']:.1f}"
            if entry["GAIN"] is not None
            else "n/a"
        )

        ax.set_title(
            f"DAQ {daq}  /  ASIC {asic}  /  CH {channel}",
            loc="left",
            fontsize=9.2,
            fontweight="bold",
            pad=8,
        )

        ax.text(
            0.0,
            1.015,
            f"{entry['N_PEAKS']} peaks     ·     "
            f"gain {gain_text} ADC/p.e.",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.5,
            color=C["muted"],
        )

        ax.text(
            0.98,
            0.94,
            f"{i + 1:02d}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=C["teal"],
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor="#EAF4F5",
                edgecolor="none",
            ),
        )

        ax.tick_params(
            labelsize=7.5,
            length=2.5,
        )

        print(
            f"  {i + 1:>4}  "
            f"DAQ{daq} ASIC-{asic} CH{channel:<7} "
            f"{entry['N_PEAKS']:>5} "
            f"{entry['LADDER']:>8.1f} "
            f"{gain_text:>9}  "
            f"{entry['ADC_MIN']:>7.0f}-"
            f"{entry['ADC_MAX']:<8.0f}"
        )

    # Hide empty panels
    for j in range(
        len(selection),
        nrows * ncols,
    ):

        axes[
            j // ncols
        ][
            j % ncols
        ].set_visible(False)

    # Figure header
    fig.text(
        0.075,
        0.965,
        "TERZINA  /  FPA CHARACTERISATION",
        fontsize=8,
        fontweight="bold",
        color=C["teal"],
        ha="left",
        va="top",
    )

    fig.text(
        0.075,
        0.925,
        f"Brightest pixels  —  {gain.upper()} gain",
        fontsize=17,
        fontweight="bold",
        color=C["ink"],
        ha="left",
        va="top",
    )

    fig.text(
        0.985,
        0.935,
        f"{yscale.upper()} SCALE\n"
        f"TOP {len(selection)} CHANNELS\n"
        f"{fit_label}",
        fontsize=8,
        fontweight="bold",
        color=C["muted"],
        ha="right",
        va="top",
        linespacing=1.5,
    )

    fig.supxlabel(
        "ADC channel",
        fontsize=10,
        y=0.035,
        color=C["ink"],
    )

    fig.supylabel(
        "Events / 2 ADC",
        fontsize=10,
        x=0.025,
        color=C["ink"],
    )

    _finish(
        fig,
        save_path,
        show,
    )

    return fig, selection


# ==========================================================================
# RUN ORCHESTRATION
# ==========================================================================

def run(
    bin_file,
    gain=DEFAULT_GAIN,
    top=DEFAULT_TOP,
    out_dir=None,
    min_peaks=DEFAULT_MIN_PEAKS,
    n_individual=DEFAULT_N_INDIVIDUAL,
    ncols=DEFAULT_NCOLS,
    max_packets=None,
    bad_asics=DEFAULT_BAD_ASICS,
    show=None,
    yscale="linear",
    fit_mode="both",
):
    """
    Parse the run, rank channels and export plots.

    yscale:
        linear (default) or log

    fit_mode:
        both    -> save with-fit and without-fit plots
        with    -> save only plots with Gaussian fits
        without -> save only plots without Gaussian fits
    """

    gain = str(gain).upper()

    if gain == "BOTH":

        gains = ["HG", "LG"]

    else:

        gains = [gain]

    for branch in gains:

        if branch not in ("HG", "LG"):

            raise ValueError(
                f"gain must be HG, LG or BOTH, got {gain!r}"
            )

    yscale = yscale.lower()

    if yscale not in ("linear", "log"):

        raise ValueError(
            "yscale must be linear or log"
        )

    fit_mode = fit_mode.lower()

    if fit_mode not in (
        "both",
        "with",
        "without",
    ):

        raise ValueError(
            "fit_mode must be both, with or without"
        )

    if fit_mode == "both":

        fit_variants = [
            (False, "nofit"),
            (True, "fit"),
        ]

    elif fit_mode == "with":

        fit_variants = [
            (True, "fit"),
        ]

    else:

        fit_variants = [
            (False, "nofit"),
        ]

    if top < 1:
        raise ValueError("top must be at least 1")

    if n_individual < 0:
        raise ValueError("individual cannot be negative")

    if min_peaks < 1:
        raise ValueError("min_peaks must be at least 1")

    if ncols < 1:
        raise ValueError("ncols must be at least 1")

    if show is None:

        show = "ipykernel" in sys.modules

    if out_dir is None:

        out_dir = os.path.dirname(
            os.path.abspath(bin_file)
        )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    channels, events_per_daq = parse_file(
        bin_file,
        max_packets,
    )

    results = {}

    for branch in gains:

        ranked = rank_channels(
            channels,
            events_per_daq,
            gain=branch,
            min_peaks=min_peaks,
            bad_asics=bad_asics,
        )

        if not ranked:

            print(
                f"  No {branch} channel resolved at least "
                f"{min_peaks} peaks — nothing to plot."
            )

            results[branch] = []

            continue

        selection = list(
            ranked[:top]
        )

        # --------------------------------------------------------------
        # Save both plot variants, or the selected one.
        # --------------------------------------------------------------

        for show_fit, fit_suffix in fit_variants:

            print()
            print(
                f"Exporting {branch} — "
                f"{fit_suffix.upper()} — "
                f"{yscale.upper()}"
            )

            suffix = (
                f"_{yscale}_{fit_suffix}"
            )

            # Grid
            plot_grid(
                channels,
                ranked,
                branch,
                top=top,
                ncols=ncols,
                save_path=os.path.join(
                    out_dir,
                    f"bright_fingerplots_{branch}{suffix}.png",
                ),
                show=show,
                yscale=yscale,
                show_fit=show_fit,
            )

            # Individual plots
            for entry in selection[:n_individual]:

                daq = entry["DAQ"]
                asic = entry["ASIC"]
                channel = entry["CH"]

                plot_single(
                    channels,
                    daq,
                    asic,
                    channel,
                    branch,
                    save_path=os.path.join(
                        out_dir,
                        f"bright_DAQ{daq}_ASIC{asic}_"
                        f"CH{channel}_{branch}{suffix}.png",
                    ),
                    show=show,
                    yscale=yscale,
                    show_fit=show_fit,
                )

        results[branch] = selection

    return results


# ==========================================================================
# COMMAND LINE
# ==========================================================================

def _asic_spec(text):

    """Parse DAQ:ASIC, e.g. 3:D."""

    try:

        daq_text, asic = text.split(":", 1)

        daq = int(daq_text)

    except ValueError as exc:

        raise argparse.ArgumentTypeError(
            f"expected DAQ:ASIC such as 3:D, got {text!r}"
        ) from exc

    asic = asic.strip().upper()

    if daq not in DAQS or asic not in ASICS:

        raise argparse.ArgumentTypeError(
            f"DAQ must be 1-4 and ASIC A-E, got {text!r}"
        )

    return daq, asic


def build_parser():

    default_excludes = ",".join(
        f"{d}:{a}"
        for d, a in DEFAULT_BAD_ASICS
    )

    parser = argparse.ArgumentParser(

        description=(
            "Publication-style finger plots "
            "of TERZINA FPA channels."
        ),

        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "bin_file",
        help="TERZINA .bin readout file",
    )

    parser.add_argument(
        "--gain",
        default=DEFAULT_GAIN,
        choices=["HG", "LG", "BOTH"],
        help="gain branch to analyse",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        help="channels to show in the grid",
    )

    parser.add_argument(
        "--out",
        dest="out_dir",
        default=None,
        help="output directory",
    )

    parser.add_argument(
        "--min-peaks",
        type=int,
        default=DEFAULT_MIN_PEAKS,
        help="minimum number of peaks for ranking",
    )

    parser.add_argument(
        "--individual",
        type=int,
        default=DEFAULT_N_INDIVIDUAL,
        help="number of individual plots to save",
    )

    parser.add_argument(
        "--ncols",
        type=int,
        default=DEFAULT_NCOLS,
        help="columns in the grid",
    )

    parser.add_argument(
        "--max-packets",
        type=int,
        default=None,
        help="stop after this many valid packets",
    )

    parser.add_argument(
        "--yscale",
        choices=["linear", "log"],
        default="linear",
        help=(
            "Y-axis scale; linear is default, "
            "log is optional"
        ),
    )

    parser.add_argument(
        "--fit",
        dest="fit_mode",
        choices=["both", "with", "without"],
        default="both",
        help=(
            "save both fit/no-fit versions, "
            "only fit, or only no-fit"
        ),
    )

    parser.add_argument(
        "--exclude",
        type=_asic_spec,
        action="append",
        default=None,
        metavar="DAQ:ASIC",
        help=(
            "exclude an ASIC; repeatable "
            f"(default: {default_excludes})"
        ),
    )

    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="analyse every ASIC",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="display figures as well as saving them",
    )

    return parser


# ==========================================================================
# MAIN
# ==========================================================================

def main(argv=None):

    args = build_parser().parse_args(argv)

    if args.keep_all:

        bad_asics = ()

    elif args.exclude:

        bad_asics = args.exclude

    else:

        bad_asics = DEFAULT_BAD_ASICS

    try:

        run(
            args.bin_file,

            gain=args.gain,

            top=args.top,

            out_dir=args.out_dir,

            min_peaks=args.min_peaks,

            n_individual=args.individual,

            ncols=args.ncols,

            max_packets=args.max_packets,

            bad_asics=bad_asics,

            show=True if args.show else None,

            yscale=args.yscale,

            fit_mode=args.fit_mode,
        )

    except (
        FileNotFoundError,
        ValueError,
        OSError,
    ) as error:

        print(
            f"error: {error}",
            file=sys.stderr,
        )

        return 1

    return 0


if __name__ == "__main__":

    sys.exit(main())
