#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TERZINA FPA — finger plots of the brightest pixels.

Parses a TERZINA binary readout file (4 DAQs x 5 ASICs x 32 channels),
scans every channel, and plots the charge spectra of those resolving the
most photoelectron peaks — the channels that actually saw light.

Channels are ranked by ``(number of p.e. peaks, ADC span of the ladder)``.
Ranking on mean ADC does not work: the pedestal dominates the mean in every
channel equally, and a stuck channel pegged at full scale outranks a
genuinely illuminated one.

Command line
------------
    fingerplot_brightest_pixels.py RUN.bin
    fingerplot_brightest_pixels.py RUN.bin --gain LG --top 16
    fingerplot_brightest_pixels.py RUN.bin --gain BOTH --out results/

Library
-------
    from fingerplot_brightest_pixels import run
    results = run("RUN.bin", gain="BOTH", top=16)

Outputs, written to ``--out`` (default: the binary's own directory)
------------------------------------------------------------------
    bright_fingerplots_<GAIN>.png           grid of the best channels
    bright_DAQ<d>_ASIC<a>_CH<c>_<GAIN>.png  full-size, top few

Requires: numpy, scipy, matplotlib.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
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

# --------------------------------------------------------------------------
# Readout format
# --------------------------------------------------------------------------

DAQ_IDS = {0x00AAAA00: 1, 0x00BBBB00: 2, 0x00CCCC00: 3, 0x00DDDD00: 4}
CONC_HEADER = 0xCA00FE11DD651983
VALID_PACK_ID = 0xA77A

DAQS: Tuple[int, ...] = (1, 2, 3, 4)
ASICS: Tuple[str, ...] = ("A", "B", "C", "D", "E")
CHANNELS_PER_ASIC = 32
N_CHANNELS = len(DAQS) * len(ASICS) * CHANNELS_PER_ASIC

# --------------------------------------------------------------------------
# Analysis defaults — tune here or override on the command line
# --------------------------------------------------------------------------

NBINS = 16384            # 14-bit ADC
REBIN_FACTOR = 2         # 2 -> 8192 bins of 2 ADC each
SMOOTH_SIGMA = 2.0       # Gaussian smoothing applied before peak finding
PEAK_PROMINENCE = 0.005  # peak prominence, as a fraction of the tallest peak
PEAK_DISTANCE = 10       # minimum peak separation, in rebinned bins
GAUSS_FIT_WINDOW = 6     # half-window for the per-peak Gaussian fit

DEFAULT_GAIN = "HG"
DEFAULT_TOP = 32          # channels in the grid
DEFAULT_MIN_PEAKS = 3     # ignore channels resolving fewer peaks than this
DEFAULT_N_INDIVIDUAL = 3  # full-size single-channel plots to also save
DEFAULT_NCOLS = 4         # columns in the grid

# The NI readout kit carries one dead ASIC that fires on every event
# regardless of light, and would otherwise dominate any brightness search.
# Bench name "DAQ 2 ASIC D"; DAQ 3 ASIC D in the binary numbering used here.
# Irrelevant to the LNGS and Geneva datasets, which use different hardware —
# pass --keep-all for those.
DEFAULT_BAD_ASICS: Tuple[Tuple[int, str], ...] = ((3, "D"),)


# ==========================================================================
# Parsing
# ==========================================================================

def parse_file(
    path: str,
    max_packets: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, List[int]], Dict[int, int]]:
    """Read a TERZINA binary file.

    Returns ``(channels, events_per_daq)`` where ``channels["D2_C"]`` is a
    flat list of raw 32-bit readout words, 32 per event in channel order,
    so channel ``c``'s history is ``words[c::32]``. HG and LG are decoded on
    demand by :func:`channel_values`; keeping the words rather than two
    decoded lists halves the memory.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Binary file not found: {path}")

    channels: Dict[str, List[int]] = {
        f"D{d}_{a}": [] for d in DAQS for a in ASICS
    }
    events_per_daq: Dict[int, int] = {d: 0 for d in DAQS}
    n_packets = 0

    if verbose:
        print(f"File: {path}")
        print(f"Size: {os.path.getsize(path) / 1e6:.1f} MB")
        print("Parsing", end="", flush=True)

    with open(path, "rb") as handle:
        while True:
            header = handle.read(12)
            if len(header) < 12:
                break

            id_pack = struct.unpack("<I", header[0:4])[0]
            len_pack = struct.unpack("<I", header[4:8])[0]

            if id_pack != VALID_PACK_ID:
                handle.read(len_pack)  # unknown packet: skip its payload
                continue

            payload = handle.read(len_pack)
            if len(payload) < len_pack:
                break  # truncated trailing packet

            _parse_payload(payload, channels, events_per_daq)
            n_packets += 1

            if verbose and n_packets % 10000 == 0:
                print(".", end="", flush=True)
            if max_packets and n_packets >= max_packets:
                break

    if verbose:
        print(f"\nParsed {n_packets} packets.")
        for daq in DAQS:
            print(f"  DAQ{daq}: {events_per_daq[daq]} events")

    return channels, events_per_daq


def _parse_payload(
    payload: bytes,
    channels: Dict[str, List[int]],
    events_per_daq: Dict[int, int],
) -> None:
    """Walk one payload's DAQ blocks, appending channel words as it goes."""
    offset = 0

    def u32() -> int:
        nonlocal offset
        value = int.from_bytes(payload[offset:offset + 4], "little")
        offset += 4
        return value

    def u64() -> int:
        nonlocal offset
        value = int.from_bytes(payload[offset:offset + 8], "little")
        offset += 8
        return value

    u64()                     # timestamp
    if u64() != CONC_HEADER:  # concentrator magic
        return

    daq_id = u32()
    id_prog = 0

    while True:
        if offset >= len(payload) - 4:
            break

        if daq_id in DAQ_IDS:
            daq = DAQ_IDS[daq_id]

            u32(), u32(), u32()  # trigger counter, data1, data2
            for asic in ASICS:
                channels[f"D{daq}_{asic}"].extend(
                    u32() for _ in range(CHANNELS_PER_ASIC)
                )
            u64(), u64()         # LOST, REAL

            events_per_daq[daq] += 1

            if daq == 4:         # DAQ4 closes the packet
                break
            daq_id = u32()
            id_prog += 1

        elif daq_id == 0xDEADBEEF:
            if id_prog == 3:
                break
            id_prog += 1
            u32()
            daq_id = u32()

        else:                    # unknown ID: attempt to resynchronise
            id_prog += 1
            offset += 796
            if offset >= len(payload):
                break
            daq_id = u32()


# ==========================================================================
# Per-channel spectra
# ==========================================================================

def channel_values(
    channels: Dict[str, List[int]],
    daq: int,
    asic: str,
    channel: int,
    gain: str = "HG",
) -> np.ndarray:
    """Decode one channel's per-event ADC values.

    Word layout is HG in bits 0-13, LG in bits 14-27, HIT in bits 28-31.
    The ASIC ADC is inverted — more light gives a lower count — so each
    field is flipped with ``0x3FFF - field``.
    """
    words = channels[f"D{daq}_{asic}"]
    if not words:
        return np.array([], dtype=np.int64)

    selected = np.array(words[channel::CHANNELS_PER_ASIC], dtype=np.int64)

    gain = gain.upper()
    if gain == "HG":
        field = selected & 0x3FFF
    elif gain == "LG":
        field = (selected >> 14) & 0x3FFF
    else:
        raise ValueError(f"gain must be 'HG' or 'LG', got {gain!r}")

    return (0x3FFF - field) & 0x3FFF


def histogram(
    channels: Dict[str, List[int]],
    daq: int,
    asic: str,
    channel: int,
    gain: str = "HG",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(counts, bin_centres)`` for one channel, rebinned."""
    values = channel_values(channels, daq, asic, channel, gain)
    n_out = NBINS // REBIN_FACTOR

    if values.size == 0:
        centres = np.arange(n_out) * REBIN_FACTOR + REBIN_FACTOR / 2
        return np.zeros(n_out), centres

    counts, edges = np.histogram(values, bins=NBINS, range=(0, NBINS))

    if REBIN_FACTOR > 1:
        usable = n_out * REBIN_FACTOR
        counts = counts[:usable].reshape(n_out, REBIN_FACTOR).sum(axis=1)
        centres = edges[:usable].reshape(n_out, REBIN_FACTOR).mean(axis=1)
    else:
        centres = (edges[:-1] + edges[1:]) / 2

    return counts, centres


# ==========================================================================
# Peak finding
# ==========================================================================

def gaussian(x, amplitude, mean, sigma):
    """Three-parameter Gaussian, for :func:`scipy.optimize.curve_fit`."""
    return amplitude * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


def _fit_peak(x: np.ndarray, y: np.ndarray, index: int) -> float:
    """Gaussian-fit one peak, falling back to the bin centre on failure."""
    lo = max(0, index - GAUSS_FIT_WINDOW)
    hi = min(len(x), index + GAUSS_FIT_WINDOW + 1)
    xw, yw = x[lo:hi], y[lo:hi]

    try:
        popt, _ = curve_fit(
            gaussian, xw, yw,
            p0=[yw.max(), x[index], (xw[-1] - xw[0]) / 4],
            maxfev=5000,
        )
        return float(popt[1])
    except (RuntimeError, ValueError):
        return float(x[index])


def analyse_channel(
    counts: np.ndarray,
    centres: np.ndarray,
) -> Tuple[np.ndarray, List[float], Optional[float]]:
    """Find the p.e. peaks in one spectrum.

    Returns ``(peak indices, fitted positions, gain)``. The gain is the mean
    spacing between consecutive peaks, with spacings far from the median
    discarded; it is ``None`` when fewer than two peaks are found.
    """
    smoothed = gaussian_filter1d(counts.astype(float), sigma=SMOOTH_SIGMA)

    peaks, _ = find_peaks(
        smoothed,
        prominence=PEAK_PROMINENCE * max(smoothed.max(), 1),
        distance=PEAK_DISTANCE,
    )
    if peaks.size == 0:
        return peaks, [], None

    positions = [_fit_peak(centres, smoothed, p) for p in peaks]

    gain = None
    spacings = np.diff(positions)
    if spacings.size:
        median = np.median(spacings)
        keep = (spacings > 0.5 * median) & (spacings < 2.0 * median)
        if keep.any():
            gain = float(np.mean(spacings[keep]))

    return peaks, positions, gain


# ==========================================================================
# Scan
# ==========================================================================

def rank_channels(
    channels: Dict[str, List[int]],
    events_per_daq: Dict[int, int],
    gain: str = DEFAULT_GAIN,
    min_peaks: int = DEFAULT_MIN_PEAKS,
    bad_asics: Sequence[Tuple[int, str]] = (),
    verbose: bool = True,
) -> List[dict]:
    """Rank every channel by how much of the p.e. ladder it resolves.

    The sort key is ``(number of peaks, ADC span of the ladder)``, best
    first; the span breaks ties in favour of the channel whose peaks reach
    higher. Channels resolving fewer than ``min_peaks`` peaks are dropped.
    """
    gain = gain.upper()
    skip = set(bad_asics)

    if verbose:
        print(f"\nScanning {N_CHANNELS} channels ({gain})...")
        if skip:
            listed = ", ".join(f"DAQ{d} ASIC-{a}" for d, a in sorted(skip))
            print(f"  Excluded: {listed}")

    ranked: List[dict] = []
    n_skipped = 0

    for daq in DAQS:
        if events_per_daq[daq] == 0:
            continue

        for asic in ASICS:
            if (daq, asic) in skip:
                n_skipped += CHANNELS_PER_ASIC
                continue

            for channel in range(CHANNELS_PER_ASIC):
                counts, centres = histogram(channels, daq, asic, channel, gain)
                if counts.max() == 0:
                    continue

                peaks, positions, chan_gain = analyse_channel(counts, centres)
                if peaks.size < min_peaks:
                    continue

                nonzero = np.flatnonzero(counts)
                ranked.append({
                    "DAQ": daq,
                    "ASIC": asic,
                    "CH": channel,
                    "N_PEAKS": int(peaks.size),
                    "LADDER": float(positions[-1] - positions[0]),
                    "GAIN": chan_gain,
                    "ADC_MIN": float(centres[nonzero.min()]),
                    "ADC_MAX": float(centres[nonzero.max()]),
                })

    ranked.sort(key=lambda r: (r["N_PEAKS"], r["LADDER"]), reverse=True)

    if verbose:
        pool = N_CHANNELS - n_skipped
        note = f" (skipped {n_skipped})" if n_skipped else ""
        print(f"  Resolving at least {min_peaks} peaks: "
              f"{len(ranked)} / {pool}{note}")

    return ranked


# ==========================================================================
# Plots
# ==========================================================================

def _finish(fig, save_path: Optional[str], show: bool) -> None:
    """Save a figure, then display or close it.

    A figure must not be closed before it is shown, or it never reaches a
    notebook's output cell.
    """
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_single(
    channels: Dict[str, List[int]],
    daq: int,
    asic: str,
    channel: int,
    gain: str,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Full-size finger plot for one channel, peaks labelled by p.e. number."""
    counts, centres = histogram(channels, daq, asic, channel, gain)
    peaks, _, chan_gain = analyse_channel(counts, centres)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(centres, counts, "b-", lw=0.8, alpha=0.7, label="Raw")

    if peaks.size:
        ax.plot(centres[peaks], counts[peaks], "go", markersize=8,
                label="Detected peaks")
        for i, p in enumerate(peaks):
            ax.annotate(
                f"{i + 1} p.e.\n({centres[p]:.0f})",
                xy=(centres[p], counts[p]),
                xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=8,
                color="darkgreen", fontweight="bold",
            )

    nonzero = np.flatnonzero(counts)
    if nonzero.size:
        ax.set_xlim(centres[nonzero.min()] - 50, centres[nonzero.max()] + 50)

    gain_text = f", gain {chan_gain:.1f} ADC/p.e." if chan_gain else ""
    ax.set_yscale("log")
    ax.set_xlabel("ADC bin", fontsize=13)
    ax.set_ylabel("Counts", fontsize=13)
    ax.set_title(f"DAQ{daq} ASIC-{asic} CH{channel} {gain}"
                 f"  ({peaks.size} peaks{gain_text})", fontsize=14)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _finish(fig, save_path, show)
    return fig


def plot_grid(
    channels: Dict[str, List[int]],
    ranked: Sequence[dict],
    gain: str,
    top: int = DEFAULT_TOP,
    ncols: int = DEFAULT_NCOLS,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Grid of the best channels' spectra, plus a ranking table on stdout."""
    selection = list(ranked[:top])
    if not selection:
        print("  Nothing to plot.")
        return None, []

    print()
    print("=" * 70)
    print(f"  TOP {len(selection)} CHANNELS BY RESOLVED p.e. PEAKS ({gain})")
    print("=" * 70)
    print(f"  {'RANK':>4}  {'CHANNEL':<18} {'PEAKS':>5} {'LADDER':>8} "
          f"{'GAIN':>7}  {'ADC RANGE':>16}")
    print("  " + "-" * 64)

    nrows = int(np.ceil(len(selection) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 2.9 * nrows), squeeze=False
    )

    for i, entry in enumerate(selection):
        ax = axes[i // ncols][i % ncols]
        daq, asic, channel = entry["DAQ"], entry["ASIC"], entry["CH"]

        counts, centres = histogram(channels, daq, asic, channel, gain)
        peaks, _, _ = analyse_channel(counts, centres)

        ax.plot(centres, counts, "-", color="darkblue", lw=0.7)
        if peaks.size:
            ax.plot(centres[peaks], counts[peaks], "o",
                    color="darkgreen", markersize=4)

        nonzero = np.flatnonzero(counts)
        if nonzero.size:
            span = centres[nonzero.max()] - centres[nonzero.min()]
            pad = max(10, 0.02 * span)
            ax.set_xlim(centres[nonzero.min()] - pad,
                        centres[nonzero.max()] + pad)

        gain_text = f"{entry['GAIN']:.1f}" if entry["GAIN"] else "n/a"

        ax.set_yscale("log")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        ax.set_title(f"DAQ{daq} ASIC-{asic} CH{channel}   "
                     f"({entry['N_PEAKS']} pk, gain {gain_text})", fontsize=9)

        print(f"  {i + 1:>4}  DAQ{daq} ASIC-{asic} CH{channel:<6} "
              f"{entry['N_PEAKS']:>5} {entry['LADDER']:>8.1f} "
              f"{gain_text:>7}  "
              f"{entry['ADC_MIN']:>7.0f}-{entry['ADC_MAX']:<8.0f}")

    for j in range(len(selection), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(
        f"TERZINA FPA — Top {len(selection)} channels by resolved p.e. peaks "
        f"({gain})", fontsize=14, y=0.995,
    )
    fig.supxlabel("ADC bin", fontsize=12)
    fig.supylabel("Counts", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))

    print()
    _finish(fig, save_path, show)

    return fig, selection


# ==========================================================================
# Orchestration
# ==========================================================================

def run(
    bin_file: str,
    gain: str = DEFAULT_GAIN,
    top: int = DEFAULT_TOP,
    out_dir: Optional[str] = None,
    min_peaks: int = DEFAULT_MIN_PEAKS,
    n_individual: int = DEFAULT_N_INDIVIDUAL,
    ncols: int = DEFAULT_NCOLS,
    max_packets: Optional[int] = None,
    bad_asics: Sequence[Tuple[int, str]] = DEFAULT_BAD_ASICS,
    show: Optional[bool] = None,
) -> Dict[str, List[dict]]:
    """Parse one run and plot its brightest channels.

    Parameters
    ----------
    bin_file
        Path to the TERZINA ``.bin`` readout file.
    gain
        ``"HG"``, ``"LG"``, or ``"BOTH"``. ``"BOTH"`` parses once and runs
        each branch in turn; parsing dominates the cost, so the second
        branch is nearly free.
    out_dir
        Where figures are written. Defaults to the binary's own directory.
    bad_asics
        ``(DAQ, ASIC)`` pairs to exclude, in binary numbering. Pass an empty
        sequence to keep everything.
    show
        Display figures as well as saving them. Defaults to True inside a
        Jupyter or Spyder kernel, False otherwise.

    Returns
    -------
    dict
        ``{gain: ranked selection}`` for each branch processed.
    """
    gain = str(gain).upper()
    gains = ["HG", "LG"] if gain == "BOTH" else [gain]

    for branch in gains:
        if branch not in ("HG", "LG"):
            raise ValueError(
                f"gain must be 'HG', 'LG' or 'BOTH', got {gain!r}"
            )

    if show is None:
        show = "ipykernel" in sys.modules

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(bin_file))
    os.makedirs(out_dir, exist_ok=True)

    channels, events_per_daq = parse_file(bin_file, max_packets)

    results: Dict[str, List[dict]] = {}

    for branch in gains:
        ranked = rank_channels(
            channels, events_per_daq,
            gain=branch, min_peaks=min_peaks, bad_asics=bad_asics,
        )

        if not ranked:
            print(f"  No {branch} channel resolved at least {min_peaks} "
                  f"peaks — nothing to plot.")
            results[branch] = []
            continue

        _, selection = plot_grid(
            channels, ranked, branch, top=top, ncols=ncols,
            save_path=os.path.join(out_dir, f"bright_fingerplots_{branch}.png"),
            show=show,
        )

        for entry in selection[:n_individual]:
            daq, asic, channel = entry["DAQ"], entry["ASIC"], entry["CH"]
            plot_single(
                channels, daq, asic, channel, branch,
                save_path=os.path.join(
                    out_dir,
                    f"bright_DAQ{daq}_ASIC{asic}_CH{channel}_{branch}.png",
                ),
                show=show,
            )

        results[branch] = selection

    return results


# ==========================================================================
# Command line
# ==========================================================================

def _asic_spec(text: str) -> Tuple[int, str]:
    """Parse a ``DAQ:ASIC`` exclusion such as ``3:D``."""
    try:
        daq_text, asic = text.split(":", 1)
        daq = int(daq_text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected DAQ:ASIC such as 3:D, got {text!r}"
        )

    asic = asic.strip().upper()
    if daq not in DAQS or asic not in ASICS:
        raise argparse.ArgumentTypeError(
            f"DAQ must be 1-4 and ASIC A-E, got {text!r}"
        )
    return daq, asic


def build_parser() -> argparse.ArgumentParser:
    default_excludes = ",".join(f"{d}:{a}" for d, a in DEFAULT_BAD_ASICS)

    parser = argparse.ArgumentParser(
        description="Finger plots of the brightest TERZINA FPA pixels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("bin_file", help="TERZINA .bin readout file")
    parser.add_argument("--gain", default=DEFAULT_GAIN,
                        choices=["HG", "LG", "BOTH"],
                        help="gain branch to analyse")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP,
                        help="channels to show in the grid")
    parser.add_argument("--out", dest="out_dir", default=None,
                        help="output directory "
                             "(default: the binary's own directory)")
    parser.add_argument("--min-peaks", type=int, default=DEFAULT_MIN_PEAKS,
                        help="ignore channels resolving fewer peaks")
    parser.add_argument("--individual", type=int,
                        default=DEFAULT_N_INDIVIDUAL,
                        help="full-size single-channel plots to also save")
    parser.add_argument("--ncols", type=int, default=DEFAULT_NCOLS,
                        help="columns in the grid")
    parser.add_argument("--max-packets", type=int, default=None,
                        help="stop after this many packets (default: all)")
    parser.add_argument("--exclude", type=_asic_spec, action="append",
                        metavar="DAQ:ASIC", default=None,
                        help="exclude an ASIC; repeatable. Overrides the "
                             f"built-in default of {default_excludes}")
    parser.add_argument("--keep-all", action="store_true",
                        help="analyse every ASIC, excluding none")
    parser.add_argument("--show", action="store_true",
                        help="display figures as well as saving them")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.keep_all:
        bad_asics: Sequence[Tuple[int, str]] = ()
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
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
