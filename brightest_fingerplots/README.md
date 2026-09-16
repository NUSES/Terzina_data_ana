# TERZINA peak fingerplots

This script scans the 640 TERZINA readout channels, counts the resolved
fingerplot peaks, and saves a grid containing the channels with the largest
number of detected peaks. HG and LG are analysed independently.The ranking is based on the number of detected peaks. 

## Installation

Use Python 3.9 or newer and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Command-line usage

Run both gain paths and save the results in `results/`:

```bash
python terzina_peak_fingerplots.py path/to/run.bin --gain BOTH --top 20
```

Examples:

```bash
python terzina_peak_fingerplots.py path/to/run.bin --gain HG --top 16
python terzina_peak_fingerplots.py path/to/run.bin --gain LG --top 16 --output-dir plots
python terzina_peak_fingerplots.py path/to/run.bin --gain BOTH --top 20 --show
```

On Windows, quote paths that contain spaces:

```powershell
python terzina_peak_fingerplots.py "C:\path\to\run.bin" --gain BOTH --top 20
```

The default output files are:

```text
results/fingerplots_most_peaks_HG.png
results/fingerplots_most_peaks_LG.png
```

The input path is deliberately supplied at runtime. It is not stored in the
source file.

## Jupyter usage

```python
from terzina_peak_fingerplots import main

results = main(
    "data/run.bin",
    gain="BOTH",
    n_top=20,
    output_dir="results",
    show_plots=True,
)
```

For a Windows path:

```python
results = main(
    r"C:\path\to\run.bin",
    gain="BOTH",
    n_top=20,
    output_dir="results",
)
```

## Optional settings

The command-line interface also supports:

```text
--min-peaks N       ignore channels with fewer than N peaks; default: 3
--max-packets N     parse only the first N valid packets
--ncols N           number of panels per row; default: 4
--exclude-asic 3:D exclude an ASIC; repeat the option when needed
```

All DAQs and ASICs are included by default. No ASIC is silently removed.
