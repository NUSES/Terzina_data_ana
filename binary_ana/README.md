# TERZINA Binary Data Processing Tool

## Overview

This project provides a complete data analysis pipeline for TERZINA binary acquisition files.

The software reads binary packets produced by TERZINA DAQ systems, decodes detector data, extracts channel information, exports event data into CSV format, computes statistics, and generates diagnostic plots.

The tool supports up to four DAQ modules and automatically produces summary reports and channel occupancy analyses.

---

## Features

### Binary Packet Decoding

The parser reads TERZINA binary packets and extracts:

- Timestamp
- Event identifier
- DAQ information
- Trigger counters
- Channel data
- Lost trigger counters
- Real trigger counters

Supported DAQ blocks:

- DAQ1
- DAQ2
- DAQ3
- DAQ4

---

### Channel Data Extraction

Each channel word is decoded into:

- HG (High Gain ADC)
- LG (Low Gain ADC)
- HIT (Hit flag / trigger information)

Bit structure:

31........28 27........14 13.........0
|   HIT   |     LG     |     HG      |

The decoded values are stored for further analysis and plotting.

---

## Generated Outputs

### Event CSV

A complete CSV file containing all decoded TERZINA packets:

terzina_data_converted.csv

Each row corresponds to a decoded event.

---

### Channel Summary CSV

A summary CSV file is generated containing:

- Channel identifier
- Number of hits
- Mean HG value
- Mean LG value

Example:

DAQ_CODE;HIT;HG_MEAN;LG_MEAN
D1_A_HG_00;152;1380.5;1212.8

---

## Statistical Analysis

The software computes:

- Active channels
- Mean HG values
- Maximum HG values
- Hit occupancy
- Hit distributions

The results are printed directly to the terminal.

---

## Plot Generation

### HG Histograms

Generated for every DAQ and channel group.

Examples:

DAQ1_HG.png
DAQ2_HG.png
DAQ3_HG.png
DAQ4_HG.png

Features:

- One histogram per channel
- Logarithmic Y-axis
- Automatic range selection

---

### LG Histograms

Generated files:

DAQ1_LG.png
DAQ2_LG.png
DAQ3_LG.png
DAQ4_LG.png

---

### HIT Histograms

Generated files:

DAQ1_HIT.png
DAQ2_HIT.png
DAQ3_HIT.png
DAQ4_HIT.png

---

### HIT Occupancy Plots

Generated files:

DAQ1_HIT_OCCUPANCY.png
DAQ2_HIT_OCCUPANCY.png
DAQ3_HIT_OCCUPANCY.png
DAQ4_HIT_OCCUPANCY.png

These plots show the total number of hits recorded by each channel.

---

## Software Structure

### TERZINApack

Dataclass representing a complete TERZINA event.

Contains:

- Event metadata
- DAQ information
- Channel arrays
- Trigger counters

---

### DataProcessing

Main processing class responsible for:

- Binary file reading
- Packet parsing
- Data decoding
- Channel extraction
- CSV generation
- Statistical analysis
- Plot creation

---

## Requirements

Python 3.9 or newer.

Required package:

pip install matplotlib

Used standard libraries:

- os
- csv
- dataclasses
- typing

---

## Usage

Set the input file path inside the constructor:

self.file_path = "data_xx_xx/data_xx_xx.bin"

Run:

python data_analyzer_pro.py

---

## Output Structure

data_xx_xx/

├── data_xx_xx.bin
├── terzina_data_converted.csv
├── channel_summary.csv
└── plots/
    ├── DAQ1_HG.png
    ├── DAQ1_LG.png
    ├── DAQ1_HIT.png
    ├── DAQ1_HIT_OCCUPANCY.png
    └── ...

---

## Processing Workflow

Binary File
    ↓
Packet Parsing
    ↓
DAQ Decoding
    ↓
Channel Extraction
    ↓
CSV Export
    ↓
Statistics
    ↓
Plot Generation

---

## Future Improvements

Potential future developments include:

- ROOT file export
- Gaussian fitting of HG/LG spectra
- Automatic bad channel detection
- Trigger efficiency studies
- Batch processing of multiple files
- Interactive analysis dashboards

---

