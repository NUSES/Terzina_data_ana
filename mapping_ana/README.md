# Terzina FPA Mapping and Camera Visualization

This project reconstructs the full pixel-level geometry of Terzina SiPM-based camera starting from hardware pin-to-pin mapping tables and DAQ-level data. It also provides visualization tools for detector occupancy and signal distributions.

## Overview

The pipeline:
1. Loads quadrant-based pin-to-pin mapping CSV files
2. Parses DAQ signals into physical detector identifiers
3. Builds a complete dataset of tiles, channels, and global IDs
4. Computes pixel-level geometry in millimeters
5. Assigns ASIC identifiers based on detector topology
6. Generates a full mapping file
7. Visualizes detector response maps

## Input Data

Quadrant mapping files:
finalmappingsTERZINAGlobalIDmapping(23_03_2026)-QUADRANT0.csv  
finalmappingsTERZINAGlobalIDmapping(23_03_2026)-QUADRANT1.csv  
finalmappingsTERZINAGlobalIDmapping(23_03_2026)-QUADRANT2.csv  
finalmappingsTERZINAGlobalIDmapping(23_03_2026)-QUADRANT3.csv  

Each file contains:
- TILE ID
- signal name
- daq_signal
- GLOBAL ID

Configuration file:
TerzinaFPA.conf

Defines:
- nx_tiles, ny_tiles
- SiPM pixel sizes
- pitch values
- mechanical offsets

DAQ summary file (example):
DAQ_CODE;HIT;HG_MEAN;LG_MEAN

Example rows:
D1_C_HG_21;12;34.5;10.2  
D2_A_HG_03;5;20.1;8.3  

## Pipeline

Step 1: Load mapping  
Quadrant CSVs are merged into a single DataFrame.

Step 2: Parse DAQ signals  
Example format:
D1_C_HG_21  

Converted into:
- Quadrant index
- ASIC identifier
- Input channel

Step 3: Build complete dataset  
Extract:
- TILE ID
- CHANNEL ID
- SIGNAL NAME
- GLOBAL ID
- ASIC ID

Step 4: Geometry reconstruction  
Each tile/channel pair is converted into:
- Pixel X position (mm)
- Pixel Y position (mm)
- Global pixel ID

Step 5: Output mapping  
complete_mapping_{nx}x{ny}.csv

Step 6: Visualization  
Function:
plot_camera_map(csv_file, value_column, title)

Produces:
- HIT occupancy map
- HG_MEAN map
- LG_MEAN map

## Requirements

numpy  
pandas  
matplotlib  

Install:
pip install numpy pandas matplotlib

## Run

python Mapping_HitMap.py

To generate plots:
plot_camera_map("channel_summary.csv", "HIT", "Camera HIT Occupancy")  
plot_camera_map("channel_summary.csv", "HG_MEAN", "Camera HG Mean")  
plot_camera_map("channel_summary.csv", "LG_MEAN", "Camera LG Mean")  

## Notes

- Geometry assumes staggered SiPM tile layout
- ASIC assignment depends on tile parity and channel position
- DAQ naming must follow: D{quadrant}_{ASIC}_{type}_{channel}
- Pixel indexing is row-wise inside tiles
