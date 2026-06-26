from typing import Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import random
import colorsys
import re
# =========================================================
# AUTHORS AND DATA SOURCES
# =========================================================
#
# S. Davarpanah
#   - Original implementation using PDF extraction (Camelot)
#   - Initial parsing of legacy ICD tables from PDF document
#   - First version of mapping pipeline
#
# C. Trimarelli
#   - Re-implementation based on CSV pin-to-pin extraction
#   - Debugging and correction of ID assignment logic and typos experienced in the first version
#   - Fixed TILE / ASIC interpretation inconsistencies experienced in the first version
#   - Final agreed mapping structure and output format
#
# A. Palmieri
#   - TerzinaFPA.conf configuration file and fpa conf in  simulation chain 
#
# H. Lima
#   - Provides reference CSV tables used for pin-to-pin mapping
#   - Source of original electronics connectivity datasets
#
# =========================================================

def filter_from_daq(pinmap, daq_signal):

    parts = daq_signal.strip().split("_")

    D_part = parts[0].strip()   # D1
    ASIC = parts[1].strip()     # C
    IN = str(int(parts[3].strip()))
    quadrant = int(D_part[1:]) - 1

    in_signal = f"In{ASIC}{IN}"

    df = pinmap.copy()

    # pulizia totale colonne
    df["daq_signal"] = df["daq_signal"].astype(str).str.strip()
    df["QUADRANT"] = pd.to_numeric(df["QUADRANT"], errors="coerce")

    result = df[
        (df["QUADRANT"] == quadrant) &
        (df["daq_signal"] == in_signal)
    ]

    return result

# =========================================================
# 1. CSV LOADER (PIN-TO-PIN)
# =========================================================

def load_quadrant_csv(path):
    df_raw = pd.read_csv(path, header=None, dtype=str, encoding="latin1")

    header_idx = df_raw.apply(
        lambda r: r.astype(str).str.contains("QUADRANT", case=False, na=False)
    ).any(axis=1).idxmax()

    df = df_raw.iloc[header_idx + 1:].reset_index(drop=True)

    rows = []
    block_size = 10

    for _, row in df.iterrows():
        row = row.fillna("")

        for i in range(0, len(row), block_size + 1):
            block = row[i:i + block_size]

            if len(block) < block_size:
                continue
            if block.iloc[0] == "":
                continue

            rows.append(block.tolist())

    columns = [
        "QUADRANT",
        "TILE",
        "10TA.J#",
        "pin of 10TA.J",
        "Hierachical signal name",
        "signal name",
        "CB.J",
        "pin of CB.J",
        "daq_signal",
        "GLOBAL ID"
    ]

    return pd.DataFrame(rows, columns=columns)


# ---- INPUT FILES ----
files = [
    "finalmappingsTERZINAGlobalIDmapping(23_03_2026)-QUADRANT0.csv",
    "finalmappingsTERZINAGlobalIDmapping(23_03_2026)-QUADRANT1.csv",
    "finalmappingsTERZINAGlobalIDmapping(23_03_2026)-QUADRANT2.csv",
    "finalmappingsTERZINAGlobalIDmapping(23_03_2026)-QUADRANT3.csv"
]

# =========================================================
# 2. CREATE PIN-TO-PIN OUTPUT
# =========================================================

pinmap = pd.concat([load_quadrant_csv(f) for f in files], ignore_index=True)
#print(
#    pinmap[["QUADRANT","daq_signal", "signal name"]]
#    .head(20)
#    .to_string(index=False)
#)
#cols = [
#    "daq_signal",
#    "signal name",
#    "GLOBAL ID",
#    "TILE"
#]

#print(
#    pinmap[
#        pinmap["daq_signal"] == "InC21"
#    ][cols].to_string(index=False)
#)
# =========================================================
# ESEMPIO CSV INPUT
# =========================================================

# esempio: hits.csv
#
# DAQ,HIT
# D1_C_HG_21,12
# D1_C_HG_22,5
# D2_A_HG_03,30
# D3_D_HG_15,50
# D4_E_HG_08,90
#

#hits_df = pd.read_csv("example_hits.csv")

#print(hits_df)
#print(
#    filter_from_daq(pinmap, "D1_C_HG_21")
#    .to_string(index=False)
#)

#data_NI = filter_from_daq(pinmap, "D1_C_HG_21")
#print(data_NI)
#pinmap.to_excel(
#    "./terzina_pin_to_pin_connections.xlsx",
#    index=False
#)
#data_NI = filter_from_daq(pinmap, "D1_C_HG_21").copy()

#col = data_NI["signal name"].astype("string").str.strip()
#parsed = col.str.extract(r'^([A-F])(\d+)_CH(\d+)$')

#valid_mask = parsed[0].notna()

#ni_valid = data_NI.loc[valid_mask].copy()
#parsed_valid = parsed.loc[valid_mask].copy()

#mini = pd.DataFrame()

#mini["QUADRANT"] = ni_valid["QUADRANT"].astype(int).values
#mini["TILE #"] = pd.to_numeric(ni_valid["TILE"], errors="coerce").values
#mini["CHANNEL #"] = pd.to_numeric(parsed_valid[2], errors="coerce").values


# =========================================================
# 3. CONFIG
# =========================================================

def read_config(filename):
    config = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue

            if ':' in line:
                k, v = line.split(':', 1)
                k = k.strip()
                v = v.strip()

                try:
                    v = float(v) if '.' in v else int(v)
                except:
                    pass

                config[k] = v
    return config


conf_path = "./TerzinaFPA.conf"
config = read_config(conf_path)

# =========================================================
# 4. BUILD COMPLETE DATA (MISSING PART ADDED)
# =========================================================

col = pinmap["signal name"].astype("string").str.strip()
parsed = col.str.extract(r'^([A-F])(\d+)_CH(\d+)$')

valid_mask = parsed[0].notna()

pin_valid = pinmap.loc[valid_mask].copy()
parsed_valid = parsed.loc[valid_mask].copy()

complete_data = pd.DataFrame()

complete_data["QUADRANT"] = pin_valid["QUADRANT"].reset_index(drop=True)
complete_data["TILE #"] = pd.to_numeric(pin_valid["TILE"], errors="coerce").reset_index(drop=True)
complete_data["CHANNEL #"] = pd.to_numeric(parsed_valid[2], errors="coerce").reset_index(drop=True)

complete_data["signal name"] = pin_valid["signal name"].reset_index(drop=True)
complete_data["GLOBAL ID"] = pin_valid["GLOBAL ID"].reset_index(drop=True)
# -----------------------------
def get_asic(daq, tile, chid):
    """
    Return ASIC number based on DAQ, tile, and channel id.
    
    side = 0 for channels 0–31
    side = 1 for channels 32–63
    """

    side = 0 if chid < 32 else 1
    t = tile % 5

    if side == 0:
        pattern = [0, 2, 4, 2, 0]
    else:
        pattern = [1, 3, 4, 3, 1]

    return pattern[t]


complete_data["ASIC #"] = complete_data.apply(
    lambda row: get_asic(int(row["QUADRANT"]), int(row["TILE #"]), int(row["CHANNEL #"])),
    axis=1
)

# =========================================================
# 5. GEOMETRY FUNCTION (UNCHANGED)
# =========================================================

def tile_ch_to_pos_and_global(tile_ID, ch_local_ID, config):
    nx_tiles = config['nx_tiles']
    ny_tiles = config['ny_tiles']
    Nx_tot = config['nx_sipm_pixel']
    Ny_tot = config['ny_sipm_pixel']

    tile_row_id = ny_tiles - tile_ID // nx_tiles - 1

    if tile_row_id % 2 == 0:
        tile_col_id = nx_tiles - tile_ID % nx_tiles - 1
    else:
        tile_col_id = tile_ID % nx_tiles

    ch_local_ID = int(ch_local_ID)

    if tile_row_id % 2 == 0:
        ch_local_ID = Nx_tot * Ny_tot - 1 - ch_local_ID

    dTx = (
        Nx_tot * config['sensitive_sipm_pixel_sizeX']
        + (Nx_tot - 1) * config['sipm_pixel_pitch_right']
        + 2 * config['sipm_array_d']
    )

    dTy = (
        Ny_tot * config['sensitive_sipm_pixel_sizeY']
        + int(Ny_tot / 2) * config['sipm_pixel_pitch_down']
        + int((Ny_tot - 1) / 2) * config['sipm_pixel_pitch_up']
        + 2 * config['sipm_array_d']
    )

    tile_posX = (
        - nx_tiles * dTx / 2
        - (nx_tiles - 1) * config['sipm_array_d'] / 2
        + dTx / 2
        + (dTx + config['sipm_array_d']) * tile_col_id
    )

    tile_posY = (
        - ny_tiles * dTy / 2
        - (ny_tiles - 1) * config['sipm_array_d'] / 2
        + dTy / 2
        + (dTy + config['sipm_array_d']) * tile_row_id
    )

    pixel_local_col = ch_local_ID // Nx_tot
    pixel_local_row = Ny_tot - 1 - (ch_local_ID % Nx_tot)

    pixel_posX = tile_posX + (
        - Nx_tot * config['sensitive_sipm_pixel_sizeX'] / 2
        - (Nx_tot - 1) * config['sipm_pixel_pitch_right'] / 2
        + config['sensitive_sipm_pixel_sizeX'] / 2
        + (config['sensitive_sipm_pixel_sizeX'] + config['sipm_pixel_pitch_right']) * pixel_local_col
    )

    pixel_posY = tile_posY + (
        - Ny_tot * config['sensitive_sipm_pixel_sizeY'] / 2
        - int(Ny_tot / 2) * config['sipm_pixel_pitch_down'] / 2
        - int((Ny_tot - 1) / 2) * config['sipm_pixel_pitch_up'] / 2
        + (2 * pixel_local_row + 1) * config['sensitive_sipm_pixel_sizeY'] / 2
        + config['sipm_pixel_pitch_down'] * int((pixel_local_row + 1) / 2)
        + config['sipm_pixel_pitch_up'] * int(pixel_local_row / 2)
    )

    ncols_total = nx_tiles * Nx_tot
    global_col = tile_col_id * Nx_tot + pixel_local_col
    global_row = tile_row_id * Ny_tot + pixel_local_row
    globalID = global_row * ncols_total + global_col

    return round(pixel_posX,2), round(pixel_posY,2), int(globalID), round(tile_posX,2), round(tile_posY,2)

# =========================================================
# 6. APPLY GEOMETRY
# =========================================================

results = complete_data.apply(
    lambda r: tile_ch_to_pos_and_global(
        int(r["TILE #"]),
        int(r["CHANNEL #"]),
        config
    ),
    axis=1,
    result_type="expand"
)

results.columns = [
    "PixelCentrePosX_mm",
    "PixelCentrePosY_mm",
    "GlobalID",
    "TileCentrePosX_mm",
    "TileCentrePosY_mm"
]

complete_data = pd.concat([complete_data, results], axis=1)






# =========================================================
# 7. SAVE COMPLETE MAPPING (ADDED FIX)
# =========================================================

nx = config['nx_tiles']
ny = config['ny_tiles']

complete_data.to_csv(
    f"complete_mapping_{nx}x{ny}.csv",
    index=False,
    float_format="%.2f"
)

# =========================================================
# 8. COLOR UTILS
# =========================================================

def generate_distinct_dark_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.75)
        colors.append('#{:02x}{:02x}{:02x}'.format(
            int(r*255), int(g*255), int(b*255)
        ))
    random.shuffle(colors)
    return colors

color_daq = generate_distinct_dark_colors(
    int(config['nx_tiles'] * config['ny_tiles'] * 2 / 5)
)

Asic_daq = {
    0: 'floralwhite',
    1: 'gold',
    2: 'azure',
    3: 'paleturquoise',
    4: 'moccasin'
}

asic_letters = ['a', 'b', 'c', 'd', 'e']



# =========================================================
# GENERIC CAMERA MAP PLOT
# =========================================================

def plot_camera_map(csv_file, value_column, title):

    summary = pd.read_csv(csv_file, sep=";")

    # --------------------------------------------
    # BUILD MINI DATAFRAME
    # --------------------------------------------

    rows = []

    for _, row in summary.iterrows():

        daq_code = row["DAQ_CODE"]
        value = row[value_column]


        if value_column == "HIT" and value != 0:
            print("DAQ_CODE =", daq_code)
            print("HIT =", value)

        if value < 0:
            continue

        data_NI = filter_from_daq(pinmap, daq_code)

        if len(data_NI) == 0:
            continue

        col = data_NI["signal name"].astype("string").str.strip()

        parsed = col.str.extract(
            r'^([A-F])(\d+)_CH(\d+)$'
        )

        valid_mask = parsed[0].notna()

        ni_valid = data_NI.loc[valid_mask].copy()
        parsed_valid = parsed.loc[valid_mask].copy()

        tmp = pd.DataFrame()

        tmp["QUADRANT"] = (
            ni_valid["QUADRANT"]
            .astype(int)
            .values
        )

        tmp["TILE #"] = pd.to_numeric(
            ni_valid["TILE"],
            errors="coerce"
        ).values

        tmp["CHANNEL #"] = pd.to_numeric(
            parsed_valid[2],
            errors="coerce"
        ).values

        tmp["VALUE"] = value
        if not tmp.empty:
            rows.append(tmp)
    if not rows:
        print(f"Nessun dato valido per {value_column}")
        return
    mini = pd.concat(rows, ignore_index=True)

    # --------------------------------------------
    # GEOMETRY
    # --------------------------------------------

    res = mini.apply(
        lambda r: tile_ch_to_pos_and_global(
            int(r["TILE #"]),
            int(r["CHANNEL #"]),
            config
        ),
        axis=1,
        result_type="expand"
    )

    res.columns = [
        "PixelCentrePosX_mm",
        "PixelCentrePosY_mm",
        "GlobalID",
        "TileCentrePosX_mm",
        "TileCentrePosY_mm"
    ]

    mini = pd.concat([mini, res], axis=1)

    # --------------------------------------------
    # PLOT
    # --------------------------------------------

    fig, ax = plt.subplots(
        figsize=(18, 18),
        dpi=300
    )

    norm = plt.Normalize(
        vmin=mini["VALUE"].min(),
        vmax=mini["VALUE"].max()
    )

    cmap = plt.cm.jet

    # --------------------------------------------
    # DRAW EMPTY PIXELS
    # --------------------------------------------

    for _, row in complete_data.iterrows():

        posx = row["PixelCentrePosX_mm"]
        posy = row["PixelCentrePosY_mm"]

        rect = patches.Rectangle(
            (
                posx - config['sensitive_sipm_pixel_sizeX'] / 2,
                posy - config['sensitive_sipm_pixel_sizeY'] / 2
            ),
            config['sensitive_sipm_pixel_sizeX'],
            config['sensitive_sipm_pixel_sizeY'],

            facecolor='whitesmoke',
            edgecolor='lightgray',
            linewidth=0.25
        )

        ax.add_patch(rect)

    # --------------------------------------------
    # DRAW ACTIVE PIXELS
    # --------------------------------------------

    for _, row in mini.iterrows():

        posx = row["PixelCentrePosX_mm"]
        posy = row["PixelCentrePosY_mm"]

        value = row["VALUE"]

        color = cmap(norm(value))

        rect = patches.Rectangle(
            (
                posx - config['sensitive_sipm_pixel_sizeX'] / 2,
                posy - config['sensitive_sipm_pixel_sizeY'] / 2
            ),
            config['sensitive_sipm_pixel_sizeX'],
            config['sensitive_sipm_pixel_sizeY'],

            facecolor=color,
            edgecolor='black',
            linewidth=0.4
        )

        ax.add_patch(rect)

    # --------------------------------------------
    # COLORBAR
    # --------------------------------------------

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=norm
    )

    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax)

    cbar.set_label(
        value_column,
        fontsize=14
    )

    # --------------------------------------------
    # AXES
    # --------------------------------------------

    ax.set_aspect('equal')

    xlim = np.max(
        complete_data["PixelCentrePosX_mm"]
    ) + 5

    ylim = np.max(
        complete_data["PixelCentrePosY_mm"]
    ) + 5

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    ax.set_title(title)

    plt.grid(
        True,
        linestyle='--',
        linewidth=0.2
    )

    plt.show()

# =========================================================
# GENERATE ALL MAPS
# =========================================================

csv_file = "/home/caterina/Documenti/mapping_ICD/mapping_code/data_05_13-17_30/channel_summary_17_30.csv"

plot_camera_map(
    csv_file,
    "HIT",
    "Camera HIT Occupancy"
)

plot_camera_map(
    csv_file,
    "HG_MEAN",
    "Camera HG Mean"
)

plot_camera_map(
    csv_file,
    "LG_MEAN",
    "Camera LG Mean"
)
