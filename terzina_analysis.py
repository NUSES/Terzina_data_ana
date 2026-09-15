"""
============================================================
TERZINA CHANNEL SUMMARY + CAMERA MAP
============================================================

Usage:

    python terzina_analysis.py data_07_09-16_23

Expected folder:

    data_07_09-16_23/
        data_07_09-16_23.bin

Output:

    data_07_09-16_23/
        channel_summary_data_07_09-16_23.csv
        plots/
            Camera_HIT_Occupancy.png
            Camera_HG_Mean.png
            Camera_LG_Mean.png

============================================================
"""

import os
import csv
import argparse
import time
import random
import colorsys

from dataclasses import dataclass, field

from typing import List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patches


# ============================================================
# DATACLASS
# ============================================================

@dataclass
class TERZINApack:

    TIMESTAMP: int = 0
    ID_CONC: int = 0

    # ===================== DAQ1 =====================

    DAQ1_ID: int = 0
    DAQ1TRIG_counter: int = 0
    DAQ1_data1: int = 0
    DAQ1_data2: int = 0

    D1_A: List[int] = field(default_factory=list)
    D1_B: List[int] = field(default_factory=list)
    D1_C: List[int] = field(default_factory=list)
    D1_D: List[int] = field(default_factory=list)
    D1_E: List[int] = field(default_factory=list)

    D1_LOST: int = 0
    D1_REAL: int = 0

    # ===================== DAQ2 =====================

    DAQ2_ID: int = 0
    DAQ2TRIG_counter: int = 0
    DAQ2_data1: int = 0
    DAQ2_data2: int = 0

    D2_A: List[int] = field(default_factory=list)
    D2_B: List[int] = field(default_factory=list)
    D2_C: List[int] = field(default_factory=list)
    D2_D: List[int] = field(default_factory=list)
    D2_E: List[int] = field(default_factory=list)

    D2_LOST: int = 0
    D2_REAL: int = 0

    # ===================== DAQ3 =====================

    DAQ3_ID: int = 0
    DAQ3TRIG_counter: int = 0
    DAQ3_data1: int = 0
    DAQ3_data2: int = 0

    D3_A: List[int] = field(default_factory=list)
    D3_B: List[int] = field(default_factory=list)
    D3_C: List[int] = field(default_factory=list)
    D3_D: List[int] = field(default_factory=list)
    D3_E: List[int] = field(default_factory=list)

    D3_LOST: int = 0
    D3_REAL: int = 0

    # ===================== DAQ4 =====================

    DAQ4_ID: int = 0
    DAQ4TRIG_counter: int = 0
    DAQ4_data1: int = 0
    DAQ4_data2: int = 0

    D4_A: List[int] = field(default_factory=list)
    D4_B: List[int] = field(default_factory=list)
    D4_C: List[int] = field(default_factory=list)
    D4_D: List[int] = field(default_factory=list)
    D4_E: List[int] = field(default_factory=list)

    D4_LOST: int = 0
    D4_REAL: int = 0


# ============================================================
# DATA PROCESSING
# ============================================================

class DataProcessing:

    def __init__(self, input_folder):

        self.output_dir = os.path.abspath(input_folder)

        if not os.path.isdir(self.output_dir):
            raise FileNotFoundError(
                f"Folder not found: {self.output_dir}"
            )

        # ----------------------------------------------------
        # BIN FILE
        # ----------------------------------------------------

        folder_name = os.path.basename(
            os.path.normpath(self.output_dir)
        )

        self.file_path = os.path.join(
            self.output_dir,
            f"{folder_name}.bin"
        )

        if not os.path.exists(self.file_path):

            raise FileNotFoundError(
                f"\nBIN file not found:\n"
                f"{self.file_path}\n"
                f"\nExpected filename:\n"
                f"{folder_name}.bin"
            )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        self.plot_dir = os.path.join(
            self.output_dir,
            "plots"
        )

        os.makedirs(
            self.plot_dir,
            exist_ok=True
        )

        self.csv_file = os.path.join(
            self.output_dir,
            f"channel_summary_{folder_name}.csv"
        )

        # ----------------------------------------------------
        # FILE INFO
        # ----------------------------------------------------

        self.datalen = os.path.getsize(
            self.file_path
        )

        # ----------------------------------------------------
        # CHANNEL STORAGE
        # ----------------------------------------------------

        self.channels = {}

        prefixes = [
            "D1_A", "D1_B", "D1_C", "D1_D", "D1_E",
            "D2_A", "D2_B", "D2_C", "D2_D", "D2_E",
            "D3_A", "D3_B", "D3_C", "D3_D", "D3_E",
            "D4_A", "D4_B", "D4_C", "D4_D", "D4_E"
        ]

        for prefix in prefixes:

            self.channels[f"{prefix}_HG"] = []
            self.channels[f"{prefix}_LG"] = []
            self.channels[f"{prefix}_HIT"] = []

        self.first_timestamp = None
        self.last_timestamp = None

        self.n_packs = 0

    # ========================================================
    # EXTRACT CHANNELS
    # ========================================================

    def extract_channels(self, channel_list):

        HG_list = []
        LG_list = []
        HIT_list = []

        for w in channel_list:

            HG = (0x3FFF - w) & 0x3FFF
            LG = (0x3FFF - (w >> 14)) & 0x3FFF
            HIT = (w >> 28) & 0xF

            HG_list.append(HG)
            LG_list.append(LG)
            HIT_list.append(HIT)

        return HG_list, LG_list, HIT_list

    # ========================================================
    # UPDATE CHANNEL DATA
    # ========================================================

    def update_channel_data(self, pack):

        channels = [

            ("D1_A", pack.D1_A),
            ("D1_B", pack.D1_B),
            ("D1_C", pack.D1_C),
            ("D1_D", pack.D1_D),
            ("D1_E", pack.D1_E),

            ("D2_A", pack.D2_A),
            ("D2_B", pack.D2_B),
            ("D2_C", pack.D2_C),
            ("D2_D", pack.D2_D),
            ("D2_E", pack.D2_E),

            ("D3_A", pack.D3_A),
            ("D3_B", pack.D3_B),
            ("D3_C", pack.D3_C),
            ("D3_D", pack.D3_D),
            ("D3_E", pack.D3_E),

            ("D4_A", pack.D4_A),
            ("D4_B", pack.D4_B),
            ("D4_C", pack.D4_C),
            ("D4_D", pack.D4_D),
            ("D4_E", pack.D4_E),
        ]

        for name, chan_list in channels:

            if len(chan_list) == 0:
                continue

            HG, LG, HIT = self.extract_channels(
                chan_list
            )

            self.channels[f"{name}_HG"].extend(HG)
            self.channels[f"{name}_LG"].extend(LG)
            self.channels[f"{name}_HIT"].extend(HIT)

    # ========================================================
    # PARSE PAYLOAD
    # ========================================================

    def parse_terzina_payload(self, payload: bytes):

        f = TERZINApack()

        offset = 0

        def u32():

            nonlocal offset

            if offset + 4 > len(payload):
                raise ValueError(
                    "Unexpected end of payload while reading u32"
                )

            val = int.from_bytes(
                payload[offset:offset + 4],
                "little"
            )

            offset += 4

            return val

        def u64():

            nonlocal offset

            if offset + 8 > len(payload):
                raise ValueError(
                    "Unexpected end of payload while reading u64"
                )

            val = int.from_bytes(
                payload[offset:offset + 8],
                "little"
            )

            offset += 8

            return val

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        f.TIMESTAMP = u64()

        if self.first_timestamp is None:
            self.first_timestamp = f.TIMESTAMP

        self.last_timestamp = f.TIMESTAMP

        f.ID_CONC = u64()

        if f.ID_CONC != 0xCA00FE11DD651983:
            return f

        # ----------------------------------------------------
        # DAQ DATA
        # ----------------------------------------------------

        while offset < len(payload):

            daq_id = u32()

            # =================================================
            # DAQ1
            # =================================================

            if daq_id == 0x00AAAA00:

                f.DAQ1_ID = daq_id

                f.DAQ1TRIG_counter = u32()
                f.DAQ1_data1 = u32()
                f.DAQ1_data2 = u32()

                f.D1_A = [u32() for _ in range(32)]
                f.D1_B = [u32() for _ in range(32)]
                f.D1_C = [u32() for _ in range(32)]
                f.D1_D = [u32() for _ in range(32)]
                f.D1_E = [u32() for _ in range(32)]

                f.D1_LOST = u64()
                f.D1_REAL = u64()

            # =================================================
            # DAQ2
            # =================================================

            elif daq_id == 0x00BBBB00:

                f.DAQ2_ID = daq_id

                f.DAQ2TRIG_counter = u32()
                f.DAQ2_data1 = u32()
                f.DAQ2_data2 = u32()

                f.D2_A = [u32() for _ in range(32)]
                f.D2_B = [u32() for _ in range(32)]
                f.D2_C = [u32() for _ in range(32)]
                f.D2_D = [u32() for _ in range(32)]
                f.D2_E = [u32() for _ in range(32)]

                f.D2_LOST = u64()
                f.D2_REAL = u64()

            # =================================================
            # DAQ3
            # =================================================

            elif daq_id == 0x00CCCC00:

                f.DAQ3_ID = daq_id

                f.DAQ3TRIG_counter = u32()
                f.DAQ3_data1 = u32()
                f.DAQ3_data2 = u32()

                f.D3_A = [u32() for _ in range(32)]
                f.D3_B = [u32() for _ in range(32)]
                f.D3_C = [u32() for _ in range(32)]
                f.D3_D = [u32() for _ in range(32)]
                f.D3_E = [u32() for _ in range(32)]

                f.D3_LOST = u64()
                f.D3_REAL = u64()

            # =================================================
            # DAQ4
            # =================================================

            elif daq_id == 0x00DDDD00:

                f.DAQ4_ID = daq_id

                f.DAQ4TRIG_counter = u32()
                f.DAQ4_data1 = u32()
                f.DAQ4_data2 = u32()

                f.D4_A = [u32() for _ in range(32)]
                f.D4_B = [u32() for _ in range(32)]
                f.D4_C = [u32() for _ in range(32)]
                f.D4_D = [u32() for _ in range(32)]
                f.D4_E = [u32() for _ in range(32)]

                f.D4_LOST = u64()
                f.D4_REAL = u64()

            # =================================================
            # END
            # =================================================

            elif daq_id == 0xDEADBEEF:

                break

            else:

                print(
                    f"UNKNOWN DAQ ID : {hex(daq_id)}"
                )

                break

        return f

    # ========================================================
    # PROCESS BINARY FILE
    # ========================================================

    def process_pck(self):

        print()
        print("===================================")
        print("PROCESSING BINARY FILE")
        print("===================================")
        print(f"INPUT : {self.file_path}")
        print(f"SIZE  : {self.datalen:,} bytes")
        print()

        start_time = time.perf_counter()

        with open(self.file_path, "rb") as file:

            while True:

                raw = file.read(4)

                if len(raw) < 4:
                    break

                ID_pack = int.from_bytes(
                    raw,
                    "little"
                )

                raw_len = file.read(4)

                if len(raw_len) < 4:
                    break

                len_pack = int.from_bytes(
                    raw_len,
                    "little"
                )

                raw_counter = file.read(4)

                if len(raw_counter) < 4:
                    break

                pack_counter = int.from_bytes(
                    raw_counter,
                    "little"
                )

                if ID_pack == 0xA77A:

                    payload = file.read(len_pack)

                    if len(payload) != len_pack:

                        print(
                            "ERROR: TRUNCATED PACK"
                        )

                        break

                    try:

                        terzina_pack = (
                            self.parse_terzina_payload(
                                payload
                            )
                        )

                        self.update_channel_data(
                            terzina_pack
                        )

                        self.n_packs += 1

                    except Exception as e:

                        print(
                            f"ERROR PACK {pack_counter}: {e}"
                        )

                else:

                    print(
                        f"INVALID PACK ID : "
                        f"{hex(ID_pack)}"
                    )

        elapsed = time.perf_counter() - start_time

        print()
        print("PROCESSING FINISHED")
        print(f"PACKETS : {self.n_packs}")
        print(f"TIME    : {elapsed:.3f} s")

        return elapsed

    # ========================================================
    # EXPORT CHANNEL SUMMARY
    # ========================================================

    def export_channel_summary(self):

        print()
        print("===================================")
        print("CREATING CHANNEL SUMMARY")
        print("===================================")

        start_time = time.perf_counter()

        prefixes = [

            "D1_A", "D1_B", "D1_C", "D1_D", "D1_E",

            "D2_A", "D2_B", "D2_C", "D2_D", "D2_E",

            "D3_A", "D3_B", "D3_C", "D3_D", "D3_E",

            "D4_A", "D4_B", "D4_C", "D4_D", "D4_E"
        ]

        with open(
            self.csv_file,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(
                file,
                delimiter=";"
            )

            writer.writerow([
                "DAQ_CODE",
                "HIT",
                "HG_MEAN",
                "LG_MEAN"
            ])

            for prefix in prefixes:

                HG_values = self.channels[
                    f"{prefix}_HG"
                ]

                LG_values = self.channels[
                    f"{prefix}_LG"
                ]

                HIT_values = self.channels[
                    f"{prefix}_HIT"
                ]

                # --------------------------------------------
                # 32 CHANNELS
                # --------------------------------------------

                for ch in range(32):

                    hg_ch = HG_values[ch::32]
                    lg_ch = LG_values[ch::32]
                    hit_ch = HIT_values[ch::32]

                    if len(hg_ch) == 0:
                        continue

                    # ----------------------------------------
                    # HG
                    # ----------------------------------------

                    hg_valid = [
                        v for v in hg_ch
                        if v > 0
                    ]

                    if len(hg_valid) > 0:

                        hg_mean = (
                            sum(hg_valid)
                            / len(hg_valid)
                        )

                    else:

                        hg_mean = 0

                    # ----------------------------------------
                    # LG
                    # ----------------------------------------

                    lg_valid = [
                        v for v in lg_ch
                        if v > 0
                    ]

                    if len(lg_valid) > 0:

                        lg_mean = (
                            sum(lg_valid)
                            / len(lg_valid)
                        )

                    else:

                        lg_mean = 0

                    # ----------------------------------------
                    # HIT
                    # ----------------------------------------

                    hit_count = sum(
                        1
                        for v in hit_ch
                        if v > 0
                    )

                    daq_code = (
                        f"{prefix}_HG_{ch:02d}"
                    )

                    writer.writerow([
                        daq_code,
                        hit_count,
                        round(hg_mean, 2),
                        round(lg_mean, 2)
                    ])

        elapsed = time.perf_counter() - start_time

        print(
            f"SAVED : {self.csv_file}"
        )

        print(
            f"SUMMARY TIME : {elapsed:.3f} s"
        )

        return elapsed


# ============================================================
# PIN-TO-PIN MAPPING
# ============================================================

def filter_from_daq(pinmap, daq_signal):

    parts = daq_signal.strip().split("_")

    D_part = parts[0].strip()
    ASIC = parts[1].strip()
    IN = str(int(parts[3].strip()))

    quadrant = int(D_part[1:]) - 1

    in_signal = f"In{ASIC}{IN}"

    df = pinmap.copy()

    df["daq_signal"] = (
        df["daq_signal"]
        .astype(str)
        .str.strip()
    )

    df["QUADRANT"] = pd.to_numeric(
        df["QUADRANT"],
        errors="coerce"
    )

    result = df[
        (df["QUADRANT"] == quadrant)
        &
        (df["daq_signal"] == in_signal)
    ]

    return result


# ============================================================
# LOAD QUADRANT CSV
# ============================================================

def load_quadrant_csv(path):

    df_raw = pd.read_csv(
        path,
        header=None,
        dtype=str,
        encoding="latin1"
    )

    header_idx = df_raw.apply(
        lambda r:
        r.astype(str)
        .str.contains(
            "QUADRANT",
            case=False,
            na=False
        )
    ).any(axis=1).idxmax()

    df = (
        df_raw
        .iloc[header_idx + 1:]
        .reset_index(drop=True)
    )

    rows = []

    block_size = 10

    for _, row in df.iterrows():

        row = row.fillna("")

        for i in range(
            0,
            len(row),
            block_size + 1
        ):

            block = row[
                i:i + block_size
            ]

            if len(block) < block_size:
                continue

            if block.iloc[0] == "":
                continue

            rows.append(
                block.tolist()
            )

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

    return pd.DataFrame(
        rows,
        columns=columns
    )


# ============================================================
# READ CONFIG
# ============================================================

def read_config(filename):

    config = {}

    with open(
        filename,
        "r"
    ) as f:

        for line in f:

            line = line.split("#")[0].strip()

            if not line:
                continue

            if ":" in line:

                k, v = line.split(
                    ":",
                    1
                )

                k = k.strip()
                v = v.strip()

                try:

                    v = (
                        float(v)
                        if "." in v
                        else int(v)
                    )

                except:

                    pass

                config[k] = v

    return config


# ============================================================
# ASIC
# ============================================================

def get_asic(
    daq,
    tile,
    chid
):

    side = 0 if chid < 32 else 1

    t = tile % 5

    if side == 0:

        pattern = [
            0, 2, 4, 2, 0
        ]

    else:

        pattern = [
            1, 3, 4, 3, 1
        ]

    return pattern[t]


# ============================================================
# GEOMETRY
# ============================================================

def tile_ch_to_pos_and_global(
    tile_ID,
    ch_local_ID,
    config
):

    nx_tiles = config[
        "nx_tiles"
    ]

    ny_tiles = config[
        "ny_tiles"
    ]

    Nx_tot = config[
        "nx_sipm_pixel"
    ]

    Ny_tot = config[
        "ny_sipm_pixel"
    ]

    tile_row_id = (
        ny_tiles
        - tile_ID // nx_tiles
        - 1
    )

    if tile_row_id % 2 == 0:

        tile_col_id = (
            nx_tiles
            - tile_ID % nx_tiles
            - 1
        )

    else:

        tile_col_id = (
            tile_ID % nx_tiles
        )

    ch_local_ID = int(
        ch_local_ID
    )

    if tile_row_id % 2 == 0:

        ch_local_ID = (
            Nx_tot * Ny_tot
            - 1
            - ch_local_ID
        )

    dTx = (

        Nx_tot
        * config[
            "sensitive_sipm_pixel_sizeX"
        ]

        +

        (Nx_tot - 1)
        * config[
            "sipm_pixel_pitch_right"
        ]

        +

        2
        * config[
            "sipm_array_d"
        ]
    )

    dTy = (

        Ny_tot
        * config[
            "sensitive_sipm_pixel_sizeY"
        ]

        +

        int(Ny_tot / 2)
        * config[
            "sipm_pixel_pitch_down"
        ]

        +

        int((Ny_tot - 1) / 2)
        * config[
            "sipm_pixel_pitch_up"
        ]

        +

        2
        * config[
            "sipm_array_d"
        ]
    )

    tile_posX = (

        -nx_tiles * dTx / 2

        - (nx_tiles - 1)
        * config[
            "sipm_array_d"
        ] / 2

        + dTx / 2

        +

        (
            dTx
            + config[
                "sipm_array_d"
            ]
        )
        * tile_col_id
    )

    tile_posY = (

        -ny_tiles * dTy / 2

        - (ny_tiles - 1)
        * config[
            "sipm_array_d"
        ] / 2

        + dTy / 2

        +

        (
            dTy
            + config[
                "sipm_array_d"
            ]
        )
        * tile_row_id
    )

    pixel_local_col = (
        ch_local_ID // Nx_tot
    )

    pixel_local_row = (
        Ny_tot
        - 1
        - (
            ch_local_ID % Nx_tot
        )
    )

    pixel_posX = (

        tile_posX

        -

        Nx_tot
        * config[
            "sensitive_sipm_pixel_sizeX"
        ] / 2

        -

        (Nx_tot - 1)
        * config[
            "sipm_pixel_pitch_right"
        ] / 2

        +

        config[
            "sensitive_sipm_pixel_sizeX"
        ] / 2

        +

        (
            config[
                "sensitive_sipm_pixel_sizeX"
            ]

            +

            config[
                "sipm_pixel_pitch_right"
            ]
        )
        * pixel_local_col
    )

    pixel_posY = (

        tile_posY

        -

        Ny_tot
        * config[
            "sensitive_sipm_pixel_sizeY"
        ] / 2

        -

        int(Ny_tot / 2)
        * config[
            "sipm_pixel_pitch_down"
        ] / 2

        -

        int((Ny_tot - 1) / 2)
        * config[
            "sipm_pixel_pitch_up"
        ] / 2

        +

        (
            2 * pixel_local_row + 1
        )
        * config[
            "sensitive_sipm_pixel_sizeY"
        ] / 2

        +

        config[
            "sipm_pixel_pitch_down"
        ]
        * int(
            (pixel_local_row + 1) / 2
        )

        +

        config[
            "sipm_pixel_pitch_up"
        ]
        * int(
            pixel_local_row / 2
        )
    )

    ncols_total = (
        nx_tiles * Nx_tot
    )

    global_col = (
        tile_col_id * Nx_tot
        + pixel_local_col
    )

    global_row = (
        tile_row_id * Ny_tot
        + pixel_local_row
    )

    globalID = (
        global_row * ncols_total
        + global_col
    )

    return (
        round(pixel_posX, 2),
        round(pixel_posY, 2),
        int(globalID),
        round(tile_posX, 2),
        round(tile_posY, 2)
    )


# ============================================================
# BUILD COMPLETE MAPPING
# ============================================================

def build_complete_mapping(
    files,
    conf_path
):

    print()
    print("===================================")
    print("LOADING PIN-TO-PIN MAPPING")
    print("===================================")

    for f in files:

        if not os.path.exists(f):

            raise FileNotFoundError(
                f"Mapping file not found:\n{f}"
            )

    if not os.path.exists(conf_path):

        raise FileNotFoundError(
            f"Config file not found:\n{conf_path}"
        )

    pinmap = pd.concat(
        [
            load_quadrant_csv(f)
            for f in files
        ],
        ignore_index=True
    )

    config = read_config(
        conf_path
    )

    # --------------------------------------------------------
    # SIGNAL PARSING
    # --------------------------------------------------------

    col = (
        pinmap["signal name"]
        .astype("string")
        .str.strip()
    )

    parsed = col.str.extract(
        r"^([A-F])(\d+)_CH(\d+)$"
    )

    valid_mask = parsed[0].notna()

    pin_valid = (
        pinmap
        .loc[valid_mask]
        .copy()
    )

    parsed_valid = (
        parsed
        .loc[valid_mask]
        .copy()
    )

    complete_data = pd.DataFrame()

    complete_data["QUADRANT"] = (
        pin_valid["QUADRANT"]
        .reset_index(drop=True)
    )

    complete_data["TILE #"] = (
        pd.to_numeric(
            pin_valid["TILE"],
            errors="coerce"
        )
        .reset_index(drop=True)
    )

    complete_data["CHANNEL #"] = (
        pd.to_numeric(
            parsed_valid[2],
            errors="coerce"
        )
        .reset_index(drop=True)
    )

    complete_data["signal name"] = (
        pin_valid["signal name"]
        .reset_index(drop=True)
    )

    complete_data["GLOBAL ID"] = (
        pin_valid["GLOBAL ID"]
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # ASIC
    # --------------------------------------------------------

    complete_data["ASIC #"] = (
        complete_data.apply(
            lambda row:
            get_asic(
                int(row["QUADRANT"]),
                int(row["TILE #"]),
                int(row["CHANNEL #"])
            ),
            axis=1
        )
    )

    # --------------------------------------------------------
    # GEOMETRY
    # --------------------------------------------------------

    results = complete_data.apply(

        lambda r:
        tile_ch_to_pos_and_global(
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

    complete_data = pd.concat(
        [
            complete_data,
            results
        ],
        axis=1
    )

    # --------------------------------------------------------
    # SAVE MAPPING
    # --------------------------------------------------------

    nx = config["nx_tiles"]
    ny = config["ny_tiles"]

    mapping_file = (
        f"complete_mapping_{nx}x{ny}.csv"
    )

    complete_data.to_csv(
        mapping_file,
        index=False,
        float_format="%.2f"
    )

    print(
        f"MAPPING SAVED : {mapping_file}"
    )

    return (
        pinmap,
        complete_data,
        config
    )


# ============================================================
# CAMERA MAP
# ============================================================

def plot_camera_map(
    pinmap,
    complete_data,
    config,
    csv_file,
    value_column,
    title,
    output_file
):

    print()
    print(
        f"CREATING MAP : {value_column}"
    )

    summary = pd.read_csv(
        csv_file,
        sep=";"
    )

    rows = []

    # --------------------------------------------------------
    # BUILD MINI DATAFRAME
    # --------------------------------------------------------

    for _, row in summary.iterrows():

        daq_code = row[
            "DAQ_CODE"
        ]

        value = row[
            value_column
        ]

        if pd.isna(value):
            continue

        value = float(value)

        if value < 0:
            continue

        data_NI = filter_from_daq(
            pinmap,
            daq_code
        )

        if len(data_NI) == 0:
            continue

        col = (
            data_NI["signal name"]
            .astype("string")
            .str.strip()
        )

        parsed = col.str.extract(
            r"^([A-F])(\d+)_CH(\d+)$"
        )

        valid_mask = parsed[0].notna()

        ni_valid = (
            data_NI
            .loc[valid_mask]
            .copy()
        )

        parsed_valid = (
            parsed
            .loc[valid_mask]
            .copy()
        )

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

        print(
            f"No valid data for {value_column}"
        )

        return

    mini = pd.concat(
        rows,
        ignore_index=True
    )

    # --------------------------------------------------------
    # GEOMETRY
    # --------------------------------------------------------

    res = mini.apply(

        lambda r:
        tile_ch_to_pos_and_global(
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

    mini = pd.concat(
        [
            mini,
            res
        ],
        axis=1
    )

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 5),
        dpi=150
    )


    min_value = mini[
        "VALUE"
    ].min()

    max_value = mini[
        "VALUE"
    ].max()

    # Evita problemi quando tutti i valori sono uguali
    if min_value == max_value:

        if min_value == 0:

            max_value = 1

        else:

            min_value = 0

    norm = plt.Normalize(
        vmin=min_value,
        vmax=max_value
    )

    cmap = plt.cm.jet

    # --------------------------------------------------------
    # EMPTY PIXELS
    # --------------------------------------------------------

    for _, row in complete_data.iterrows():

        posx = row[
            "PixelCentrePosX_mm"
        ]

        posy = row[
            "PixelCentrePosY_mm"
        ]

        rect = patches.Rectangle(

            (
                posx
                -
                config[
                    "sensitive_sipm_pixel_sizeX"
                ] / 2,

                posy
                -
                config[
                    "sensitive_sipm_pixel_sizeY"
                ] / 2
            ),

            config[
                "sensitive_sipm_pixel_sizeX"
            ],

            config[
                "sensitive_sipm_pixel_sizeY"
            ],

            facecolor="whitesmoke",
            edgecolor="lightgray",
            linewidth=0.25
        )

        ax.add_patch(rect)

    # --------------------------------------------------------
    # ACTIVE PIXELS
    # --------------------------------------------------------

    for _, row in mini.iterrows():

        posx = row[
            "PixelCentrePosX_mm"
        ]

        posy = row[
            "PixelCentrePosY_mm"
        ]

        value = row[
            "VALUE"
        ]

        color = cmap(
            norm(value)
        )

        rect = patches.Rectangle(

            (
                posx
                -
                config[
                    "sensitive_sipm_pixel_sizeX"
                ] / 2,

                posy
                -
                config[
                    "sensitive_sipm_pixel_sizeY"
                ] / 2
            ),

            config[
                "sensitive_sipm_pixel_sizeX"
            ],

            config[
                "sensitive_sipm_pixel_sizeY"
            ],

            facecolor=color,
            edgecolor="black",
            linewidth=0.4
        )

        ax.add_patch(rect)

    # --------------------------------------------------------
    # COLORBAR
    # --------------------------------------------------------

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=norm
    )

    sm.set_array([])

    cbar = plt.colorbar(
        sm,
        ax=ax
    )

    cbar.set_label(
        value_column,
        fontsize=14
    )

    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

    ax.set_aspect(
        "equal"
    )

    xlim = (
        np.max(
            complete_data[
                "PixelCentrePosX_mm"
            ]
        )
        + 5
    )

    ylim = (
        np.max(
            complete_data[
                "PixelCentrePosY_mm"
            ]
        )
        + 5
    )

    ax.set_xlim(
        -xlim,
        xlim
    )

    ax.set_ylim(
        -ylim,
        ylim
    )

    ax.set_xlabel(
        "x (mm)"
    )

    ax.set_ylabel(
        "y (mm)"
    )

    ax.set_title(
        title
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.2
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"SAVED : {output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.perf_counter()

    # ========================================================
    # ARGUMENTS
    # ========================================================

    parser = argparse.ArgumentParser(
        description=(
            "TERZINA channel summary and "
            "camera map generator"
        )
    )

    parser.add_argument(
        "folder",
        help=(
            "Folder containing the TERZINA .bin file"
        )
    )

    args = parser.parse_args()

    input_folder = args.folder

    # ========================================================
    # DATA PROCESSING
    # ========================================================

    data_proc = DataProcessing(
        input_folder
    )

    # --------------------------------------------------------
    # PROCESS BIN
    # --------------------------------------------------------

    data_proc.process_pck()

    # --------------------------------------------------------
    # CREATE CHANNEL SUMMARY
    # --------------------------------------------------------

    data_proc.export_channel_summary()

    # ========================================================
    # MAPPING FILES
    # ========================================================

    mapping_files = [

        "finalmappingsTERZINAGlobalIDmapping(23_03_2026)-QUADRANT0.csv",

        "finalmappingsTERZINAGlobalIDmapping(23_03_2026)-QUADRANT1.csv",

        "finalmappingsTERZINAGlobalIDmapping(23_03_2026)-QUADRANT2.csv",

        "finalmappingsTERZINAGlobalIDmapping(23_03_2026)-QUADRANT3.csv"
    ]

    conf_path = "./TerzinaFPA.conf"

    # ========================================================
    # BUILD MAPPING
    # ========================================================

    (
        pinmap,
        complete_data,
        config
    ) = build_complete_mapping(
        mapping_files,
        conf_path
    )

    # ========================================================
    # CAMERA MAPS
    # ========================================================

    print()
    print("===================================")
    print("CREATING CAMERA MAPS")
    print("===================================")

    # --------------------------------------------------------
    # HIT
    # --------------------------------------------------------

    plot_camera_map(

        pinmap,
        complete_data,
        config,

        data_proc.csv_file,

        "HIT",

        "Camera HIT Occupancy",

        os.path.join(
            data_proc.plot_dir,
            "Camera_HIT_Occupancy.png"
        )
    )

    # --------------------------------------------------------
    # HG
    # --------------------------------------------------------

    plot_camera_map(

        pinmap,
        complete_data,
        config,

        data_proc.csv_file,

        "HG_MEAN",

        "Camera HG Mean",

        os.path.join(
            data_proc.plot_dir,
            "Camera_HG_Mean.png"
        )
    )

    # --------------------------------------------------------
    # LG
    # --------------------------------------------------------

    plot_camera_map(

        pinmap,
        complete_data,
        config,

        data_proc.csv_file,

        "LG_MEAN",

        "Camera LG Mean",

        os.path.join(
            data_proc.plot_dir,
            "Camera_LG_Mean.png"
        )
    )

    # ========================================================
    # TOTAL TIME
    # ========================================================

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    print()
    print("===================================")
    print("DONE")
    print("===================================")

    print(
        f"TOTAL TIME : "
        f"{total_elapsed:.3f} s"
    )

    print()
    print(
        f"CHANNEL SUMMARY : "
        f"{data_proc.csv_file}"
    )

    print(
        f"PLOTS DIRECTORY : "
        f"{data_proc.plot_dir}"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
