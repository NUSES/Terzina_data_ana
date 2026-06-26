"""
============================================================
TERZINA Binary Data Processing Tool
============================================================

Author: Caterina Trimarelli, Luigi Ferrentino

Description:
    This software reads TERZINA binary acquisition files,
    decodes DAQ packets, extracts HG/LG/HIT channel data,
    exports event information to CSV, computes detector
    statistics and generates diagnostic plots.

Main Features:
    - Binary packet parsing
    - Multi-DAQ support (DAQ1-DAQ4)
    - HG/LG/HIT extraction
    - CSV export
    - Channel occupancy analysis
    - Automatic histogram generation

Output:
    - Event CSV
    - Channel summary CSV
    - HG/LG/HIT histograms
    - Occupancy plots

============================================================
"""
import os
import csv
from dataclasses import dataclass, field, fields, is_dataclass
from typing import List
import matplotlib.pyplot as plt


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

    def __init__(self):

        #DATE = "05_11"
        #TIME = "18_35"

        #        self.file_path = f"download_data_{DATE}-{TIME}.bin"
        self.file_path = "data_05_28-11_32/data_05_28-11_32.bin"
        self.output_dir = "data_05_28-11_32/"
        self.plot_dir = os.path.join(self.output_dir, "plots")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        self.csv_file = os.path.join(
            self.output_dir,
            "terzina_data_converted_05_28-11_32.csv"
        )

        if os.path.exists(self.csv_file):
            os.remove(self.csv_file)

        self.datalen = os.path.getsize(self.file_path)

        print()
        print("===================================")
        print(f"INPUT FILE : {self.file_path}")
        print(f"FILE SIZE  : {self.datalen} bytes")
        print("===================================")
        print()

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




    # ========================================================
    # EXPORT CHANNEL SUMMARY CSV
    # ========================================================
    
    def export_channel_summary(self):
        
        print()
        print("===================================")
        print("EXPORTING CHANNEL SUMMARY")
        print("===================================")
        
        output_file = os.path.join(
            self.output_dir,
            "channel_summary_11_32.csv"
        )
        
        prefixes = [
            "D1_A", "D1_B", "D1_C", "D1_D", "D1_E",
            "D2_A", "D2_B", "D2_C", "D2_D", "D2_E",
            "D3_A", "D3_B", "D3_C", "D3_D", "D3_E",
            "D4_A", "D4_B", "D4_C", "D4_D", "D4_E"
        ]
        
        with open(output_file, "w", newline="") as file:
            
            writer = csv.writer(file, delimiter=";")
            
            # header
            writer.writerow([
                "DAQ_CODE",
                "HIT",
                "HG_MEAN",
                "LG_MEAN"
            ])
            
            # loop groups
            for prefix in prefixes:
                
                HG_values = self.channels[f"{prefix}_HG"]
                LG_values = self.channels[f"{prefix}_LG"]
                HIT_values = self.channels[f"{prefix}_HIT"]
                
                for ch in range(32):
                    
                    hg_ch = HG_values[ch::32]
                    lg_ch = LG_values[ch::32]
                    hit_ch = HIT_values[ch::32]
                    
                    if len(hg_ch) == 0:
                        continue
                    
                    # mean HG
                    hg_valid = [v for v in hg_ch if v > 0]
                    
                    hg_mean = (
                        sum(hg_valid) / len(hg_valid)
                        if len(hg_valid) > 0 else 0
                    )
                    
                    # mean LG
                    lg_valid = [v for v in lg_ch if v > 0]
                    
                    lg_mean = (
                        sum(lg_valid) / len(lg_valid)
                        if len(lg_valid) > 0 else 0
                    )
                    
                    # occupancy HIT
                    hit_count = sum(1 for v in hit_ch if v > 0)
                    
                    daq_code = f"{prefix}_HG_{ch:02d}"
                    
                    writer.writerow([
                        daq_code,
                        hit_count,
                        round(hg_mean, 2),
                        round(lg_mean, 2)
                    ])
                    
            print(f"SAVED : {output_file}")
            
    # ========================================================
    # CSV WRITER
    # ========================================================

    def write_pack_to_csv(self, pack):

        if not is_dataclass(pack):
            raise ValueError("pack must be dataclass")

        with open(self.csv_file, "a", newline="") as file:

            writer = csv.writer(file, delimiter=";")

            if os.path.getsize(self.csv_file) == 0:

                header = []

                for f in fields(pack):

                    value = getattr(pack, f.name)

                    if isinstance(value, list):

                        header.extend(
                            [f"{f.name}[{i}]" for i in range(len(value))]
                        )

                    else:
                        header.append(f.name)

                writer.writerow(header)

            row = []

            for f in fields(pack):

                value = getattr(pack, f.name)

                if isinstance(value, list):
                    row.extend(value)
                else:
                    row.append(value)

            writer.writerow(row)

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
    # UPDATE CHANNEL STORAGE
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

            HG, LG, HIT = self.extract_channels(chan_list)

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

            val = int.from_bytes(
                payload[offset:offset + 4],
                "little"
            )

            offset += 4

            return val

        def u64():

            nonlocal offset

            val = int.from_bytes(
                payload[offset:offset + 8],
                "little"
            )

            offset += 8

            return val

        f.TIMESTAMP = u64()
        f.ID_CONC = u64()

        if f.ID_CONC != 0xCA00FE11DD651983:
            return f

        while offset < len(payload):

            daq_id = u32()

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

            elif daq_id == 0xDEADBEEF:
                break

            else:

                print(f"UNKNOWN DAQ ID : {hex(daq_id)}")
                break

        return f

    # ========================================================
    # PROCESS FILE
    # ========================================================

    def process_pck(self):

        with open(self.file_path, "rb") as file:

            while True:

                raw = file.read(4)

                if len(raw) < 4:
                    print()
                    print("END OF FILE")
                    print()
                    break

                ID_pack = int.from_bytes(raw, "little")
                len_pack = int.from_bytes(file.read(4), "little")
                pack_counter = int.from_bytes(file.read(4), "little")

                if ID_pack == 0xA77A:

                    print(f"PACK {pack_counter}")

                    payload = file.read(len_pack)

                    if len(payload) != len_pack:

                        print("TRUNCATED PACK")
                        break

                    terzina_pack = self.parse_terzina_payload(payload)

                    self.update_channel_data(terzina_pack)

                    self.write_pack_to_csv(terzina_pack)

                else:

                    print(
                        f"INVALID PACK ID : "
                        f"{hex(ID_pack)}"
                    )

    # ========================================================
    # PRINT ACTIVE CHANNELS
    # ========================================================

    def print_active_channels(self):

        print()
        print("===================================")
        print("ACTIVE CHANNELS")
        print("===================================")

        for name, values in self.channels.items():

            if "HG" not in name:
                continue

            for ch in range(32):

                ch_values = values[ch::32]

                if len(ch_values) == 0:
                    continue

                vmax = max(ch_values)

                if vmax > 0:

                    mean = sum(ch_values) / len(ch_values)

                    print(
                        f"{name} CH{ch:02d} | "
                        f"MEAN={mean:.2f} | "
                        f"MAX={vmax}"
                    )

    # ========================================================
    # PRINT HIT STATISTICS
    # ========================================================

    def print_hit_stats(self):

        print()
        print("===================================")
        print("HIT STATISTICS")
        print("===================================")

        for name, values in self.channels.items():

            if "HIT" not in name:
                continue

            for ch in range(32):

                ch_values = values[ch::32]

                if len(ch_values) == 0:
                    continue

                nhits = sum(1 for v in ch_values if v > 0)

                if nhits > 0:

                    max_hit = max(ch_values)
                    mean_hit = sum(ch_values) / len(ch_values)

                    print(
                        f"{name} CH{ch:02d} | "
                        f"HITS={nhits} | "
                        f"MEAN={mean_hit:.2f} | "
                        f"MAX={max_hit}"
                    )

    # ========================================================
    # PLOT ALL CHANNELS
    # ========================================================

    def plot_all_channels(self, mode="HG"):

        print()
        print("===================================")
        print(f"GENERATING {mode} PLOTS")
        print("===================================")

        daq_map = {
            "DAQ1": [f"D1_A_{mode}", f"D1_B_{mode}", f"D1_C_{mode}", f"D1_D_{mode}", f"D1_E_{mode}"],
            "DAQ2": [f"D2_A_{mode}", f"D2_B_{mode}", f"D2_C_{mode}", f"D2_D_{mode}", f"D2_E_{mode}"],
            "DAQ3": [f"D3_A_{mode}", f"D3_B_{mode}", f"D3_C_{mode}", f"D3_D_{mode}", f"D3_E_{mode}"],
            "DAQ4": [f"D4_A_{mode}", f"D4_B_{mode}", f"D4_C_{mode}", f"D4_D_{mode}", f"D4_E_{mode}"],
        }

        for daq_name, groups in daq_map.items():

            active_groups = []

            for group in groups:

                values = self.channels[group]

                if len(values) == 0:
                    continue

                if max(values) > 0:
                    active_groups.append(group)

            if len(active_groups) == 0:

                print(f"{daq_name} EMPTY")
                continue

            nrows = len(active_groups)

            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=32,
                figsize=(48, 3 * nrows)
            )

            if nrows == 1:
                axes = [axes]

            fig.suptitle(
                f"{daq_name} {mode} CHANNELS",
                fontsize=24
            )

            for row, group_name in enumerate(active_groups):

                values = self.channels[group_name]

                for ch in range(32):

                    ax = axes[row][ch]

                    ch_values = values[ch::32]

                    if mode != "HIT":
                        ch_values = [v for v in ch_values if v > 0]

                    if len(ch_values) == 0:

                        ax.axis("off")
                        continue

                    if mode == "HIT":

                        ax.hist(
                            ch_values,
                            bins=16,
                            range=(0, 16)
                        )

                        ax.set_xlim(0, 15)

                    else:

                        xmin = min(ch_values)
                        xmax = max(ch_values)

                        if xmin == xmax:
                            xmin -= 1
                            xmax += 1

                        margin = (xmax - xmin) * 0.1

                        xmin = max(0, xmin - margin)
                        xmax = min(16384, xmax + margin)

                        ax.hist(
                            ch_values,
                            bins=100,
                            range=(xmin, xmax)
                        )

                        ax.set_yscale("log")

                    ax.set_title(
                        f"{group_name[-6:-5]}{ch}",
                        fontsize=7
                    )

                    ax.tick_params(
                        axis='both',
                        labelsize=5
                    )

            plt.tight_layout()

            output_file = os.path.join(
                self.plot_dir,
                f"{daq_name}_{mode}.png"
            )

            plt.savefig(
                output_file,
                dpi=300
            )

            plt.close()

            print(f"SAVED : {output_file}")

    # ========================================================
    # HIT OCCUPANCY
    # ========================================================

    def plot_hit_occupancy(self):

        print()
        print("===================================")
        print("GENERATING HIT OCCUPANCY")
        print("===================================")

        daq_map = {
            "DAQ1": ["D1_A_HIT", "D1_B_HIT", "D1_C_HIT", "D1_D_HIT", "D1_E_HIT"],
            "DAQ2": ["D2_A_HIT", "D2_B_HIT", "D2_C_HIT", "D2_D_HIT", "D2_E_HIT"],
            "DAQ3": ["D3_A_HIT", "D3_B_HIT", "D3_C_HIT", "D3_D_HIT", "D3_E_HIT"],
            "DAQ4": ["D4_A_HIT", "D4_B_HIT", "D4_C_HIT", "D4_D_HIT", "D4_E_HIT"],
        }

        for daq_name, groups in daq_map.items():

            fig, axes = plt.subplots(
                len(groups),
                1,
                figsize=(18, 10)
            )

            fig.suptitle(
                f"{daq_name} HIT OCCUPANCY",
                fontsize=20
            )

            for row, group in enumerate(groups):

                values = self.channels[group]

                occupancy = []

                for ch in range(32):

                    ch_values = values[ch::32]

                    hits = sum(1 for v in ch_values if v > 0)

                    occupancy.append(hits)

                axes[row].bar(
                    range(32),
                    occupancy
                )

                axes[row].set_title(group)
                axes[row].set_xlabel("Channel")
                axes[row].set_ylabel("Hits")

            plt.tight_layout()

            output_file = os.path.join(
                self.plot_dir,
                f"{daq_name}_HIT_OCCUPANCY.png"
            )

            plt.savefig(output_file, dpi=300)

            plt.close()

            print(f"SAVED {output_file}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    data_proc = DataProcessing()

    data_proc.process_pck()

    data_proc.print_active_channels()
    data_proc.print_hit_stats()
    data_proc.export_channel_summary()
    data_proc.plot_all_channels(mode="HG")
    data_proc.plot_all_channels(mode="LG")
    data_proc.plot_all_channels(mode="HIT")

    data_proc.plot_hit_occupancy()

    print()
    print("===================================")
    print("DONE")
    print("===================================")
