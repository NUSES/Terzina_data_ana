"""export_timestamps.py — estrae timestamp e trig counter dal binario TERZINA.

Uso:
    python export_timestamps.py <file.bin> [output.csv]

Se output.csv non viene specificato, viene creato accanto al .bin
con suffisso _timestamps.csv.

Colonne CSV (separatore ;):
    pkt_count, conc_timestamp,
    d1_trig, d2_trig, d3_trig, d4_trig,
    d1_timestamp, d2_timestamp, d3_timestamp, d4_timestamp
    
    
FAQ:
Q1. How's the T0 of the package timestamp defined?
A1. The T0 of the timestamps is no universally defined or synchronized between
the DAQ boards and the concentrator board. Further, the timestamps are produced
by independent clocks with a 10 ns period for the concentrator board, and with
a 6.67 ns period for the DAQ board (100 MHz and 150 MHz frequency respectively).


Q2. How often are the timestamps collected?
A2. With every trigger.

Q3. What are the differences between the concentrator and DAQ timestamps.
A3. The concentrator timestamp corresponds to the instant when the concentrator
recieved the data packets from the DAQ board and made a readout request to the
DAQ software. The DAQ timestamp corresponds instead to the moment that the DAQ
board finishes reading out the CITIROC (following a trigger) and sends the data
to the concentrator board.

Q4. How can you determine the DAQ duration from the data?
A4. You cannot, the software should be modified to provide this information.
"""

import sys
import os
import csv
import struct

PACKET_ID = 0xA77A
CONC_ID   = 0xCA00FE11DD651983

DAQ_SENTINELS = {
    0x00AAAA00: 1,
    0x00BBBB00: 2,
    0x00CCCC00: 3,
    0x00DDDD00: 4,
}

CSV_HEADER = [
    "pkt_count", "conc_timestamp",
    "d1_trig", "d2_trig", "d3_trig", "d4_trig",
    "d1_timestamp", "d2_timestamp", "d3_timestamp", "d4_timestamp",
]


def _parse_payload(payload: bytes) -> dict | None:
    offset = 0

    def u32():
        nonlocal offset
        v = int.from_bytes(payload[offset:offset + 4], "little")
        offset += 4
        return v

    def u64():
        nonlocal offset
        v = int.from_bytes(payload[offset:offset + 8], "little")
        offset += 8
        return v

    def skip_asics():
        nonlocal offset
        offset += 5 * 32 * 4   # 5 ASIC × 32 ch × 4 byte

    if len(payload) < 16:
        return None

    timestamp = u64()
    conc_id   = u64()

    if conc_id != CONC_ID:
        return None

    row = {
        "conc_timestamp": timestamp,
        "d1_trig": "", "d2_trig": "", "d3_trig": "", "d4_trig": "",
        "d1_timestamp": "", "d2_timestamp": "", "d3_timestamp": "", "d4_timestamp": "",
    }

    id_prog = 0
    daq_id  = u32()

    while True:
        n = DAQ_SENTINELS.get(daq_id)
        if n is not None:
            trig  = u32()
            _val  = u32()   # validated field
            ts    = u32()   # DAQ timestamp
            skip_asics()
            u64()           # LOST
            u64()           # REAL
            row[f"d{n}_trig"]      = trig
            row[f"d{n}_timestamp"] = ts
            id_prog += 1
            if n == 4:
                break
            daq_id = u32()

        elif daq_id == 0xDEADBEEF:
            if id_prog == 3:
                break
            id_prog += 1
            daq_id = u32()
            daq_id = u32()

        else:
            break

    return row


def export(bin_path: str, csv_path: str) -> None:
    total = 0
    skipped = 0

    with open(bin_path, "rb") as fbin, open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=CSV_HEADER,
                                delimiter=";", extrasaction="ignore")
        writer.writeheader()

        while True:
            header = fbin.read(12)
            if len(header) < 12:
                break

            id_pack, len_pack, pkt_count = struct.unpack("<III", header)

            payload = fbin.read(len_pack)
            if len(payload) < len_pack:
                print(f"  Pacchetto troncato #{pkt_count} — fine file.")
                break

            if id_pack != PACKET_ID:
                skipped += 1
                continue

            row = _parse_payload(payload)
            if row is None:
                skipped += 1
                continue

            row["pkt_count"] = pkt_count
            writer.writerow(row)
            total += 1

            if total % 50000 == 0:
                print(f"  {total} eventi scritti…")

    print(f"\nEventi scritti : {total}")
    if skipped:
        print(f"Pacchetti saltati: {skipped}")
    print(f"Output         : {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    bin_path = sys.argv[1]
    if not os.path.isfile(bin_path):
        print(f"Errore: file non trovato — {bin_path}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        csv_path = sys.argv[2]
    else:
        base = os.path.splitext(bin_path)[0]
        csv_path = base + "_timestamps.csv"

    print(f"Elaborazione di: {bin_path}")
    export(bin_path, csv_path)
