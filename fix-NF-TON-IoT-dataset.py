"""
fix-NF-TON-IoT-dataset.py
=========================
Fixes the NF-ToN-IoT dataset column schema so it matches what the FedGATSage
code expects.

Problems solved:
  1. Renames all columns to the CIC-style names the code uses
     (e.g. IPV4_SRC_ADDR -> Src IP, FLOW_DURATION_MILLISECONDS -> Flow Duration)
  2. Derives Flow Pkts/s from existing packet + duration columns
  3. Derives SYN / RST / ACK flag counts from the TCP_FLAGS bitmask
  4. Fills Flow IAT Mean / Std with 0 (no equivalent exists in the NF schema)

Output: data/NF-ToN-IoT-fixed.csv

Usage:
    python fix-NF-TON-IoT-dataset.py
"""

import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

INPUT_FILE  = os.path.join('data', 'NF-ToN-IoT.csv')
OUTPUT_FILE = os.path.join('data', 'NF-ToN-IoT-fixed.csv')
CHUNK_SIZE  = 100_000   # rows per chunk — keeps memory under ~1 GB

# ------------------------------------------------------------------
# Column rename map: NF name -> CIC/code name
# ------------------------------------------------------------------
RENAME_MAP = {
    'IPV4_SRC_ADDR':            'Src IP',
    'IPV4_DST_ADDR':            'Dst IP',
    'L4_SRC_PORT':              'Src Port',
    'L4_DST_PORT':              'Dst Port',
    'PROTOCOL':                 'Protocol',
    'FLOW_DURATION_MILLISECONDS': 'Flow Duration',
    'IN_PKTS':                  'Tot Fwd Pkts',
    'OUT_PKTS':                 'Tot Bwd Pkts',
    'IN_BYTES':                 'TotLen Fwd Pkts',
    'OUT_BYTES':                'TotLen Bwd Pkts',
}

# TCP_FLAGS bitmask positions (standard RFC 793)
_SYN_BIT = 0x02
_RST_BIT = 0x04
_ACK_BIT = 0x10


def derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the columns the code needs that are not directly present in NF schema."""

    # --- Flow Pkts/s ---------------------------------------------------
    # IN_PKTS / OUT_PKTS are *before* renaming here, so use original names
    # (this function is called before rename)
    duration_s = df['FLOW_DURATION_MILLISECONDS'] / 1000.0
    total_pkts  = df['IN_PKTS'] + df['OUT_PKTS']
    df['Flow Pkts/s'] = total_pkts / (duration_s + 1e-9)

    # --- Flow IAT Mean / Std  ------------------------------------------
    # No inter-arrival time data in NF schema; fill with 0 so feature
    # engineering skips gracefully (code guards with `if col in df.columns`)
    df['Flow IAT Mean'] = 0.0
    df['Flow IAT Std']  = 0.0

    # --- TCP Flag counts from TCP_FLAGS bitmask ------------------------
    flags = df['TCP_FLAGS'].fillna(0).astype(int)
    df['SYN Flag Cnt'] = ((flags & _SYN_BIT) > 0).astype(int)
    df['RST Flag Cnt'] = ((flags & _RST_BIT) > 0).astype(int)
    df['ACK Flag Cnt'] = ((flags & _ACK_BIT) > 0).astype(int)

    return df


def fix_dataset():
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        logger.error("Make sure you are running this script from the Fed_GNN/ directory.")
        return

    logger.info(f"Reading {INPUT_FILE} in chunks of {CHUNK_SIZE:,} rows ...")

    total_rows = 0
    first_chunk = True

    reader = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False)

    for i, chunk in enumerate(reader):
        # 1. Derive new columns while original names still exist
        chunk = derive_columns(chunk)

        # 2. Rename columns
        chunk.rename(columns=RENAME_MAP, inplace=True)

        # 3. Write — header only for the first chunk
        write_mode = 'w' if first_chunk else 'a'
        chunk.to_csv(OUTPUT_FILE, index=False, mode=write_mode, header=first_chunk)

        total_rows += len(chunk)
        first_chunk = False
        logger.info(f"  Processed chunk {i+1} — {total_rows:,} rows written so far")

    logger.info(f"Done. Fixed dataset saved to: {OUTPUT_FILE}")
    logger.info(f"Total rows: {total_rows:,}")

    # ------------------------------------------------------------------
    # Quick validation
    # ------------------------------------------------------------------
    logger.info("Running quick validation ...")
    sample = pd.read_csv(OUTPUT_FILE, nrows=5)

    required = ['Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol',
                'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
                'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
                'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std',
                'SYN Flag Cnt', 'RST Flag Cnt', 'ACK Flag Cnt',
                'Attack']

    all_ok = True
    for col in required:
        if col in sample.columns:
            logger.info(f"  ✓  {col}")
        else:
            logger.error(f"  ✗  {col}  <-- STILL MISSING")
            all_ok = False

    if all_ok:
        logger.info("All required columns present. NF-ToN-IoT is ready to use.")
        logger.info(f"\nTo preprocess for federated clients, run:")
        logger.info(f"  python preprocess_data.py --input_file {OUTPUT_FILE} "
                    f"--output_dir data/nf --num_clients 5")
    else:
        logger.error("Some columns are still missing — check the script logic above.")


if __name__ == '__main__':
    fix_dataset()
