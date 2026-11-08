import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = PROJECT_ROOT / "data/raw_sample/snapshots/binance/BTCUSDT/snapshot.parquet"
DEPTH_PATH = PROJECT_ROOT / "data/raw_sample/depth/binance/BTCUSDT/2026-06_snapshot.parquet"

print("=== SNAPSHOT FILE ===")
try:
    snap = pd.read_parquet(SNAPSHOT_PATH)
    print(f"Columns: {snap.columns.tolist()}")
    print(f"Shape: {snap.shape}")
    print(snap.head(3))
    print(f"Dtypes:\n{snap.dtypes}")
except Exception as e:
    print(f"Error reading snapshot: {e}")

print("\n=== DEPTH FILE ===")
try:
    depth = pd.read_parquet(DEPTH_PATH)
    print(f"Columns: {depth.columns.tolist()}")
    print(f"Shape: {depth.shape}")
    print(depth.head(3))
    print(f"Dtypes:\n{depth.dtypes}")
except Exception as e:
    print(f"Error reading depth: {e}")