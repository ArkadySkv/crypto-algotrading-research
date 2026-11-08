import pyarrow.parquet as pq
from pathlib import Path
import shutil

SOURCE_DEPTH = Path("crypto_research/data/raw/depth/binance/BTCUSDT/2026-06.parquet")
SOURCE_SNAPSHOT = Path("crypto_research/data/raw/snapshots/binance/BTCUSDT/2026-06_snapshot.parquet")

TARGET_DIR = Path("crypto_research/data/raw_sample")
TARGET_DEPTH = TARGET_DIR / "depth/binance/BTCUSDT"
TARGET_SNAPSHOT = TARGET_DIR / "snapshots/binance/BTCUSDT"

TARGET_DEPTH.mkdir(parents=True, exist_ok=True)
TARGET_SNAPSHOT.mkdir(parents=True, exist_ok=True)

print("Reading first 100,000 rows from depth file...")
parquet_file = pq.ParquetFile(SOURCE_DEPTH)
batch = next(parquet_file.iter_batches(batch_size=100_000))
df_sample = batch.to_pandas()

df_sample.to_parquet(TARGET_DEPTH / "sample.parquet", index=False)
print(f"Saved {len(df_sample)} rows to {TARGET_DEPTH / 'sample.parquet'}")

shutil.copy(SOURCE_SNAPSHOT, TARGET_SNAPSHOT / "snapshot.parquet")
print(f"Copied snapshot to {TARGET_SNAPSHOT / 'snapshot.parquet'}")