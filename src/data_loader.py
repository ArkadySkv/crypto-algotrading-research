import pandas as pd
import numpy as np
from pathlib import Path

class LOBDataLoader:
    def __init__(self, asset="BTCUSDT", data_dir="data"):
        self.asset = asset
        self.processed_dir = Path(data_dir) / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def generate_realistic_lob(self, num_snapshots=50_000, step_ms=100):
        """
        Генерирует реалистичные LOB-снимки (Top-5) с правильным шагом времени.
        Каждый снапшот - новый момент времени, а не микросекунда.
        """
        out_path = self.processed_dir / f"{self.asset}_lob_realistic.parquet"
        if out_path.exists():
            print(f"[INFO] Realistic LOB already exists: {out_path}")
            return out_path

        print(f"[INFO] Generating {num_snapshots} realistic LOB snapshots (step={step_ms}ms)...")
        
        # Стартовая цена BTC
        start_price = 63000.0
        
        # Random walk по mid-price
        np.random.seed(42)
        returns = np.random.normal(0, 0.0002, size=num_snapshots)
        mid_prices = start_price * np.exp(np.cumsum(returns))
        
        # Спред (типичный для BTCUSDT: 0.5 - 5.0)
        spreads = np.random.uniform(0.5, 5.0, size=num_snapshots)
        
        best_bids = mid_prices - (spreads / 2)
        best_asks = mid_prices + (spreads / 2)
        
        # Объемы на 5 уровнях (bid и ask)
        bid_vols = np.random.uniform(5.0, 50.0, size=num_snapshots)
        ask_vols = np.random.uniform(5.0, 50.0, size=num_snapshots)
        
        # Таймстемпы: каждые step_ms миллисекунд
        start_ts = pd.Timestamp("2026-06-04 00:00:00").timestamp() * 1000
        timestamps = [start_ts + i * step_ms for i in range(num_snapshots)]
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps, unit='ms'),
            'best_bid': best_bids,
            'best_ask': best_asks,
            'bid_vol_5': bid_vols,
            'ask_vol_5': ask_vols,
            'mid_price': mid_prices,
            'spread': spreads
        })
        df.set_index('timestamp', inplace=True)
        df.to_parquet(out_path)
        print(f"[SUCCESS] Saved {len(df)} realistic LOB snapshots to {out_path}")
        return out_path

if __name__ == "__main__":
    loader = LOBDataLoader()
    loader.generate_realistic_lob(num_snapshots=50_000, step_ms=100)