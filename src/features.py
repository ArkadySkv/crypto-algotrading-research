import pandas as pd
import numpy as np
from pathlib import Path

def calculate_obi(bids, asks, levels=5):
    bid_vol = sum(qty for _, qty in bids[:levels])
    ask_vol = sum(qty for _, qty in asks[:levels])
    if (bid_vol + ask_vol) == 0:
        return 0.0
    return (bid_vol - ask_vol) / (bid_vol + ask_vol)

def calculate_ofi(prev_bid, prev_bid_qty, prev_ask, prev_ask_qty, curr_bid, curr_bid_qty, curr_ask, curr_ask_qty):
    ofi = 0.0
    if curr_bid > prev_bid: ofi += curr_bid_qty
    elif curr_bid < prev_bid: ofi -= prev_bid_qty
    elif curr_bid == prev_bid: ofi += (curr_bid_qty - prev_bid_qty)
        
    if curr_ask < prev_ask: ofi -= curr_ask_qty
    elif curr_ask > prev_ask: ofi += prev_ask_qty
    elif curr_ask == prev_ask: ofi -= (curr_ask_qty - prev_ask_qty)
    return ofi

import pandas as pd
import numpy as np

def build_features_from_raw_files(depth_path, top_levels=5):
    """
    Строит признаки из диффов стакана (без снапшота, т.к. снапшот не совпадает по времени).
    """
    print(f"Loading depth diffs from {depth_path}...")
    diffs = pd.read_parquet(depth_path)
    diffs = diffs.sort_values('timestamp_ms').reset_index(drop=True)
    
    bids = {}
    asks = {}
    features = []
    
    prev_bid, prev_bid_qty, prev_ask, prev_ask_qty = None, None, None, None
    
    for idx, row in diffs.iterrows():
        price = float(row['price'])
        qty = float(row['quantity'])
        side = str(row['side']).lower()
        ts = row['timestamp_ms']
        
        # Применяем дифф к стакану
        if side == 'bid':
            if qty == 0: bids.pop(price, None)
            else: bids[price] = qty
        elif side == 'ask':
            if qty == 0: asks.pop(price, None)
            else: asks[price] = qty
        
        # Ждем, пока наберется достаточная глубина
        if len(bids) < top_levels or len(asks) < top_levels:
            continue
        
        # Берем топ-N уровней
        top_bids = sorted(bids.items(), key=lambda x: x[0], reverse=True)[:top_levels]
        top_asks = sorted(asks.items(), key=lambda x: x[0])[:top_levels]
        
        best_bid, best_bid_qty = top_bids[0]
        best_ask, best_ask_qty = top_asks[0]
        
        if best_bid >= best_ask:
            continue
            
        mid_price = (best_bid + best_ask) / 2.0
        
        # Microprice
        total_qty = best_bid_qty + best_ask_qty
        microprice = (best_bid * best_ask_qty + best_ask * best_bid_qty) / total_qty if total_qty > 0 else mid_price
        
        # OBI
        bid_vol = sum(q for _, q in top_bids)
        ask_vol = sum(q for _, q in top_asks)
        obi = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0.0
        
        # OFI
        ofi = 0.0
        if prev_bid is not None:
            ofi = calculate_ofi(prev_bid, prev_bid_qty, prev_ask, prev_ask_qty,
                                best_bid, best_bid_qty, best_ask, best_ask_qty)
        
        features.append({
            'timestamp': pd.to_datetime(ts, unit='ms'),
            'mid_price': mid_price,
            'microprice': microprice,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': best_ask - best_bid,
            'obi': obi,
            'ofi': ofi
        })
        
        prev_bid, prev_bid_qty = best_bid, best_bid_qty
        prev_ask, prev_ask_qty = best_ask, best_ask_qty
        
        if idx % 20000 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(diffs)} diffs...")
    
    print(f"Processed {len(features)} valid snapshots.")
    return pd.DataFrame(features)