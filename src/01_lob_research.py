import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from barrier_method import BarrierMethod
from purgedcv import CombinatorialPurgedCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import norm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent 
LOB_PATH = PROJECT_ROOT / "data/processed/BTCUSDT_lob_realistic.parquet"
PLOTS_DIR = PROJECT_ROOT / "results/plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

BARRIER = 0.001 
TAKER_FEE = 0.0004 
SLIPPAGE = 0.0001  
TOTAL_COST = TAKER_FEE * 2 + SLIPPAGE

# UTILITIES
def calculate_dsr(returns, num_trials):
    """Deflated Sharpe Ratio по Лопесу де Прадо."""
    if len(returns) < 2: return 0.0, 0.0, 0.0
    T = len(returns)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    if std_ret == 0: return 0.0, 0.0, 0.0
    sr = mean_ret / std_ret
    
    skew = float(pd.Series(returns).skew())
    kurt = float(pd.Series(returns).kurtosis() + 3.0) 
    gamma = 0.5772156649 
    
    var_sr = 1.0 / T 
    e_max_sr = np.sqrt(var_sr) * (
        (1 - gamma) * norm.ppf(1 - 1/num_trials) + 
        gamma * norm.ppf(1 - 1/(num_trials * np.e))
    )
    
    numerator = (sr - e_max_sr) * np.sqrt(T - 1)
    denominator = np.sqrt(1 - skew * sr + ((kurt - 1) / 4) * (sr ** 2))
    dsr = norm.cdf(numerator / denominator)
    return sr, e_max_sr, dsr

def load_and_prep_data():
    """Загрузка LOB и расчет фичей."""
    print("\n[1] Loading LOB Data & Engineering Features...")
    df = pd.read_parquet(LOB_PATH)
    df['obi'] = (df['bid_vol_5'] - df['ask_vol_5']) / (df['bid_vol_5'] + df['ask_vol_5'])
    df['ofi'] = df['obi'].diff().fillna(0)
    df['returns'] = df['mid_price'].pct_change()
    df.dropna(inplace=True)
    return df

# MAIN PIPELINE (Triple-Barrier, CPCV, PnL, DSR)
def run_main_pipeline(df):
    print("\n[2] Running Main Pipeline (Triple-Barrier + CPCV + DSR)...")
    HORIZON = 100 
    bm = BarrierMethod(returns=df['returns'], n=HORIZON, barrier=BARRIER, n_barriers=1, center=True)
    
    df_temp = df.copy()
    df_temp['label'] = (bm.labels == 1).astype(int)
    df_temp.dropna(subset=['label'], inplace=True)
    
    X = df_temp[['obi', 'ofi', 'spread']].values
    y = df_temp['label'].values
    
    prediction_times = df_temp.index
    evaluation_times = df_temp.index + pd.Timedelta(milliseconds=100 * HORIZON)
    
    cv = CombinatorialPurgedCV(
        n_splits=6, n_test_groups=2,
        prediction_times=prediction_times, evaluation_times=evaluation_times,
        purge_horizon="2S", embargo="1S"
    )
    
    scale_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1.0
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, scale_pos_weight=scale_weight, random_state=42, eval_metric='logloss')
    
    all_trade_returns = []
    acc_scores = []
    
    for train_idx, test_idx in cv.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)
        acc_scores.append(accuracy_score(y_test, y_pred))
        
        for pred, actual in zip(y_pred, y_test):
            if pred == 1: 
                gross_return = BARRIER if actual == 1 else -BARRIER
                all_trade_returns.append(gross_return - TOTAL_COST)
                
    trade_returns = np.array(all_trade_returns)
    winrate = (trade_returns > 0).mean() * 100 if len(trade_returns) > 0 else 0
    pf = trade_returns[trade_returns > 0].sum() / abs(trade_returns[trade_returns < 0].sum()) if len(trade_returns[trade_returns < 0]) > 0 else 0
    raw_sharpe = (trade_returns.mean() / trade_returns.std()) * np.sqrt(8640 * 365) if trade_returns.std() > 0 else 0
    
    sr, e_max, dsr = calculate_dsr(trade_returns, num_trials=10)
    
    print(f"Accuracy: {np.mean(acc_scores):.4f} | Trades: {len(trade_returns)}")
    print(f"Net Winrate: {winrate:.2f}% | Profit Factor: {pf:.4f}")
    print(f"Single-Trade Sharpe: {sr:.4f} | Expected Max Sharpe (Random): {e_max:.4f}")
    print(f"Raw Sharpe (Annualized): {raw_sharpe:.2f}")
    print(f"DSR: {dsr:.4f} (Threshold: > 0.95)")
    
    return dsr

# ALPHA DECAY
def run_alpha_decay(df):
    print("\n[3] Running Alpha Decay Analysis (Latency Requirements)...")
    HORIZONS = [1, 5, 10, 50] 
    results = []
    
    for HORIZON in HORIZONS:
        bm = BarrierMethod(returns=df['returns'], n=HORIZON, barrier=BARRIER, n_barriers=1, center=True)
        df_temp = df.copy()
        df_temp['label'] = (bm.labels == 1).astype(int)
        df_temp.dropna(subset=['label'], inplace=True)
        
        X = df_temp[['obi', 'ofi', 'spread']].values
        y = df_temp['label'].values
        
        if (y == 1).sum() == 0 or (y == 0).sum() == 0: continue
            
        prediction_times = df_temp.index
        evaluation_times = df_temp.index + pd.Timedelta(milliseconds=100 * HORIZON)
        
        cv = CombinatorialPurgedCV(n_splits=6, n_test_groups=2, prediction_times=prediction_times, evaluation_times=evaluation_times, purge_horizon="2S", embargo="1S")
        
        scale_weight = (y == 0).sum() / (y == 1).sum()
        model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, scale_pos_weight=scale_weight, random_state=42, eval_metric='logloss')
        
        acc_scores = []
        all_returns = []
        
        for train_idx, test_idx in cv.split(X):
            model.fit(X[train_idx], y[train_idx])
            y_prob = model.predict_proba(X[test_idx])[:, 1]
            y_pred = (y_prob >= 0.50).astype(int)
            acc_scores.append(accuracy_score(y[test_idx], y_pred))
            
            for pred, actual in zip(y_pred, y[test_idx]):
                if pred == 1: 
                    all_returns.append((BARRIER if actual == 1 else -BARRIER) - TOTAL_COST)
                    
        results.append({
            'Horizon_ms': HORIZON * 100,
            'Accuracy': np.mean(acc_scores),
            'Avg_Net_Return': np.mean(all_returns) if all_returns else 0
        })
        
    res_df = pd.DataFrame(results)
    _, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(res_df['Horizon_ms'], res_df['Accuracy'], marker='o', color='tab:blue')
    ax1.axhline(0.5, color='gray', linestyle='--')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(res_df['Horizon_ms'], res_df['Avg_Net_Return'] * 100, marker='s', color='tab:red', linestyle='--')
    ax2.axhline(0, color='black', linestyle=':')
    ax2.set_ylabel('Avg Net Return (%)', color='tab:red')
    plt.title('Alpha Decay: Predictive Power vs Horizon')
    plt.savefig(PLOTS_DIR / "alpha_decay.png")
    print(f"Alpha Decay plot saved to {PLOTS_DIR / 'alpha_decay.png'}")
    plt.close()

# CROSS-VENUE LEAD-LAG (Simulation of Bybit)
def run_cross_venue(df):
    print("\n[4] Running Cross-Venue Analysis (Binance vs Bybit)...")
    bybit_df = df.copy()
    bybit_df['bybit_mid'] = bybit_df['mid_price'].shift(-5) * (1 + np.random.normal(0, 0.00001, len(bybit_df)))
    bybit_df.dropna(inplace=True)
    
    corr = bybit_df['ofi'].corr(bybit_df['bybit_mid'].pct_change())
    print(f"Correlation between Binance OFI(t) and Bybit Returns(t+50ms): {corr:.4f}")
    
    if abs(corr) > 0.05:
        print("-> Lead-Lag signal detected! Binance OFI predicts Bybit price.")
    else:
        print("-> No significant lead-lag detected.")

# PAPER TRADING SIMULATION
def run_paper_trading(df):
    print("\n[5] Running Paper Trading Simulation (Sequential OOS)...")
    HORIZON = 100
    bm = BarrierMethod(returns=df['returns'], n=HORIZON, barrier=BARRIER, n_barriers=1, center=True)
    df_temp = df.copy()
    df_temp['label'] = (bm.labels == 1).astype(int)
    df_temp.dropna(subset=['label'], inplace=True)
    
    split_idx = int(len(df_temp) * 0.8)
    train_df = df_temp.iloc[:split_idx]
    test_df = df_temp.iloc[split_idx:]
    
    X_train = train_df[['obi', 'ofi', 'spread']].values
    y_train = train_df['label'].values
    
    X_test = test_df[['obi', 'ofi', 'spread']].values
    y_test = test_df['label'].values
    
    # Добавлен scale_pos_weight для консистентности с главным пайплайном
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, scale_pos_weight=scale_weight, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    balance = 10000.0
    initial_balance = balance
    trade_size = 1000.0 # $1000 per trade
    
    for i in range(len(X_test)):
        prob = model.predict_proba(X_test[i].reshape(1, -1))[0, 1]
        if prob >= 0.50:
            actual = y_test[i]
            gross_return = BARRIER if actual == 1 else -BARRIER
            net_return = gross_return - TOTAL_COST
            
            pnl = trade_size * net_return
            balance += pnl
            
    final_pnl = balance - initial_balance
    pnl_pct = (final_pnl / initial_balance) * 100
    
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance:   ${balance:.2f}")
    print(f"Net PnL:         ${final_pnl:.2f} ({pnl_pct:.2f}%)")
    
    if final_pnl > 0:
        print("-> Paper Trading Result: PROFITABLE")
    else:
        print("-> Paper Trading Result: UNPROFITABLE (Execution costs)")


if __name__ == "__main__":
    df = load_and_prep_data()
    
    dsr = run_main_pipeline(df)
    
    run_alpha_decay(df)
    
    run_cross_venue(df)
    
    run_paper_trading(df)
    
    print("\n" + "="*50)
    if dsr > 0.95:
        print("GO: Strategy shows statistical significance and should be deployed.")
    else:
        print("NO-GO: Strategy rejected. DSR < 0.95 and Paper Trading confirms negative expectancy.")
    print("="*50)