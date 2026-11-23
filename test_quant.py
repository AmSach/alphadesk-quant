import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import quant
import options
import db
import data
import alerts
import insiders
from dotenv import load_dotenv

load_dotenv()

def run_tests():
    print("=== Testing Options Pricing & Greeks ===")
    c_price = options.bs(100, 100, 1, 0.05, 0.2, "c")
    p_price = options.bs(100, 100, 1, 0.05, 0.2, "p")
    # Put-call parity: C - P = S - K * exp(-r*T)
    parity_diff = abs((c_price - p_price) - (100 - 100 * np.exp(-0.05)))
    print(f"Call price: {c_price:.4f}, Put price: {p_price:.4f}, Parity diff: {parity_diff:.6f}")
    assert parity_diff < 1e-4, "Put-call parity failed!"

    g = options.greeks(100, 100, 1, 0.05, 0.2, "c")
    print(f"Call delta: {g['delta']:.4f}, gamma: {g['gamma']:.4f}, vega: {g['vega']:.4f}, theta: {g['theta']:.4f}")

    recovered_iv = options.iv(c_price, 100, 100, 1, 0.05, "c")
    print(f"Recovered IV: {recovered_iv:.4f} (target 0.2000)")
    assert abs(recovered_iv - 0.2) < 1e-4, "IV recovery failed!"

    rn_prob = options.rn_prob_above(100, 105, 1, 0.05, 0.2)
    print(f"Risk-neutral P(S_T > 105): {rn_prob:.4f}")

    print("\n=== Testing Quant Algorithms ===")
    p_touch = quant.touch(100, 110, 0.015, 10)
    print(f"GBM Barrier Touch prob (100 -> 110 in 10d, 1.5% daily vol): {p_touch:.4f}")

    dates = pd.date_range("2020-01-01", periods=900, freq="B")
    np.random.seed(42)
    ret = np.random.normal(0.0005, 0.015, size=900)
    prices = 100 * np.exp(np.cumsum(ret))
    df_synth = pd.DataFrame({
        "Open": prices * 0.99,
        "High": prices * 1.01,
        "Low": prices * 0.98,
        "Close": prices,
        "Volume": 1000000
    }, index=dates)

    cleaned = quant.clean(df_synth)
    assert cleaned is not None, "Clean failed on synthetic dataframe"

    rsi_series = quant.rsi(df_synth.Close)
    print(f"RSI latest: {rsi_series.iloc[-1]:.2f}")

    atr_series = quant.atr(df_synth)
    print(f"ATR latest: {atr_series.iloc[-1]:.2f}")

    lv = quant.levels(df_synth, capital=100000, risk_pct=1.0, horizon=10)
    assert lv is not None, "quant.levels returned None"
    print(f"Levels: Entry={lv['entry']:.2f}, Stop={lv['stop']:.2f}, Target={lv['target']:.2f}")
    print(f"Odds: P(win)={lv['p_win']:.1%}, P(loss)={lv['p_loss']:.1%}, Exp_R={lv['exp_R']:+.2f}R, Qty={lv['qty']}")

    print("\n=== Testing Walk-Forward ML Engine ===")
    wf = quant.walk_forward(df_synth, horizon=5, min_train=300, step=21)
    print("Walk-forward status:", wf.get("ok"), "trusted:", wf.get("trusted"), "trades:", wf.get("n_trades"))

    print("\n=== Testing Database (SQLite) ===")
    db.save("test_table", {"col1": "AlphaDesk_Test", "ts": 12345678})
    loaded = db.load("test_table")
    assert len(loaded) > 0, "Database save/load failed"
    print(f"Loaded {len(loaded)} row(s) from test_table")

    print("\n=== Testing Data Normalization & SEC CIK ===")
    assert data.norm("AAPL") == "AAPL"
    assert data.norm("NSE:RELIANCE") == "RELIANCE.NS"
    assert data.norm("BSE:500325") == "500325.BO"
    assert data.is_us("AAPL") == True
    assert data.is_us("RELIANCE.NS") == False
    assert data.ccy("AAPL") == "local/USD"
    assert data.ccy("RELIANCE.NS") == "INR"
    print("Data ticker normalization and currency mappings verified.")

    print("\nALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
