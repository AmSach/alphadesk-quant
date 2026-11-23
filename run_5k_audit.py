"""
AlphaDesk Quant - 5K INR Portfolio Audit on Liquid Sub-400 INR Equities
"""
from autonomous_trader import trader_engine

tickers = ["TATASTEEL.NS", "IRFC.NS", "IOC.NS", "PNB.NS", "IDFCFIRSTB.NS", "GAIL.NS", "POWERGRID.NS", "BEL.NS", "NHPC.NS"]

print("=========================================================================================")
print(" ALPHADESK QUANT: 5,000 INR OPTIMAL UNIVERSE (SUB-400 INR NSE EQUITIES)")
print("=========================================================================================")
print(f"{'TICKER':14} | {'INITIAL':8} | {'ENDING':9} | {'NET RETURN':10} | {'WIN RATE':8} | {'TRADES':6} | {'CHARGES':9} | {'SHARPE':6}")
print("-----------------------------------------------------------------------------------------")

for sym in tickers:
    res = trader_engine.run_multiyear_backtest(sym, period="1y", capital=5000.0, risk_pct=1.5, order_type="INTRADAY")
    if "error" in res:
        print(f"{sym:14} | ERROR: {res['error']}")
        continue
    ret_str = f"{res['net_return_pct']:+.1f}%"
    print(f"{sym:14} | Rs {res['initial_capital']:<5.0f} | Rs {res['ending_capital']:<7.2f} | {ret_str:10} | {res['win_rate_pct']:4.1f}%   | {res['total_trades']:<6} | Rs {res['total_broker_charges']:<7.2f} | {res['sharpe_ratio']:<6.2f}")

print("=========================================================================================")
