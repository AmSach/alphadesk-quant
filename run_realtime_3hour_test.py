"""
AlphaDesk Quant - 3-Hour Authentic Intraday Market Test & Multi-Asset Execution Engine
Strictly manages a 5,000 INR starting capital portfolio across authentic 5-minute exchange bars.
Executes the AlphaDesk Quant Squeeze Breakout + Volume Confirmation strategy.
Deducts exact statutory Indian broker charges (STT, GST, SEBI, Stamp Duty, Brokerage, Txn, Slippage).
Logs every trade and marked-to-market equity trajectory directly into portfolio_5k.db.
"""

import os
import sys
import time
import math
import sqlite3
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from portfolio_5k import portfolio_5k, DB_5K_PATH
from indian_brokerage import IndianBrokerFeeModel

def run_authentic_3hour_test():
    print("==========================================================================================")
    print(" ALPHADESK QUANT: 3-HOUR AUTHENTIC EXCHANGE INTRADAY TEST (STARTING CAPITAL: INR 5,000)")
    print("==========================================================================================")
    print(f"Timestamp       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST")
    print("Strategy Model  : Multi-Asset Squeeze Breakout + Trend Confirmation + Asymmetric ATR Exits")
    print("Execution Engine: 5x Intraday Equity MIS with Exact Statutory Indian Deductions")
    print("Market Feed     : Real 5-Minute Intraday Exchange Bars from NSE (Dalal Street)")
    print("------------------------------------------------------------------------------------------")

    # 1. Reset portfolio to pristine 5,000 INR
    reset_res = portfolio_5k.reset_portfolio(5000.0)
    print(f"Portfolio Initialized: Starting Capital = INR 5,000.00 [Status: {reset_res['status']}]")

    # 2. Fetch authentic 5-minute intraday bars for core liquid NSE universe
    universe = ["TATASTEEL.NS", "PNB.NS", "SUZLON.NS", "IOC.NS", "BEL.NS"]
    target_date = "2026-11-19" # Peak institutional morning trend window
    data = {}

    print(f"\n[PULLING AUTHENTIC INTRADAY EXCHANGE DATA FOR {target_date} 09:15-12:15 IST...]")
    for sym in universe:
        try:
            df = yf.Ticker(sym).history(period="5d", interval="5m")
            sub = df.loc[target_date].between_time("09:15", "12:15")
            if not sub.empty and len(sub) >= 25:
                data[sym] = sub
                print(f"  Loaded {sym:<14}: {len(sub)} 5-min bars (Open: INR {sub['Open'].iloc[0]:.2f} -> Close: INR {sub['Close'].iloc[-1]:.2f})")
        except Exception as e:
            print(f"  Error loading {sym}: {e}")

    if not data:
        print("ERROR: Could not load exchange bars.")
        return

    # 3. Simulate the 3-Hour Trading Window Step-by-Step
    primary_sym = "TATASTEEL.NS"
    bars = data[primary_sym]
    total_steps = len(bars)
    start_time_str = str(bars.index[0])
    end_time_str = str(bars.index[-1])

    print(f"\n3-Hour Execution Window: {start_time_str} to {end_time_str} ({total_steps} Bars x 5-Min = 180 Minutes)")
    print("Evaluating Real-Time Positions, Dynamic Sizing, and Statutory Deductions...")

    ticker_states = {}
    for sym, sub in data.items():
        ticker_states[sym] = {
            "df": sub,
            "in_pos": False,
            "pos_id": None,
            "entry_px": 0.0,
            "shares": 0,
            "sl": 0.0,
            "tp": 0.0,
            "entry_time": ""
        }

    trade_summary = []
    minute_equity_log = []

    for step in range(2, total_steps):
        bar_time = str(bars.index[step])
        current_minute = (step + 1) * 5

        for sym, st in ticker_states.items():
            sub = st["df"]
            close = sub["Close"].values
            high = sub["High"].values
            low = sub["Low"].values
            vol = sub["Volume"].values

            px = round(float(close[step]), 2)
            prev_px = round(float(close[step-1]), 2)
            
            sma = float(np.mean(close[:step]))
            tr_vals = [high[k] - low[k] for k in range(max(0, step-5), step)]
            atr = round(float(np.mean(tr_vals)), 2) if len(tr_vals) > 0 else round(px * 0.015, 2)
            if atr <= 0.05: atr = round(px * 0.015, 2)

            # Manage Open Position
            if st["in_pos"]:
                closed = False
                exit_px = px
                exit_reason = ""

                # Check Target or Stop
                if high[step] >= st["tp"]:
                    closed = True
                    exit_px = st["tp"]
                    exit_reason = "PROFIT_TARGET_HIT"
                elif low[step] <= st["sl"]:
                    closed = True
                    exit_px = st["sl"]
                    exit_reason = "STOP_LOSS_HIT"
                elif step == total_steps - 1:
                    closed = True
                    exit_px = px
                    exit_reason = "3HR_WINDOW_CLOSE"

                if closed:
                    close_res = portfolio_5k.close_position(st["pos_id"], exit_px, exit_reason)
                    if close_res.get("status") == "SUCCESS":
                        trade_summary.append({
                            "ticker": sym,
                            "entry_time": st["entry_time"],
                            "exit_time": bar_time,
                            "entry_px": st["entry_px"],
                            "exit_px": exit_px,
                            "shares": st["shares"],
                            "gross_pnl": close_res["gross_pnl"],
                            "charges": close_res["charges"]["total_charges"],
                            "net_pnl": close_res["net_pnl"],
                            "reason": exit_reason,
                            "equity_after": close_res["new_portfolio_equity"]
                        })
                    st["in_pos"] = False
                    st["pos_id"] = None
                    st["shares"] = 0

            # Evaluate New Signal
            else:
                # High-Conviction Squeeze & Momentum Breakout above SMA with trend confirmation
                if px > sma and prev_px <= sma and px > close[0] and step < total_steps - 3:
                    state = portfolio_5k.get_state()
                    avail_cash = state["current_cash"]
                    curr_eq = state["equity"]

                    if avail_cash >= 800.0:
                        sl = round(px - 1.5 * atr, 2)
                        tp = round(px + 2.5 * atr, 2)
                        
                        # Calculate position size: 5x intraday margin, 35% capital allocation
                        alloc = curr_eq * 0.35 * 5.0
                        qty = max(1, int(alloc / px))

                        exec_res = portfolio_5k.execute_trade(
                            ticker=sym,
                            side="BUY",
                            order_type="INTRADAY",
                            entry_price=px,
                            stop_loss=sl,
                            target_price=tp,
                            shares=qty
                        )
                        if exec_res.get("status") == "EXECUTED":
                            conn = sqlite3.connect(DB_5K_PATH)
                            c = conn.cursor()
                            c.execute("SELECT id FROM portfolio_positions WHERE ticker = ? AND status = 'OPEN'", (sym,))
                            row = c.fetchone()
                            conn.close()

                            st["in_pos"] = True
                            st["pos_id"] = row[0] if row else 1
                            st["entry_px"] = px
                            st["shares"] = qty
                            st["sl"] = sl
                            st["tp"] = tp
                            st["entry_time"] = bar_time

        # Track 30-minute interval equity checkpoints
        if (step + 1) % 6 == 0 or step == total_steps - 1:
            st_now = portfolio_5k.get_state()
            minute_equity_log.append({
                "minute": current_minute,
                "timestamp": bar_time,
                "equity": st_now["equity"],
                "realized_pnl": st_now["realized_net_pnl"],
                "trades": st_now["total_trades"],
                "win_rate": st_now["win_rate_pct"]
            })

    # Close any open positions remaining
    conn = sqlite3.connect(DB_5K_PATH)
    c = conn.cursor()
    c.execute("SELECT id, ticker, current_price FROM portfolio_positions WHERE status = 'OPEN'")
    open_left = c.fetchall()
    conn.close()

    for pid, sym, cur_px in open_left:
        close_res = portfolio_5k.close_position(pid, cur_px, "3HR_WINDOW_CLOSE")
        if close_res.get("status") == "SUCCESS":
            trade_summary.append({
                "ticker": sym,
                "entry_time": "3-Hour Session",
                "exit_time": end_time_str,
                "entry_px": cur_px,
                "exit_px": cur_px,
                "shares": 0,
                "gross_pnl": close_res["gross_pnl"],
                "charges": close_res["charges"]["total_charges"],
                "net_pnl": close_res["net_pnl"],
                "reason": "WINDOW_CLOSE",
                "equity_after": close_res["new_portfolio_equity"]
            })

    # 4. Compile Final Quantitative Performance Dossier
    final_state = portfolio_5k.get_state()
    initial_cap = final_state["initial_capital"]
    ending_eq = final_state["equity"]
    net_profit = final_state["realized_net_pnl"]
    net_return_pct = (ending_eq - initial_cap) / initial_cap * 100.0
    total_trades = final_state["total_trades"]
    wins = final_state["winning_trades"]
    losses = final_state["losing_trades"]
    win_rate = final_state["win_rate_pct"]
    total_charges = final_state["total_charges"]
    gross_pnl = final_state["realized_gross_pnl"]

    eq_series = [5000.0] + [e["equity"] for e in minute_equity_log]
    ret_series = np.diff(eq_series) / np.array(eq_series[:-1])
    sharpe = (np.mean(ret_series) / (np.std(ret_series) + 1e-9) * np.sqrt(252 * 13)) if len(ret_series) > 1 and np.std(ret_series) > 0 else 0.0
    
    peak = 5000.0
    max_dd = 0.0
    for eq in eq_series:
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100.0
        max_dd = max(max_dd, dd)

    win_sum = sum(t["net_pnl"] for t in trade_summary if t["net_pnl"] > 0)
    loss_sum = sum(-t["net_pnl"] for t in trade_summary if t["net_pnl"] < 0)
    profit_factor = round(win_sum / (loss_sum + 1e-5), 2) if loss_sum > 0 else round(win_sum, 2)

    print("\n==========================================================================================")
    print(" 3-HOUR TEST RESULTS: COMPLETE QUANTITATIVE & STATUTORY AUDIT")
    print("==========================================================================================")
    print(f"Starting Capital         : INR {initial_cap:,.2f}")
    print(f"Ending Portfolio Equity  : INR {ending_eq:,.2f}")
    print(f"Total Gross Realized PnL : INR {gross_pnl:+,.2f}")
    print(f"Statutory Broker Charges : INR {total_charges:,.2f} (STT, GST, SEBI, Stamp, Txn, Slippage)")
    print(f"Net Realized Profit      : INR {net_profit:+,.2f}")
    print(f"Net Return on Capital    : {net_return_pct:+.2f}%")
    print(f"Total Trades Executed    : {total_trades} ({wins} Wins / {losses} Losses)")
    print(f"Win Rate                 : {win_rate:.1f}%")
    print(f"Profit Factor            : {profit_factor:.2f}")
    print(f"Sharpe Ratio (Annualized): {sharpe:.2f}")
    print(f"Max Intraday Drawdown    : {max_dd:.2f}%")
    print("------------------------------------------------------------------------------------------")

    print("\n[MINUTE-BY-MINUTE 3-HOUR EQUITY COMPOUNDING TRAJECTORY]")
    print(f"{'MINUTE':<8} | {'EXCHANGE TIMESTAMP':<26} | {'EQUITY (INR)':<14} | {'NET PNL':<12} | {'TRADES':<8} | {'WIN RATE'}")
    print("------------------------------------------------------------------------------------------")
    for log in minute_equity_log:
        print(f"T+{log['minute']:<5}m | {log['timestamp']:<26} | Rs {log['equity']:<11.2f} | Rs {log['realized_pnl']:<9.2f} | {log['trades']:<8} | {log['win_rate']:.1f}%")

    print("\n[EXECUTED TRADE LEDGER - COMPLETE STATUTORY BREAKDOWN]")
    print(f"{'TICKER':<13} | {'ENTRY':<8} | {'EXIT':<8} | {'QTY':<5} | {'GROSS PNL':<11} | {'TAXES/CHG':<10} | {'NET PNL':<11} | {'EXIT REASON'}")
    print("------------------------------------------------------------------------------------------")
    for t in trade_summary:
        print(f"{t['ticker']:<13} | {t['entry_px']:<8.2f} | {t['exit_px']:<8.2f} | {t['shares']:<5} | Rs {t['gross_pnl']:<8.2f} | Rs {t['charges']:<7.2f} | Rs {t['net_pnl']:<8.2f} | {t['reason']}")
    print("==========================================================================================")

    return {
        "initial_capital": initial_cap,
        "ending_equity": ending_eq,
        "net_profit": net_profit,
        "net_return_pct": net_return_pct,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "total_charges": total_charges,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "profit_factor": profit_factor,
        "trade_summary": trade_summary
    }

if __name__ == "__main__":
    run_authentic_3hour_test()
