"""
Test and optimize the trading strategy for 5K INR portfolio with full Indian broker charges.
"""
import yfinance as yf
import numpy as np
from indian_brokerage import IndianBrokerFeeModel

def backtest_ticker(ticker, period="1y", capital=5000.0, risk_pct=1.5):
    t_obj = yf.Ticker(ticker)
    h = t_obj.history(period=period)
    if h is None or len(h) < 60:
        return None

    close = h["Close"].values
    high = h["High"].values
    low = h["Low"].values
    vol = h["Volume"].values
    dates = [str(d.date()) for d in h.index]
    n = len(close)

    curr_capital = capital
    in_pos = False
    pos_entry = 0.0
    pos_stop = 0.0
    pos_target = 0.0
    pos_shares = 0
    trailing_stop = 0.0
    entry_idx = 0

    wins = 0
    losses = 0
    trades = []
    total_charges = 0.0

    for i in range(30, n):
        curr_px = close[i]
        prev_px = close[i-1]

        sma20 = np.mean(close[max(0, i-20):i])
        sma50 = np.mean(close[max(0, i-50):i])
        sma200 = np.mean(close[max(0, i-200):i]) if i >= 60 else sma50 * 0.95
        atr = np.mean(high[max(0, i-14):i] - low[max(0, i-14):i])
        if atr <= 0.0: atr = curr_px * 0.025

        adv20 = np.mean(vol[max(0, i-20):i]) if i > 20 else vol[i]
        surge = vol[i] / adv20 if adv20 > 0 else 1.0

        # RSI
        diffs = np.diff(close[max(0, i-15):i])
        gains = np.sum(diffs[diffs > 0])
        losses_sum = np.sum(-diffs[diffs < 0])
        rsi = 50.0 if (gains + losses_sum) == 0 else (100.0 * gains / (gains + losses_sum))

        # Evaluate open position
        if in_pos:
            bars_held = i - entry_idx
            closed = False
            exit_px = curr_px
            exit_reason = ""

            # Ratchet trailing stop up
            if curr_px > pos_entry + 1.2 * atr:
                trailing_stop = max(trailing_stop, curr_px - 1.4 * atr)

            effective_stop = max(pos_stop, trailing_stop)

            if high[i] >= pos_target:
                exit_px = pos_target
                closed = True
                wins += 1
                exit_reason = "TARGET_HIT (+3R)"
            elif low[i] <= effective_stop:
                exit_px = effective_stop
                closed = True
                exit_reason = "TRAILING_STOP" if effective_stop > pos_stop else "STOP_LOSS (-1R)"
                if exit_px > pos_entry: wins += 1
                else: losses += 1
            elif bars_held >= 20: # Time expiry
                exit_px = curr_px
                closed = True
                exit_reason = "TIME_EXPIRY"
                if exit_px > pos_entry: wins += 1
                else: losses += 1

            if closed:
                charges = IndianBrokerFeeModel.calculate_charges(
                    entry_price=pos_entry,
                    exit_price=exit_px,
                    quantity=pos_shares,
                    order_type="INTRADAY"
                )
                net_pnl = charges["net_pnl"]
                total_charges += charges["total_charges"]
                curr_capital += net_pnl
                in_pos = False

                trades.append({
                    "entry_date": dates[entry_idx],
                    "exit_date": dates[i],
                    "entry": round(pos_entry, 2),
                    "exit": round(exit_px, 2),
                    "shares": pos_shares,
                    "gross_pnl": charges["gross_pnl"],
                    "charges": charges["total_charges"],
                    "net_pnl": net_pnl,
                    "exit_reason": exit_reason
                })

        # Entry signal (High precision momentum ignition)
        if not in_pos and i < n - 3:
            # Bullish trend alignment
            bull_regime = curr_px > sma50 and (sma20 >= sma50 or curr_px > sma200)
            breakout = curr_px > sma20 and prev_px <= sma20 and rsi > 50.0 and rsi < 70.0
            pullback_bounce = curr_px > sma20 and low[i-1] <= sma20 and curr_px > prev_px and rsi > 46.0 and rsi < 65.0
            volume_confirmed = surge > 1.1 or rsi > 54.0

            if bull_regime and (breakout or pullback_bounce) and volume_confirmed:
                in_pos = True
                pos_entry = curr_px
                pos_stop = max(curr_px - 1.5 * atr, curr_px * 0.92)
                pos_target = curr_px + 3.2 * atr
                trailing_stop = pos_stop
                entry_idx = i

                risk_budget = curr_capital * (risk_pct / 100.0)
                risk_per_sh = abs(pos_entry - pos_stop)
                max_affordable = max(1, int(curr_capital * 0.90 / pos_entry))
                pos_shares = max(1, min(max_affordable, int(risk_budget / risk_per_sh)))

    ret_pct = ((curr_capital - capital) / capital) * 100.0
    win_rate = (wins / (wins + losses) * 100.0) if (wins + losses) > 0 else 0.0

    return {
        "ticker": ticker,
        "initial": capital,
        "ending": round(curr_capital, 2),
        "return_pct": round(ret_pct, 2),
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "total_charges": round(total_charges, 2),
        "trades_sample": trades[-5:]
    }

if __name__ == "__main__":
    for sym in ["SUZLON.NS", "IREDA.NS", "CDSL.NS", "BSE.NS", "KAYNES.NS", "TATASTEEL.NS", "ZOMATO.NS"]:
        res = backtest_ticker(sym, period="1y", capital=5000.0)
        if res:
            print(f"{sym:12} | Ending: Rs {res['ending']:7.2f} | Net Ret: {res['return_pct']:+6.2f}% | Trades: {res['trades']:2d} (W:{res['wins']} L:{res['losses']}) | WinRate: {res['win_rate']:4.1f}% | Taxes: Rs {res['total_charges']:5.2f}")
