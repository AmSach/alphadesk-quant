"""
AlphaDesk Quant - 5K INR Active Portfolio Manager & Execution Engine
Strictly manages a 5,000 INR starting capital portfolio.
Applies exact Indian statutory charges (STT, Brokerage, GST, Stamp Duty, Exchange Txn, SEBI, Slippage).
Executes simulated and walk-forward paper trades, tracks marked-to-market equity, and logs every paisa.
"""

import os
import sys
import sqlite3
import math
from datetime import datetime
from typing import Dict, List, Any, Optional

from indian_brokerage import IndianBrokerFeeModel
from fast_ticker import fast_ticker
from deep_quant_net import neural_trader

DB_5K_PATH = "E:/Alphadesk Quant/portfolio_5k.db"
STARTING_CAPITAL = 5000.0

class Portfolio5KManager:
    def __init__(self, db_path: str = DB_5K_PATH):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # 1. Portfolio Account State
        cur.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            initial_capital REAL,
            current_cash REAL,
            allocated_margin REAL,
            realized_gross_pnl REAL,
            total_brokerage REAL,
            total_stt REAL,
            total_exchange_txn REAL,
            total_sebi REAL,
            total_stamp_duty REAL,
            total_gst REAL,
            total_dp_charges REAL,
            total_slippage REAL,
            total_charges REAL,
            realized_net_pnl REAL,
            equity REAL,
            total_trades INTEGER,
            winning_trades INTEGER,
            losing_trades INTEGER,
            win_rate_pct REAL,
            updated_at TEXT
        )
        """)

        # 2. Active Open Positions
        cur.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            exchange TEXT,
            order_type TEXT,
            side TEXT,
            entry_timestamp TEXT,
            entry_price REAL,
            stop_loss REAL,
            target_price REAL,
            trailing_stop REAL,
            quantity INTEGER,
            allocated_cash REAL,
            current_price REAL,
            unrealized_gross_pnl REAL,
            estimated_charges REAL,
            unrealized_net_pnl REAL,
            status TEXT
        )
        """)

        # 3. Itemized Trade Execution Ledger
        cur.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_trade_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            side TEXT,
            order_type TEXT,
            entry_time TEXT,
            exit_time TEXT,
            entry_px REAL,
            exit_px REAL,
            qty INTEGER,
            gross_pnl REAL,
            brokerage REAL,
            stt REAL,
            exchange_txn REAL,
            sebi_charges REAL,
            stamp_duty REAL,
            gst REAL,
            dp_charges REAL,
            slippage REAL,
            total_charges REAL,
            net_pnl REAL,
            r_multiple REAL,
            exit_reason TEXT,
            portfolio_equity_after REAL
        )
        """)

        # Initialize State if missing
        cur.execute("SELECT COUNT(*) FROM portfolio_state WHERE id = 1")
        if cur.fetchone()[0] == 0:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cur.execute("""
            INSERT INTO portfolio_state (
                id, initial_capital, current_cash, allocated_margin,
                realized_gross_pnl, total_brokerage, total_stt, total_exchange_txn,
                total_sebi, total_stamp_duty, total_gst, total_dp_charges, total_slippage,
                total_charges, realized_net_pnl, equity, total_trades,
                winning_trades, losing_trades, win_rate_pct, updated_at
            ) VALUES (
                1, ?, ?, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, ?, 0,
                0, 0, 0.0, ?
            )
            """, (STARTING_CAPITAL, STARTING_CAPITAL, STARTING_CAPITAL, now_str))

        conn.commit()
        conn.close()

    def reset_portfolio(self, initial_capital: float = STARTING_CAPITAL):
        """Resets the portfolio state to initial capital and clears positions/ledger."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cur.execute("DELETE FROM portfolio_positions")
        cur.execute("DELETE FROM portfolio_trade_ledger")
        cur.execute("""
        UPDATE portfolio_state SET
            initial_capital = ?,
            current_cash = ?,
            allocated_margin = 0.0,
            realized_gross_pnl = 0.0,
            total_brokerage = 0.0,
            total_stt = 0.0,
            total_exchange_txn = 0.0,
            total_sebi = 0.0,
            total_stamp_duty = 0.0,
            total_gst = 0.0,
            total_dp_charges = 0.0,
            total_slippage = 0.0,
            total_charges = 0.0,
            realized_net_pnl = 0.0,
            equity = ?,
            total_trades = 0,
            winning_trades = 0,
            losing_trades = 0,
            win_rate_pct = 0.0,
            updated_at = ?
        WHERE id = 1
        """, (initial_capital, initial_capital, initial_capital, now_str))
        conn.commit()
        conn.close()
        return {"status": "SUCCESS", "message": f"Portfolio reset to INR {initial_capital:.2f}"}

    def get_state(self) -> Dict[str, Any]:
        """Returns the current portfolio snapshot and marked-to-market totals."""
        self.update_market_prices()
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("SELECT * FROM portfolio_state WHERE id = 1")
        row = cur.fetchone()

        # Fetch Open Positions
        cur.execute("""
        SELECT id, ticker, exchange, order_type, side, entry_timestamp,
               entry_price, stop_loss, target_price, trailing_stop, quantity,
               allocated_cash, current_price, unrealized_gross_pnl,
               estimated_charges, unrealized_net_pnl, status
        FROM portfolio_positions WHERE status = 'OPEN' ORDER BY id DESC
        """)
        open_pos_rows = cur.fetchall()

        # Fetch Recent Closed Trades (last 20)
        cur.execute("""
        SELECT id, ticker, side, order_type, entry_time, exit_time,
               entry_px, exit_px, qty, gross_pnl, brokerage, stt,
               exchange_txn, sebi_charges, stamp_duty, gst, dp_charges,
               slippage, total_charges, net_pnl, r_multiple, exit_reason,
               portfolio_equity_after
        FROM portfolio_trade_ledger ORDER BY id DESC LIMIT 20
        """)
        trade_rows = cur.fetchall()
        conn.close()

        open_positions = []
        unrealized_net_total = 0.0
        allocated_total = 0.0

        for r in open_pos_rows:
            pos_dict = {
                "id": r[0],
                "ticker": r[1],
                "exchange": r[2],
                "order_type": r[3],
                "side": r[4],
                "entry_timestamp": r[5],
                "entry_price": r[6],
                "stop_loss": r[7],
                "target_price": r[8],
                "trailing_stop": r[9],
                "quantity": r[10],
                "allocated_cash": r[11],
                "current_price": r[12],
                "unrealized_gross_pnl": r[13],
                "estimated_charges": r[14],
                "unrealized_net_pnl": r[15],
                "status": r[16]
            }
            open_positions.append(pos_dict)
            unrealized_net_total += r[15]
            allocated_total += r[11]

        trades = []
        for t in trade_rows:
            trades.append({
                "id": t[0],
                "ticker": t[1],
                "side": t[2],
                "order_type": t[3],
                "entry_time": t[4],
                "exit_time": t[5],
                "entry_px": t[6],
                "exit_px": t[7],
                "qty": t[8],
                "gross_pnl": t[9],
                "brokerage": t[10],
                "stt": t[11],
                "exchange_txn": t[12],
                "sebi_charges": t[13],
                "stamp_duty": t[14],
                "gst": t[15],
                "dp_charges": t[16],
                "slippage": t[17],
                "total_charges": t[18],
                "net_pnl": t[19],
                "r_multiple": t[20],
                "exit_reason": t[21],
                "portfolio_equity_after": t[22]
            })

        current_cash = row[2]
        realized_net = row[14]
        initial_cap = row[1]
        total_equity = round(current_cash + allocated_total + unrealized_net_total, 2)
        total_net_pnl = round(total_equity - initial_cap, 2)
        total_return_pct = round((total_net_pnl / initial_cap) * 100.0, 2) if initial_cap > 0 else 0.0

        return {
            "initial_capital": initial_cap,
            "current_cash": round(current_cash, 2),
            "allocated_margin": round(allocated_total, 2),
            "unrealized_net_pnl": round(unrealized_net_total, 2),
            "realized_gross_pnl": round(row[4], 2),
            "realized_net_pnl": round(realized_net, 2),
            "total_brokerage": round(row[5], 2),
            "total_stt": round(row[6], 2),
            "total_exchange_txn": round(row[7], 2),
            "total_sebi": round(row[8], 2),
            "total_stamp_duty": round(row[9], 2),
            "total_gst": round(row[10], 2),
            "total_dp_charges": round(row[11], 2),
            "total_slippage": round(row[12], 2),
            "total_charges": round(row[13], 2),
            "equity": total_equity,
            "total_net_pnl": total_net_pnl,
            "total_return_pct": total_return_pct,
            "total_trades": row[16],
            "winning_trades": row[17],
            "losing_trades": row[18],
            "win_rate_pct": round(row[19], 1),
            "open_positions": open_positions,
            "recent_trades": trades,
            "updated_at": row[20]
        }

    def can_open_position(self, required_cash: float) -> bool:
        """Enforces 5K INR strict capital discipline."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT current_cash FROM portfolio_state WHERE id = 1")
        cash = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM portfolio_positions WHERE status = 'OPEN'")
        open_cnt = cur.fetchone()[0]
        conn.close()

        # Allow max 2 concurrent positions in 5K account to prevent overexposure
        if open_cnt >= 2:
            return False
        if cash < required_cash or cash < 500.0:
            return False
        return True

    def calculate_position_size(self, ticker: str, entry_px: float, stop_loss: float, risk_pct: float = 1.5) -> int:
        """
        Calculates exact shares for a 5K INR account.
        Risk is capped at 1.5% of 5,000 INR (INR 75 max risk).
        Total position cost cannot exceed 60% of available cash (INR 3,000 max single stock allocation).
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT current_cash, equity FROM portfolio_state WHERE id = 1")
        row = cur.fetchone()
        cash = row[0]
        equity = row[1]
        conn.close()

        risk_budget = equity * (risk_pct / 100.0) # Rs 75 on 5k
        per_share_risk = abs(entry_px - stop_loss)
        if per_share_risk <= 0:
            per_share_risk = entry_px * 0.03

        # Sizing by risk budget
        qty_by_risk = int(risk_budget / per_share_risk)
        
        # Sizing by cash availability (max 60% of cash in single asset)
        max_capital_for_trade = min(cash * 0.60, 3000.0)
        qty_by_cash = int(max_capital_for_trade / entry_px)

        final_qty = max(1, min(qty_by_risk, qty_by_cash))
        return final_qty

    def execute_trade(
        self,
        ticker: str,
        side: str = "BUY",
        order_type: str = "INTRADAY",
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        target_price: Optional[float] = None,
        shares: Optional[int] = None,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Opens a new trade position in the 5K INR portfolio.
        Deducts allocated cash and registers stop-loss / target.
        """
        tick = fast_ticker.get_tick(ticker)
        current_px = tick["last_price"] if tick else 65.0
        px = entry_price if entry_price and entry_price > 0 else current_px

        # Determine Stop & Target if not passed
        atr = round(px * 0.03, 2)
        sl = stop_loss if stop_loss and stop_loss > 0 else round(px - 1.5 * atr, 2)
        tp = target_price if target_price and target_price > 0 else round(px + 2.5 * atr, 2)

        qty = shares if shares and shares > 0 else self.calculate_position_size(ticker, px, sl)
        margin_pct = 0.20 if str(order_type).upper() == "INTRADAY" else 1.0
        required_cash = round(px * qty * margin_pct, 2)

        if not self.can_open_position(required_cash):
            return {
                "status": "REJECTED",
                "reason": f"Insufficient available cash or max position limit reached. Need INR {required_cash:.2f}."
            }

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Disallow duplicate open position in same ticker
        cur.execute("SELECT COUNT(*) FROM portfolio_positions WHERE ticker = ? AND status = 'OPEN'", (ticker,))
        if cur.fetchone()[0] > 0:
            conn.close()
            return {"status": "REJECTED", "reason": f"Position already open in {ticker}"}

        now_str = timestamp if timestamp else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        exchange = "BSE" if "BO" in ticker.upper() or "5" in ticker else "NSE"

        # Calculate initial estimated charges
        est_charges = IndianBrokerFeeModel.calculate_charges(px, px, qty, order_type, exchange)

        cur.execute("""
        INSERT INTO portfolio_positions (
            ticker, exchange, order_type, side, entry_timestamp,
            entry_price, stop_loss, target_price, trailing_stop,
            quantity, allocated_cash, current_price,
            unrealized_gross_pnl, estimated_charges, unrealized_net_pnl, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, ?, ?, 'OPEN')
        """, (ticker, exchange, order_type, side, now_str, px, sl, tp, sl, qty, required_cash, px, est_charges["total_charges"], -est_charges["total_charges"]))

        # Deduct cash from portfolio state
        cur.execute("""
        UPDATE portfolio_state SET
            current_cash = current_cash - ?,
            allocated_margin = allocated_margin + ?,
            updated_at = ?
        WHERE id = 1
        """, (required_cash, required_cash, now_str))

        conn.commit()
        conn.close()

        return {
            "status": "EXECUTED",
            "ticker": ticker,
            "side": side,
            "order_type": order_type,
            "entry_price": px,
            "shares": qty,
            "stop_loss": sl,
            "target_price": tp,
            "allocated_cash": required_cash,
            "message": f"Bought {qty} shares of {ticker} at INR {px:.2f}. Total INR {required_cash:.2f} allocated."
        }

    def close_position(self, pos_id: int, exit_price: float, exit_reason: str = "MANUAL_EXIT", timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Closes an open position, computes statutory charges, updates cash, and logs to ledger.
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
        SELECT ticker, exchange, order_type, side, entry_timestamp, entry_price,
               stop_loss, target_price, quantity, allocated_cash
        FROM portfolio_positions WHERE id = ? AND status = 'OPEN'
        """, (pos_id,))
        row = cur.fetchone()

        if not row:
            conn.close()
            return {"status": "ERROR", "message": f"No open position with ID {pos_id}"}

        ticker, exchange, order_type, side, entry_time, entry_px, sl, tp, qty, allocated_cash = row
        now_str = timestamp if timestamp else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Compute exact statutory taxes and brokerage
        charges = IndianBrokerFeeModel.calculate_charges(
            entry_price=entry_px,
            exit_price=exit_price,
            quantity=qty,
            order_type=order_type,
            exchange=exchange
        )

        gross_pnl = charges["gross_pnl"]
        net_pnl = charges["net_pnl"]
        risk_per_sh = abs(entry_px - sl) if abs(entry_px - sl) > 0 else entry_px * 0.02
        r_mult = round(net_pnl / (risk_per_sh * qty), 2)

        # Mark position CLOSED
        cur.execute("UPDATE portfolio_positions SET status = 'CLOSED' WHERE id = ?", (pos_id,))

        # Update portfolio state: cash returned = allocated_cash + net_pnl
        cash_return = allocated_cash + net_pnl
        is_win = 1 if net_pnl > 0 else 0
        is_loss = 1 if net_pnl < 0 else 0

        cur.execute("""
        UPDATE portfolio_state SET
            current_cash = current_cash + ?,
            allocated_margin = allocated_margin - ?,
            realized_gross_pnl = realized_gross_pnl + ?,
            total_brokerage = total_brokerage + ?,
            total_stt = total_stt + ?,
            total_exchange_txn = total_exchange_txn + ?,
            total_sebi = total_sebi + ?,
            total_stamp_duty = total_stamp_duty + ?,
            total_gst = total_gst + ?,
            total_dp_charges = total_dp_charges + ?,
            total_slippage = total_slippage + ?,
            total_charges = total_charges + ?,
            realized_net_pnl = realized_net_pnl + ?,
            equity = equity + ?,
            total_trades = total_trades + 1,
            winning_trades = winning_trades + ?,
            losing_trades = losing_trades + ?,
            updated_at = ?
        WHERE id = 1
        """, (
            cash_return, allocated_cash, gross_pnl,
            charges["brokerage"], charges["stt"], charges["exchange_txn"],
            charges["sebi_charges"], charges["stamp_duty"], charges["gst"],
            charges["dp_charges"], charges["slippage"], charges["total_charges"],
            net_pnl, net_pnl, is_win, is_loss, now_str
        ))

        # Re-fetch new equity
        cur.execute("SELECT equity, winning_trades, total_trades FROM portfolio_state WHERE id = 1")
        st = cur.fetchone()
        new_equity = st[0]
        win_rate = (st[1] / st[2] * 100.0) if st[2] > 0 else 0.0

        cur.execute("UPDATE portfolio_state SET win_rate_pct = ? WHERE id = 1", (win_rate,))

        # Insert into Trade Ledger
        cur.execute("""
        INSERT INTO portfolio_trade_ledger (
            ticker, side, order_type, entry_time, exit_time,
            entry_px, exit_px, qty, gross_pnl, brokerage, stt,
            exchange_txn, sebi_charges, stamp_duty, gst, dp_charges,
            slippage, total_charges, net_pnl, r_multiple, exit_reason,
            portfolio_equity_after
        ) VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?
        )
        """, (
            ticker, side, order_type, entry_time, now_str,
            entry_px, exit_price, qty, gross_pnl,
            charges["brokerage"], charges["stt"], charges["exchange_txn"],
            charges["sebi_charges"], charges["stamp_duty"], charges["gst"],
            charges["dp_charges"], charges["slippage"], charges["total_charges"],
            net_pnl, r_mult, exit_reason, new_equity
        ))

        conn.commit()
        conn.close()

        return {
            "status": "CLOSED",
            "ticker": ticker,
            "entry_px": entry_px,
            "exit_px": exit_price,
            "quantity": qty,
            "gross_pnl": gross_pnl,
            "total_charges": charges["total_charges"],
            "net_pnl": net_pnl,
            "r_multiple": r_mult,
            "exit_reason": exit_reason,
            "new_portfolio_equity": new_equity
        }

    def update_market_prices(self):
        """Streams real-time ticks to revalue open positions and triggers auto-exits."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
        SELECT id, ticker, exchange, order_type, side, entry_price, stop_loss, target_price, trailing_stop, quantity
        FROM portfolio_positions WHERE status = 'OPEN'
        """)
        open_rows = cur.fetchall()
        conn.close()

        for r in open_rows:
            pos_id, sym, exch, otype, side, entry_px, sl, tp, trailing_sl, qty = r
            tick = fast_ticker.get_tick(sym)
            if not tick:
                continue

            curr_px = tick["last_price"]
            charges = IndianBrokerFeeModel.calculate_charges(entry_px, curr_px, qty, otype, exch)
            unreal_gross = charges["gross_pnl"]
            unreal_net = charges["net_pnl"]

            # Ratchet trailing stop if in profit
            new_trailing = trailing_sl
            atr = entry_px * 0.025
            if curr_px > entry_px + 1.2 * atr:
                new_trailing = max(trailing_sl, curr_px - 1.2 * atr)

            effective_stop = max(sl, new_trailing)

            # Check Exit Conditions
            if curr_px >= tp:
                self.close_position(pos_id, tp, exit_reason="TARGET_HIT (+2.5R)")
            elif curr_px <= effective_stop:
                reason = "TRAILING_STOP" if effective_stop > sl else "STOP_LOSS (-1R)"
                self.close_position(pos_id, effective_stop, exit_reason=reason)
            else:
                conn2 = sqlite3.connect(self.db_path)
                cur2 = conn2.cursor()
                cur2.execute("""
                UPDATE portfolio_positions SET
                    current_price = ?,
                    trailing_stop = ?,
                    unrealized_gross_pnl = ?,
                    estimated_charges = ?,
                    unrealized_net_pnl = ?
                WHERE id = ?
                """, (curr_px, new_trailing, unreal_gross, charges["total_charges"], unreal_net, pos_id))
                conn2.commit()
                conn2.close()

portfolio_5k = Portfolio5KManager()

if __name__ == "__main__":
    print("Testing 5K INR Portfolio Manager...")
    p = Portfolio5KManager()
    state = p.get_state()
    print("Initial Portfolio State:")
    print(f"  Cash: Rs {state['current_cash']}")
    print(f"  Equity: Rs {state['equity']}")
    print(f"  Trades: {state['total_trades']}")
