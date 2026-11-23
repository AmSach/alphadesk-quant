"""
AlphaDesk Quant - Indian Brokerage & Statutory Tax Engine
Models real-world transaction costs for Indian Equities across NSE and BSE.
Enforces statutory schedule: STT, Stamp Duty, Exchange Txn Fees, SEBI Turnover,
GST (18%), DP charges, and discount broker commission schedules.
"""

from typing import Dict, Any

class IndianBrokerFeeModel:
    @staticmethod
    def calculate_charges(
        entry_price: float,
        exit_price: float,
        quantity: int,
        order_type: str = "INTRADAY", # "INTRADAY" (MIS) or "DELIVERY" (CNC)
        exchange: str = "NSE",         # "NSE" or "BSE"
        broker: str = "ZERODHA"        # "ZERODHA", "GROWW", "ANGELONE"
    ) -> Dict[str, Any]:
        """
        Calculates exact statutory charges and net P&L for Indian Equity trades.
        Strictly adheres to SEBI, Indian Stamp Act, and Finance Act norms.
        """
        if quantity <= 0 or entry_price <= 0 or exit_price <= 0:
            return {
                "turnover_buy": 0.0,
                "turnover_sell": 0.0,
                "total_turnover": 0.0,
                "gross_pnl": 0.0,
                "brokerage": 0.0,
                "stt": 0.0,
                "exchange_txn": 0.0,
                "sebi_charges": 0.0,
                "stamp_duty": 0.0,
                "gst": 0.0,
                "dp_charges": 0.0,
                "slippage": 0.0,
                "total_charges": 0.0,
                "net_pnl": 0.0,
                "breakeven_points": 0.0,
                "cost_drag_pct": 0.0
            }

        turnover_buy = float(entry_price * quantity)
        turnover_sell = float(exit_price * quantity)
        total_turnover = turnover_buy + turnover_sell
        gross_pnl = turnover_sell - turnover_buy

        # 1. Brokerage Calculation
        if order_type.upper() == "DELIVERY":
            if broker.upper() == "GROWW":
                # Groww: 0.05% or Rs 20 (whichever lower) on buy and sell
                b_buy = min(20.0, turnover_buy * 0.0005)
                b_sell = min(20.0, turnover_sell * 0.0005)
                brokerage = round(b_buy + b_sell, 2)
            else:
                # Zerodha / AngelOne Delivery: Rs 0
                brokerage = 0.0
        else: # INTRADAY
            # 0.03% or Rs 20 per executed order (whichever lower) on buy and sell
            b_buy = min(20.0, turnover_buy * 0.0003)
            b_sell = min(20.0, turnover_sell * 0.0003)
            brokerage = round(b_buy + b_sell, 2)

        # 2. STT / CTT (Securities Transaction Tax)
        if order_type.upper() == "DELIVERY":
            # 0.1% on buy and 0.1% on sell turnover
            stt = round(total_turnover * 0.001, 2)
        else: # INTRADAY
            # 0.025% on sell turnover only
            stt = round(turnover_sell * 0.00025, 2)

        # 3. Exchange Transaction Charges
        if exchange.upper() == "BSE":
            # BSE: 0.00375% of total turnover
            exchange_txn = round(total_turnover * 0.0000375, 2)
        else:
            # NSE: 0.00297% of total turnover
            exchange_txn = round(total_turnover * 0.0000297, 2)

        # 4. SEBI Turnover Charges (Rs 10 per crore = 0.0001% of total turnover)
        sebi_charges = round(total_turnover * 0.000001, 2)

        # 5. Stamp Duty (State Stamp Act)
        if order_type.upper() == "DELIVERY":
            # 0.015% or Rs 1500 per crore on buy side
            stamp_duty = round(turnover_buy * 0.00015, 2)
        else: # INTRADAY
            # 0.003% or Rs 300 per crore on buy side
            stamp_duty = round(turnover_buy * 0.00003, 2)

        # 6. GST (18% on Brokerage + Exchange Txn + SEBI)
        taxable_services = brokerage + exchange_txn + sebi_charges
        gst = round(taxable_services * 0.18, 2)

        # 7. DP (Depository Participant) Charges (Delivery sell only)
        # Zerodha/Groww: Rs 13.50 + 18% GST = Rs 15.93 per scrip debit
        dp_charges = 15.93 if order_type.upper() == "DELIVERY" else 0.0

        # 8. Execution Slippage Buffer (0.04% of total turnover)
        slippage = round(total_turnover * 0.0004, 2)

        total_charges = round(
            brokerage + stt + exchange_txn + sebi_charges + stamp_duty + gst + dp_charges + slippage,
            2
        )

        net_pnl = round(gross_pnl - total_charges, 2)
        breakeven_points = round(total_charges / quantity, 2) if quantity > 0 else 0.0
        cost_drag_pct = round((total_charges / turnover_buy) * 100.0, 3) if turnover_buy > 0 else 0.0

        return {
            "turnover_buy": round(turnover_buy, 2),
            "turnover_sell": round(turnover_sell, 2),
            "total_turnover": round(total_turnover, 2),
            "gross_pnl": round(gross_pnl, 2),
            "brokerage": brokerage,
            "stt": stt,
            "exchange_txn": exchange_txn,
            "sebi_charges": sebi_charges,
            "stamp_duty": stamp_duty,
            "gst": gst,
            "dp_charges": dp_charges,
            "slippage": slippage,
            "total_charges": total_charges,
            "net_pnl": net_pnl,
            "breakeven_points": breakeven_points,
            "cost_drag_pct": cost_drag_pct
        }

if __name__ == "__main__":
    test_trade = IndianBrokerFeeModel.calculate_charges(
        entry_price=64.20,
        exit_price=70.62,
        quantity=70,
        order_type="INTRADAY"
    )
    print("Test Indian Broker Charges calculation:")
    for k, v in test_trade.items():
        print(f"  {k}: {v}")
