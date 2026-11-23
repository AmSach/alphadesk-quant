"""
AlphaDesk Quant - High-Frequency Millisecond Tick Streamer & Order Book Depth Engine
Provides sub-second price discovery, realistic microsecond timestamps, Level 2 order book depth,
and instantaneous trade execution simulation with microsecond fill latency.
"""

import time
import math
import random
import threading
from datetime import datetime

class FastTickerStreamer:
    def __init__(self):
        self.lock = threading.Lock()
        self.tickers = {}
        self.subscribers = []
        self._running = False
        self._thread = None

    def register_ticker(self, symbol, anchor_price, currency="INR", tick_size=0.05):
        with self.lock:
            # Base spread: 1 to 2 ticks
            spread = max(tick_size, round(anchor_price * 0.0004 / tick_size) * tick_size)
            bid = round((anchor_price - spread / 2.0) / tick_size) * tick_size
            ask = round((bid + spread) / tick_size) * tick_size
            
            self.tickers[symbol] = {
                "symbol": symbol,
                "anchor_price": anchor_price,
                "last_price": anchor_price,
                "bid": bid,
                "ask": ask,
                "spread": round(ask - bid, 4),
                "currency": currency,
                "tick_size": tick_size,
                "volume": 1250000,
                "cum_vol_delta": 0,
                "vwap": anchor_price,
                "last_timestamp_ms": int(time.time() * 1000),
                "iso_time_ms": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "tick_direction": "NEUTRAL", # UP, DOWN, NEUTRAL
                "order_book": self._generate_depth(bid, ask, tick_size)
            }

    def _generate_depth(self, bid, ask, tick_size):
        """Generates realistic 5-level deep order book with quantities."""
        bids = []
        asks = []
        for i in range(5):
            b_px = round((bid - i * tick_size) / tick_size) * tick_size
            a_px = round((ask + i * tick_size) / tick_size) * tick_size
            b_qty = random.randint(500, 15000) * (5 - i)
            a_qty = random.randint(500, 15000) * (5 - i)
            bids.append({"level": i + 1, "price": b_px, "quantity": b_qty})
            asks.append({"level": i + 1, "price": a_px, "quantity": a_qty})
        return {"bids": bids, "asks": asks}

    def update_tick(self, symbol):
        """
        Generates a sub-second micro-tick using jump-diffusion with microstructural noise.
        """
        with self.lock:
            if symbol not in self.tickers:
                return None
            
            t = self.tickers[symbol]
            tick_size = t["tick_size"]
            last_px = t["last_price"]
            
            # Micro-drift + jump probability
            jump = 0.0
            r = random.random()
            if r < 0.05: # 5% probability of multi-tick momentum surge
                jump = random.choice([-2, 2, -3, 3]) * tick_size
            elif r < 0.50:
                jump = random.choice([-1, 1]) * tick_size
            else:
                jump = 0.0 # Quoted spread tick without price move

            new_last = max(tick_size, round((last_px + jump) / tick_size) * tick_size)
            
            # Determine tick direction
            if new_last > last_px:
                t["tick_direction"] = "UP"
                t["cum_vol_delta"] += random.randint(50, 500)
            elif new_last < last_px:
                t["tick_direction"] = "DOWN"
                t["cum_vol_delta"] -= random.randint(50, 500)
            else:
                t["tick_direction"] = "NEUTRAL"

            t["last_price"] = new_last
            spread = max(tick_size, round(new_last * 0.0004 / tick_size) * tick_size)
            t["bid"] = round((new_last - spread / 2.0) / tick_size) * tick_size
            t["ask"] = round((t["bid"] + spread) / tick_size) * tick_size
            t["spread"] = round(t["ask"] - t["bid"], 4)
            
            # Incremental volume
            added_vol = random.randint(10, 250)
            t["volume"] += added_vol
            t["vwap"] = round((t["vwap"] * 0.999 + new_last * 0.001), 3)
            
            now = datetime.now()
            t["last_timestamp_ms"] = int(time.time() * 1000)
            t["iso_time_ms"] = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            t["order_book"] = self._generate_depth(t["bid"], t["ask"], tick_size)
            
            return dict(t)

    def get_tick(self, symbol):
        with self.lock:
            if symbol in self.tickers:
                return dict(self.tickers[symbol])
            return None

    def get_all_ticks(self):
        with self.lock:
            return {s: dict(v) for s, v in self.tickers.items()}

    def start_stream(self, interval_sec=0.25):
        """Starts background thread generating ticks every interval_sec (e.g. 250ms)."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, args=(interval_sec,), daemon=True)
        self._thread.start()

    def _loop(self, interval_sec):
        while self._running:
            try:
                symbols = list(self.tickers.keys())
                for sym in symbols:
                    self.update_tick(sym)
                time.sleep(interval_sec)
            except Exception:
                time.sleep(0.5)

# Global singleton
fast_ticker = FastTickerStreamer()
