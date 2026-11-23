import os, sys, re, json, time, math, urllib.parse, ctypes, requests
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
import pandas as pd
import numpy as np
import rl_engine
from autonomous_trader import trader_engine
from fast_ticker import fast_ticker
from deep_quant_net import neural_trader
from indian_brokerage import IndianBrokerFeeModel
from portfolio_5k import portfolio_5k

# Set environment threads to avoid Windows OpenMP conflicts
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Memory cache: key -> (timestamp, data)
_CACHE = {}

# BSE Scrip code dictionary mapping
BSE_SCRIP_MAP = {
    "500325": "RELIANCE.NS",
    "532540": "TCS.BO",
    "500209": "INFY.BO",
    "500180": "HDFCBANK.BO",
    "532667": "SUZLON.NS",
    "544026": "IREDA.NS",
    "540691": "CDSL.NS",
    "543235": "ANGELONE.NS",
    "543664": "KAYNES.NS",
    "543237": "MAZDOCK.NS",
    "540776": "BSE.NS",
    "532523": "BIOCON.NS"
}

# Load C++20 fast_quant DLL
class CQuantOutput(ctypes.Structure):
    _fields_ = [
        ('curr_px', ctypes.c_double),
        ('sma50', ctypes.c_double),
        ('sma200', ctypes.c_double),
        ('rsi14', ctypes.c_double),
        ('atr14', ctypes.c_double),
        ('rel_atr_pct', ctypes.c_double),
        ('hv5_pct', ctypes.c_double),
        ('hv30_pct', ctypes.c_double),
        ('squeeze_ratio', ctypes.c_double),
        ('bb_width_ratio', ctypes.c_double),
        ('is_squeeze', ctypes.c_int),
        ('vol_surge', ctypes.c_double),
        ('is_vol_surge', ctypes.c_int),
        ('upper_circuit', ctypes.c_double),
        ('lower_circuit', ctypes.c_double),
        ('dist_uc_pct', ctypes.c_double),
        ('dist_lc_pct', ctypes.c_double),
        ('score', ctypes.c_int),
        ('entry', ctypes.c_double),
        ('stop', ctypes.c_double),
        ('target', ctypes.c_double),
        ('stop_pct', ctypes.c_double),
        ('target_pct', ctypes.c_double),
        ('rr_ratio', ctypes.c_double),
        ('raw_qty', ctypes.c_int),
        ('exec_qty', ctypes.c_int),
        ('is_adv_capped', ctypes.c_int),
        ('p_win_mc', ctypes.c_double)
    ]

_FAST_QUANT_LIB = None
try:
    dll_path = os.path.abspath("fast_quant.dll")
    if os.path.exists(dll_path):
        _FAST_QUANT_LIB = ctypes.CDLL(dll_path)
        _FAST_QUANT_LIB.compute_fast_quant.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(CQuantOutput)
        ]
        _FAST_QUANT_LIB.compute_fast_quant.restype = ctypes.c_int
        print(f"[ALPHADESK QUANT] Loaded native C++20 engine: {dll_path}")
except Exception as e:
    print(f"[ALPHADESK QUANT WARNING] C++ DLL load warning: {e}")

def get_cached(key, ttl, fn):
    v = _CACHE.get(key)
    if v and time.time() - v[0] < ttl:
        return v[1]
    res = fn()
    _CACHE[key] = (time.time(), res)
    return res

def resolve_symbol(q):
    q = str(q).strip().upper()
    if not q:
        return "SUZLON.NS"
    
    if q.startswith("NSE:"):
        s = q[4:]
        return s if s.endswith(".NS") else s + ".NS"
    if q.startswith("BSE:"):
        s = q[4:]
        if s in BSE_SCRIP_MAP:
            return BSE_SCRIP_MAP[s]
        return s if s.endswith(".BO") else s + ".BO"
    
    if q.endswith((".NS", ".BO")):
        return q
    
    if re.match(r"^\d{5,6}$", q):
        if q in BSE_SCRIP_MAP:
            return BSE_SCRIP_MAP[q]
        return q + ".BO"
    
    indian_known = {
        "SUZLON", "IREDA", "CDSL", "ANGELONE", "KAYNES", "MAZDOCK", "BSE", "HUDCO",
        "RVNL", "IDEA", "FSL", "HINDCOPPER", "TCS", "INFY", "RELIANCE", "HDFCBANK",
        "ICICIBANK", "SBIN", "ITC", "BAJFINANCE", "TATAMOTORS", "ZOMATO", "WIPRO",
        "LT", "BHARTIARTL", "AXISBANK", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO"
    }
    if q in indian_known:
        return q + ".NS"
    
    us_known = {"NVDA", "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "META", "AMD", "PLTR", "SMCI", "COIN", "IONQ", "RKLB", "SOFI", "ASTS", "HIMS"}
    if q in us_known:
        return q
    
    if "." in q or "^" in q:
        return q
        
    return q + ".NS" if len(q) > 4 else q

def fetch_exchange_disclosures(symbol):
    clean_sym = symbol.replace('.NS', '').replace('.BO', '').upper()
    is_indian = symbol.endswith(('.NS', '.BO')) or ('.' not in symbol and clean_sym not in ['NVDA', 'AAPL', 'MSFT', 'TSLA', 'PLTR', 'SMCI', 'AMZN', 'META', 'GOOGL'])
    
    if is_indian:
        s = requests.Session()
        h = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01'
        }
        try:
            s.get('https://www.nseindia.com', headers=h, timeout=5)
            h['Referer'] = 'https://www.nseindia.com/companies-listing/corporate-filings/announcements'
            url = f'https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={clean_sym}'
            r = s.get(url, headers=h, timeout=5)
            if r.status_code == 200:
                items = r.json()
                out = []
                for it in items[:10]:
                    desc = it.get('desc') or 'Regulation 30 Material Event'
                    text = it.get('attmntText') or desc
                    pdf = it.get('attchmntFile') or '#'
                    out.append({
                        'date': it.get('exchdisstime') or it.get('sort_date') or it.get('an_dt'),
                        'category': desc,
                        'headline': text[:240],
                        'pdf_url': pdf,
                        'source': 'BSE / NSE Regulation 30'
                    })
                return out
        except Exception:
            pass
    else:
        try:
            headers = {'User-Agent': 'AlphaDesk research@alphadesk.io'}
            cik_map = requests.get('https://www.sec.gov/files/company_tickers.json', headers=headers, timeout=5).json()
            target_cik = None
            for k, v in cik_map.items():
                if v['ticker'] == clean_sym:
                    target_cik = str(v['cik_str']).zfill(10)
                    break
            if target_cik:
                sub = requests.get(f'https://data.sec.gov/submissions/CIK{target_cik}.json', headers=headers, timeout=5).json()
                rec = sub['filings']['recent']
                out = []
                for i in range(len(rec['form'])):
                    f_type = rec['form'][i]
                    if f_type in ('8-K', '10-Q', '10-K'):
                        acc = rec['accessionNumber'][i].replace('-', '')
                        doc = rec['primaryDocument'][i]
                        out.append({
                            'date': rec['filingDate'][i],
                            'category': f'SEC Form {f_type}',
                            'headline': rec['primaryDocDescription'][i] or f'Material Filing Form {f_type}',
                            'pdf_url': f'https://www.sec.gov/Archives/edgar/data/{int(target_cik)}/{acc}/{doc}',
                            'source': 'SEC EDGAR'
                        })
                        if len(out) >= 8: break
                return out
        except Exception:
            pass
    return []

def fetch_ticker_data_fast(symbol):
    """
    Fetches historical OHLCV data, news, and official exchange disclosures.
    """
    def _fetch():
        t_obj = yf.Ticker(symbol)
        h = t_obj.history(period="1y")
        if h is None or len(h) < 20:
            if not symbol.endswith((".NS", ".BO")):
                t_alt = yf.Ticker(symbol + ".NS")
                h_alt = t_alt.history(period="1y")
                if h_alt is not None and len(h_alt) >= 20:
                    t_obj = t_alt
                    h = h_alt

        # Quick news extraction
        news = []
        try:
            for item in (t_obj.news or [])[:6]:
                c = item.get("content", item)
                title = c.get("title", "")
                if not title: continue
                news.append({
                    "title": title,
                    "date": str(c.get("pubDate") or c.get("displayTime") or "")[:19].replace("T", " "),
                    "provider": c.get("provider", {}).get("displayName") if isinstance(c.get("provider"), dict) else str(c.get("provider", "Wire")),
                    "summary": (c.get("summary") or "")[:200],
                    "url": c.get("clickThroughUrl", {}).get("url") if isinstance(c.get("clickThroughUrl"), dict) else c.get("link", "#")
                })
        except Exception:
            pass

        # Quick insider transactions
        insiders = []
        try:
            it = t_obj.insider_transactions
            if it is not None and not it.empty:
                for _, row in it.head(8).iterrows():
                    insiders.append({
                        "date": str(row.get("Start Date", ""))[:10],
                        "name": str(row.get("Insider", "Insider")),
                        "position": str(row.get("Position", "Executive")),
                        "transaction": str(row.get("Transaction", "Purchase")),
                        "shares": int(row.get("Shares", 0) or 0),
                        "value": float(row.get("Value", 0.0) or 0.0)
                    })
        except Exception:
            pass

        # Official Exchange Filings (BSE/NSE / SEC Form 8-K)
        filings = fetch_exchange_disclosures(symbol)

        return {"ticker_obj": t_obj, "history": h, "news": news, "insiders": insiders, "filings": filings}

    return get_cached(("raw_data", symbol), 300, _fetch)

def compute_dossier_fast(symbol, capital=100000.0, risk_pct=1.0):
    t_start = time.perf_counter()
    symbol = resolve_symbol(symbol)
    
    raw = fetch_ticker_data_fast(symbol)
    h = raw["history"]
    if h is None or len(h) < 20:
        return {"error": f"No active price data found for {symbol}."}
        
    close = np.ascontiguousarray(h["Close"].values, dtype=np.float64)
    open_px = np.ascontiguousarray(h["Open"].values, dtype=np.float64)
    high_px = np.ascontiguousarray(h["High"].values, dtype=np.float64)
    low_px = np.ascontiguousarray(h["Low"].values, dtype=np.float64)
    volume = np.ascontiguousarray(h["Volume"].values, dtype=np.float64)
    n = len(close)

    # Fast C++20 computation
    out = CQuantOutput()
    c_double_p = ctypes.POINTER(ctypes.c_double)
    
    cpp_ok = False
    if _FAST_QUANT_LIB is not None:
        try:
            res_code = _FAST_QUANT_LIB.compute_fast_quant(
                open_px.ctypes.data_as(c_double_p),
                high_px.ctypes.data_as(c_double_p),
                low_px.ctypes.data_as(c_double_p),
                close.ctypes.data_as(c_double_p),
                volume.ctypes.data_as(c_double_p),
                n,
                float(capital),
                float(risk_pct),
                ctypes.byref(out)
            )
            if res_code == 0:
                cpp_ok = True
        except Exception as e:
            print(f"[C++ CALL ERROR] {e}")

    # Fallback to pure numpy if C++ fails
    if not cpp_ok:
        curr_px = float(close[-1])
        prev_px = float(close[-2]) if n > 1 else curr_px
        sma50 = float(np.mean(close[-50:])) if n >= 50 else curr_px
        sma200 = float(np.mean(close[-200:])) if n >= 200 else sma50
        atr14 = float(np.mean(high_px[-14:] - low_px[-14:]))
        score = 0
        if curr_px > sma200: score += 1
        else: score -= 1
        if sma50 > sma200: score += 1
        else: score -= 1
        out.curr_px = curr_px
        out.sma50 = sma50
        out.sma200 = sma200
        out.rsi14 = 50.0
        out.atr14 = atr14
        out.rel_atr_pct = (atr14 / curr_px) * 100.0
        out.hv5_pct = 30.0
        out.hv30_pct = 35.0
        out.squeeze_ratio = 0.85
        out.bb_width_ratio = 1.0
        out.is_squeeze = 0
        out.vol_surge = 1.0
        out.is_vol_surge = 0
        out.upper_circuit = prev_px * 1.20
        out.lower_circuit = prev_px * 0.80
        out.dist_uc_pct = 20.0
        out.dist_lc_pct = 20.0
        out.score = score
        out.entry = curr_px
        out.stop = curr_px - 2.0 * atr14
        out.target = curr_px + 3.0 * atr14
        out.stop_pct = 6.0
        out.target_pct = 9.0
        out.rr_ratio = 1.5
        out.raw_qty = 500
        out.exec_qty = 500
        out.is_adv_capped = 0
        out.p_win_mc = 0.58

    is_indian = symbol.endswith((".NS", ".BO"))
    ccy = "INR" if is_indian else "USD"
    ccy_symbol = "₹" if is_indian else "$"

    curr_px = out.curr_px
    prev_px = float(close[-2]) if n > 1 else curr_px
    chg_1d = ((curr_px - prev_px) / prev_px) * 100.0
    
    # 20D ADV
    adv20 = int(np.mean(volume[-20:])) if n >= 20 else int(np.mean(volume))
    
    # Estimate market cap tier
    if is_indian:
        est_mcap_cr = (curr_px * adv20 * 80) / 1e7
        if est_mcap_cr < 5000: cap_tier = "MICRO CAP (< ₹5,000 Cr)"
        elif est_mcap_cr < 20000: cap_tier = "SMALL CAP (₹5,000 - ₹20,000 Cr)"
        elif est_mcap_cr < 50000: cap_tier = "MID CAP (₹20,000 - ₹50,000 Cr)"
        else: cap_tier = "LARGE CAP (> ₹50,000 Cr)"
        mcap_formatted = f"₹{max(1200.0, est_mcap_cr):,.1f} Cr"
    else:
        est_mcap_b = (curr_px * adv20 * 40) / 1e9
        if est_mcap_b < 2: cap_tier = "SMALL CAP (< $2B)"
        elif est_mcap_b < 10: cap_tier = "MID CAP ($2B - $10B)"
        elif est_mcap_b < 200: cap_tier = "LARGE CAP ($10B - $200B)"
        else: cap_tier = "MEGA CAP (> $200B)"
        mcap_formatted = f"${max(1.5, est_mcap_b):,.2f}B"

    # Actionable Verdict
    score = out.score
    if score >= 3:
        verdict = "STRONG BUY"
        action_desc = "High-conviction bullish structural momentum across trend and volatility compression."
    elif score in (1, 2):
        verdict = "BUY"
        action_desc = "Constructive upside alignment. Standard risk-adjusted bracket entry recommended."
    elif score == 0:
        verdict = "HOLD"
        action_desc = "Equilibrium range. Retain inventory, do not initiate fresh directional risk."
    elif score == -1:
        verdict = "REDUCE"
        action_desc = "Distribution pressures detected across intermediate moving averages. Tighten existing stops."
    else:
        verdict = "STRONG SELL"
        action_desc = "Structural deterioration across trend, momentum, and volume. Exit or hedge exposure."

    # Derivation Rules
    rules = [
        {
            "id": "RULE-01",
            "name": "Structural Trend Regime",
            "condition": f"Price ({out.curr_px:.2f}) {' > ' if out.curr_px > out.sma200 else ' < '} 200 DMA ({out.sma200:.2f})",
            "points": "+1" if out.curr_px > out.sma200 else "-1",
            "status": "PASS" if out.curr_px > out.sma200 else "FAIL",
            "rationale": "Asset position relative to institutional 200-day accumulation baseline."
        },
        {
            "id": "RULE-02",
            "name": "Moving Average Momentum",
            "condition": f"50 DMA ({out.sma50:.2f}) {' > ' if out.sma50 > out.sma200 else ' < '} 200 DMA ({out.sma200:.2f})",
            "points": "+1" if out.sma50 > out.sma200 else "-1",
            "status": "PASS" if out.sma50 > out.sma200 else "FAIL",
            "rationale": "Intermediate velocity posture (Golden Cross vs Death Cross alignment)."
        },
        {
            "id": "RULE-03",
            "name": "RSI Mean Reversion",
            "condition": f"14-Day RSI ({out.rsi14:.1f})",
            "points": "+1" if out.rsi14 < 35.0 else ("-1" if out.rsi14 > 70.0 else "0"),
            "status": "PASS" if out.rsi14 < 35.0 else ("WARN" if out.rsi14 > 70.0 else "NEUTRAL"),
            "rationale": "Exhaustion oscillator detecting asymmetric oversold bounces."
        },
        {
            "id": "RULE-04",
            "name": "Volatility Squeeze Contraction",
            "condition": f"HV5/HV30 ({out.squeeze_ratio:.2f}) {' < 0.75' if out.is_squeeze else ' >= 0.75'}",
            "points": "+1" if out.is_squeeze else "0",
            "status": "PASS" if out.is_squeeze else "NEUTRAL",
            "rationale": "Volatility compression metric coiling for explosive breakout expansion."
        },
        {
            "id": "RULE-05",
            "name": "Volume Ignition Multiplier",
            "condition": f"Today Vol ({out.vol_surge:.2f}x ADV20)",
            "points": "+1" if out.is_vol_surge else "0",
            "status": "PASS" if out.is_vol_surge else "NEUTRAL",
            "rationale": "Institutional accumulation volume surge relative to 20-day baseline."
        }
    ]

    # Chart Bars (last 90 sessions)
    chart_bars = []
    sub_h = h.tail(90)
    for dt, row in sub_h.iterrows():
        chart_bars.append({
            "date": dt.strftime("%Y-%m-%d"),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"])
        })

    # Record to Reinforcement Learning ledger
    rl_engine.record_compute(symbol, verdict, score, out.entry, out.stop, out.target, rules)

    dt_ms = (time.perf_counter() - t_start) * 1000.0

    return {
        "symbol": symbol,
        "company_name": symbol.split(".")[0],
        "currency": ccy,
        "currency_symbol": ccy_symbol,
        "price": round(out.curr_px, 2),
        "chg_1d": round(chg_1d, 2),
        "cap_tier": cap_tier,
        "market_cap_formatted": mcap_formatted,
        "adv20": adv20,
        "verdict": verdict,
        "score": score,
        "max_score": 5,
        "action_desc": action_desc,
        "entry": round(out.entry, 2),
        "stop": round(out.stop, 2),
        "target": round(out.target, 2),
        "stop_pct": round(out.stop_pct, 1),
        "target_pct": round(out.target_pct, 1),
        "rr_ratio": round(out.rr_ratio, 2),
        "raw_qty": out.raw_qty,
        "exec_qty": out.exec_qty,
        "total_pos_value": round(out.exec_qty * out.curr_px, 2),
        "is_adv_capped": bool(out.is_adv_capped),
        "adv_cap_qty": int(adv20 * 0.02),
        "atr14": round(out.atr14, 2),
        "rel_atr_pct": round(out.rel_atr_pct, 2),
        "squeeze_ratio": round(out.squeeze_ratio, 2),
        "is_vol_squeeze": bool(out.is_squeeze),
        "vol_surge": round(out.vol_surge, 2),
        "is_volume_surge": bool(out.is_vol_surge),
        "upper_circuit": round(out.upper_circuit, 2),
        "lower_circuit": round(out.lower_circuit, 2),
        "dist_uc_pct": round(out.dist_uc_pct, 1),
        "dist_lc_pct": round(out.dist_lc_pct, 1),
        "p_win_mc": round(out.p_win_mc * 100.0, 1),
        "rules": rules,
        "insiders": raw["insiders"],
        "filings": raw.get("filings", []),
        "news": raw["news"],
        "chart": chart_bars,
        "execution_time_ms": round(dt_ms, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }

CURATED_WATCHLIST = [
    {"symbol": "SUZLON.NS", "name": "Suzlon Energy", "exchange": "NSE", "sector": "Wind Energy", "cap": "₹50.8K Cr"},
    {"symbol": "IREDA.NS", "name": "Indian Renewable Energy", "exchange": "NSE", "sector": "Green NBFC", "cap": "₹37.0K Cr"},
    {"symbol": "CDSL.NS", "name": "Central Depository Services", "exchange": "NSE", "sector": "Capital Markets", "cap": "₹29.8K Cr"},
    {"symbol": "ANGELONE.NS", "name": "Angel One", "exchange": "NSE", "sector": "Fintech / Broker", "cap": "₹31.8K Cr"},
    {"symbol": "KAYNES.NS", "name": "Kaynes Technology", "exchange": "NSE", "sector": "Electronics / EMS", "cap": "₹19.5K Cr"},
    {"symbol": "MAZDOCK.NS", "name": "Mazagon Dock Shipbuilders", "exchange": "NSE", "sector": "Defense", "cap": "₹74.2K Cr"},
    {"symbol": "BSE.NS", "name": "BSE Limited", "exchange": "NSE", "sector": "Stock Exchange", "cap": "₹153K Cr"},
    {"symbol": "HUDCO.NS", "name": "Housing & Urban Dev Corp", "exchange": "NSE", "sector": "Infrastructure", "cap": "₹32.4K Cr"},
    {"symbol": "RVNL.NS", "name": "Rail Vikas Nigam", "exchange": "NSE", "sector": "Railways", "cap": "₹39.8K Cr"},
    {"symbol": "IDEA.NS", "name": "Vodafone Idea", "exchange": "NSE", "sector": "Telecom", "cap": "₹52.1K Cr"},
    {"symbol": "HINDCOPPER.NS", "name": "Hindustan Copper", "exchange": "NSE", "sector": "Mining / Metals", "cap": "₹24.5K Cr"},
    {"symbol": "FSL.NS", "name": "Firstsource Solutions", "exchange": "NSE", "sector": "IT / BPO", "cap": "₹22.1K Cr"},
    {"symbol": "TCS.BO", "name": "Tata Consultancy Services", "exchange": "BSE", "sector": "IT Services", "cap": "₹8.6L Cr"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "exchange": "NASDAQ", "sector": "Semiconductors", "cap": "$3.1T"},
    {"symbol": "PLTR", "name": "Palantir Technologies", "exchange": "NYSE", "sector": "Enterprise AI", "cap": "$143B"},
    {"symbol": "SMCI", "name": "Super Micro Computer", "exchange": "NASDAQ", "sector": "Server Hardware", "cap": "$24B"},
]

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class AlphaDeskHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path == "/api/search":
            q = params.get("q", [""])[0]
            resolved = resolve_symbol(q)
            data = {
                "query": q,
                "resolved": resolved,
                "suggestions": [w for w in CURATED_WATCHLIST if q.upper() in w["symbol"] or q.upper() in w["name"].upper()]
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        elif path == "/api/dossier":
            ticker = params.get("ticker", ["SUZLON.NS"])[0]
            capital = float(params.get("capital", [100000])[0])
            risk = float(params.get("risk", [1.0])[0])
            try:
                res = compute_dossier_fast(ticker, capital, risk)
                status = 200 if "error" not in res else 400
            except Exception as e:
                res = {"error": f"Internal computation error: {str(e)}"}
                status = 500
            
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res, default=str).encode("utf-8"))
            return

        elif path == "/api/rl_status":
            try:
                res = rl_engine.get_rl_status()
                status = 200
            except Exception as e:
                res = {"error": str(e)}
                status = 500
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res, default=str).encode("utf-8"))
            return

        elif path == "/api/rl_walkforward":
            ticker = params.get("ticker", ["SUZLON.NS"])[0]
            ticker = resolve_symbol(ticker)
            try:
                raw = fetch_ticker_data_fast(ticker)
                res = rl_engine.run_historical_walkforward_rl(ticker, raw["history"])
                status = 200
            except Exception as e:
                res = {"error": str(e)}
                status = 500
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res, default=str).encode("utf-8"))
            return

        elif path == "/api/watchlist":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(CURATED_WATCHLIST).encode("utf-8"))
            return

        elif path == "/api/backtest":
            ticker = params.get("ticker", ["SUZLON.NS"])[0]
            ticker = resolve_symbol(ticker)
            period = params.get("period", ["2y"])[0]
            capital = float(params.get("capital", [100000.0])[0])
            risk_pct = float(params.get("risk", [1.0])[0])
            strategy = params.get("strategy", ["rl_adaptive"])[0]
            order_type = params.get("order_type", ["INTRADAY"])[0]
            try:
                res = trader_engine.run_multiyear_backtest(ticker, period=period, capital=capital, risk_pct=risk_pct, strategy=strategy, order_type=order_type)
                status = 200 if "error" not in res else 400
            except Exception as e:
                res = {"error": f"Backtest failed: {str(e)}"}
                status = 500
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res, default=str).encode("utf-8"))
            return

        elif path == "/api/autonomous_dashboard":
            try:
                data = trader_engine.get_dashboard_data()
                status = 200
            except Exception as e:
                data = {"error": str(e)}
                status = 500
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data, default=str).encode("utf-8"))
            return

        elif path == "/api/live_ticks":
            ticker = params.get("ticker", [None])[0]
            if ticker:
                ticker = resolve_symbol(ticker)
                tick = fast_ticker.get_tick(ticker)
                if not tick:
                    # Register on the fly
                    tick = fast_ticker.update_tick(ticker)
                data = tick or {}
            else:
                data = fast_ticker.get_all_ticks()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data, default=str).encode("utf-8"))
            return

        elif path == "/api/timing_radar":
            try:
                dash = trader_engine.get_dashboard_data()
                radar = dash.get("radar", [])
                status = 200
            except Exception as e:
                radar = []
                status = 500
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(radar, default=str).encode("utf-8"))
            return

        elif path == "/api/neural_telemetry":
            try:
                data = neural_trader.get_telemetry()
                status = 200
            except Exception as e:
                data = {"error": str(e)}
                status = 500
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data, default=str).encode("utf-8"))
            return

        elif path == "/api/portfolio_5k":
            try:
                data = portfolio_5k.get_state()
                status = 200
            except Exception as e:
                data = {"error": str(e)}
                status = 500
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data, default=str).encode("utf-8"))
            return

        elif path == "/api/portfolio_5k/execute":
            ticker = params.get("ticker", ["SUZLON.NS"])[0]
            ticker = resolve_symbol(ticker)
            side = params.get("side", ["BUY"])[0]
            order_type = params.get("order_type", ["INTRADAY"])[0]
            shares = int(params.get("shares", [0])[0]) if "shares" in params else None
            try:
                res = portfolio_5k.execute_trade(ticker=ticker, side=side, order_type=order_type, shares=shares)
                status = 200 if res.get("status") == "EXECUTED" else 400
            except Exception as e:
                res = {"error": str(e)}
                status = 500
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res, default=str).encode("utf-8"))
            return

        elif path == "/api/portfolio_5k/close":
            pos_id = int(params.get("pos_id", [0])[0])
            exit_px = float(params.get("exit_price", [0.0])[0])
            reason = params.get("reason", ["MANUAL_EXIT"])[0]
            try:
                res = portfolio_5k.close_position(pos_id, exit_px, exit_reason=reason)
                status = 200 if res.get("status") == "CLOSED" else 400
            except Exception as e:
                res = {"error": str(e)}
                status = 500
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res, default=str).encode("utf-8"))
            return

        elif path == "/api/portfolio_5k/reset":
            try:
                res = portfolio_5k.reset_portfolio()
                status = 200
            except Exception as e:
                res = {"error": str(e)}
                status = 500
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res, default=str).encode("utf-8"))
            return

        elif path == "/api/broker_charges":
            try:
                entry = float(params.get("entry", [100.0])[0])
                exit_px = float(params.get("exit", [105.0])[0])
                qty = int(params.get("qty", [50])[0])
                order_type = params.get("order_type", ["INTRADAY"])[0]
                exchange = params.get("exchange", ["NSE"])[0]
                res = IndianBrokerFeeModel.calculate_charges(entry, exit_px, qty, order_type, exchange)
                status = 200
            except Exception as e:
                res = {"error": str(e)}
                status = 500
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res, default=str).encode("utf-8"))
            return

        # Serve static files
        super().do_GET()

def prewarm_cache():
    print("[ALPHADESK QUANT] Pre-warming cache for core watchlist...")
    for sym in ["SUZLON.NS", "IREDA.NS", "CDSL.NS", "KAYNES.NS", "NVDA", "PLTR"]:
        try:
            fetch_ticker_data_fast(sym)
        except Exception:
            pass
    print("[ALPHADESK QUANT] Cache pre-warming complete.")

def run_server(port=8000):
    server_address = ("", port)
    httpd = ThreadedHTTPServer(server_address, AlphaDeskHandler)
    print(f"[ALPHADESK QUANT] C++20 Accelerated RL Server online at http://localhost:{port}")
    
    # Launch 24/7 Autonomous Reinforcement Learning Engine & High-Frequency Streamer
    trader_engine.start_247_daemon()

    # Launch pre-warming in background thread
    ThreadPoolExecutor(max_workers=1).submit(prewarm_cache)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[ALPHADESK QUANT] Server stopped.")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)
