"""
AlphaDesk Quant - Authentic Intraday CUDA Neural Training Pipeline
1. Ingests 60 days of real 5-minute OHLCV exchange bars across 50 institutional NSE equities
   (selected strictly via First-20-Day ADV >= INR 100 Cr & Daily Range >= 1.7%) plus 20 US momentum leaders.
2. Builds authentic chronological state transitions (deducting real 0.19% round-trip friction) into
   E:/Alphadesk Quant/real_replay_buffer.npz and trains DuelingQNetwork (deep_q_network.pth) with
   holdout validation gating on CUDA GPU.
3. Extracts C6 Morning-Leader VWAP-Pullback intraday events across NSE institutional leaders and trains
   the 3-seed IntradayAlphaCritic ensemble (intraday_alpha_critic.pth) to predict post-fee Net-R edge.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "E:/Alphadesk Quant")

from deep_quant_net import (
    neural_trader,
    DuelingQNetwork,
    IntradayAlphaCritic,
    MODEL_SAVE_PATH,
    INTRADAY_CRITIC_PATH,
    REAL_BUFFER_PATH,
    DEVICE,
)

NSE_CANDIDATES = [
    "SUZLON.NS", "IREDA.NS", "IRFC.NS", "RVNL.NS", "RAILTEL.NS", "HUDCO.NS",
    "NBCC.NS", "SJVN.NS", "NHPC.NS", "NLCINDIA.NS", "BEL.NS", "HAL.NS",
    "BDL.NS", "MAZDOCK.NS", "COCHINSHIP.NS", "GRSE.NS", "IRCTC.NS", "CONCOR.NS",
    "INDUSINDBK.NS", "IDFCFIRSTB.NS", "BANDHANBNK.NS", "RBLBANK.NS", "FEDERALBNK.NS",
    "KARURVYSYA.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS", "UNIONBANK.NS",
    "BANKINDIA.NS", "IOB.NS", "CENTRALBK.NS", "SBIN.NS",
    "HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS", "BAJFINANCE.NS",
    "JIOFIN.NS", "ANGELONE.NS", "CDSL.NS", "PFC.NS", "RECLTD.NS", "LICHSGFIN.NS",
    "MANAPPURAM.NS", "MUTHOOTFIN.NS", "M&MFIN.NS", "SHRIRAMFIN.NS", "CHOLAFIN.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "SAIL.NS", "NMDC.NS", "VEDL.NS", "HINDALCO.NS",
    "NATIONALUM.NS", "HINDCOPPER.NS", "JINDALSTEL.NS", "COALINDIA.NS",
    "IOC.NS", "BPCL.NS", "HINDPETRO.NS", "ONGC.NS", "GAIL.NS", "IGL.NS", "MGL.NS",
    "POWERGRID.NS", "NTPC.NS", "TATAPOWER.NS", "ADANIPOWER.NS", "CESC.NS", "JSWENERGY.NS",
    "ASHOKLEY.NS", "MOTHERSON.NS", "SONACOMS.NS", "EXIDEIND.NS", "BHARATFORG.NS",
    "ETERNAL.NS", "PAYTM.NS", "NYKAA.NS", "POLICYBZR.NS", "DELHIVERY.NS", "GMRAIRPORT.NS",
    "INFY.NS", "WIPRO.NS", "TECHM.NS", "HCLTECH.NS", "KPITTECH.NS", "BSOFT.NS", "CYIENT.NS",
    "INDUSTOWER.NS", "TATACOMM.NS", "BHARTIARTL.NS", "ADANIENT.NS",
    "SUNPHARMA.NS", "BIOCON.NS", "LAURUSLABS.NS", "AUROPHARMA.NS", "GRANULES.NS",
]

US_LEADERS = [
    "NVDA", "PLTR", "TSLA", "AMD", "AAPL", "MSFT", "META", "AMZN", "GOOGL", "AVGO",
    "NFLX", "SMCI", "COIN", "MARA", "SOFI", "UBER", "SHOP", "ARM", "MU", "MRVL",
]


def _fetch_5m(sym, price_filter=True):
    try:
        df = yf.Ticker(sym).history(period="60d", interval="5m")
        if df is None or len(df) < 800:
            return sym, None
        last_px = float(df["Close"].iloc[-1])
        if price_filter and not (20.0 <= last_px <= 2500.0):
            return sym, None
        df["date"] = df.index.date
        df["slot"] = df.index.hour * 60 + df.index.minute
        df["med_vol"] = df.groupby("slot")["Volume"].transform(
            lambda s: s.rolling(10, min_periods=3).median().shift(1)
        )
        return sym, df
    except Exception:
        return sym, None


def train_neural_network_on_universe():
    print("==========================================================================================")
    print(" ALPHADESK QUANT: AUTHENTIC 5-MIN INTRADAY CUDA NEURAL TRAINING PIPELINE")
    print("==========================================================================================")
    print(f"Compute Device   : {DEVICE}")
    print(f"DQN Weight Path  : {MODEL_SAVE_PATH}")
    print(f"Critic WeightPath: {INTRADAY_CRITIC_PATH}")
    print(f"Replay BufferPath: {REAL_BUFFER_PATH}")
    print("------------------------------------------------------------------------------------------")

    # 1. Fetch Benchmark & Equity Universe in parallel
    nifty = yf.Ticker("^NSEI").history(period="60d", interval="5m")
    nifty["date"] = nifty.index.date
    nifty_map = {
        d: (g["Close"].values - float(g["Open"].iloc[0])) / float(g["Open"].iloc[0])
        for d, g in nifty.groupby("date")
        if len(g) > 10
    }

    nse_raw = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        for sym, df in ex.map(lambda s: _fetch_5m(s, True), NSE_CANDIDATES):
            if df is not None:
                nse_raw[sym] = df

    all_dates = sorted(set(d for df in nse_raw.values() for d in df["date"].unique()))
    cutoff_liqd = all_dates[20]

    # Strict First-20-Day In-Sample Institutional Filter (ADV >= 100 Cr & Daily Range >= 1.7%)
    inst_nse = {}
    daily_aggs = {}
    for sym, df in nse_raw.items():
        first20 = df[df["date"] < cutoff_liqd]
        if len(first20) < 300:
            continue
        da0 = first20.groupby("date").agg({"Close": "last", "Volume": "sum", "High": "max", "Low": "min"})
        adv_cr = float((da0["Close"] * da0["Volume"]).mean()) / 1e7
        rng_pct = float(((da0["High"] - da0["Low"]) / da0["Close"]).mean())
        if adv_cr >= 100.0 and rng_pct >= 0.017:
            inst_nse[sym] = df
            da = df.groupby("date").agg({"High": "max", "Low": "min", "Close": "last", "Open": "first"})
            da["tr"] = np.maximum(
                da["High"] - da["Low"],
                np.maximum(abs(da["High"] - da["Close"].shift(1)), abs(da["Low"] - da["Close"].shift(1))),
            )
            da["atr_pct"] = (da["tr"].rolling(10, min_periods=3).mean() / da["Close"]).shift(1)
            da["prev_close"] = da["Close"].shift(1)
            da["sma10d"] = da["Close"].rolling(10, min_periods=3).mean().shift(1)
            daily_aggs[sym] = da

    us_raw = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        for sym, df in ex.map(lambda s: _fetch_5m(s, False), US_LEADERS):
            if df is not None:
                us_raw[sym] = df

    print(f"NSE Institutional High-Beta Universe: {len(inst_nse)} equities ({len(all_dates)} sessions)")
    print(f"US Global Momentum Benchmark Universe: {len(us_raw)} equities")

    # 2. Extract authentic 5-min DuelingQNetwork transitions across NSE + US
    timed_transitions = []
    combined_for_dqn = list(inst_nse.items()) + list(us_raw.items())
    for sym, df in combined_for_dqn:
        close = df["Close"].values.astype(np.float64)
        high = df["High"].values.astype(np.float64)
        low = df["Low"].values.astype(np.float64)
        vol = df["Volume"].values.astype(np.float64)
        times = df.index
        n = len(close)
        if n < 220:
            continue

        s_cache = {}
        # Subsample every 4th bar (20 min) across 60 days so full multi-month history fits cleanly
        for i in range(60, n - 7, 4):
            px = float(close[i])
            sma20 = float(np.mean(close[i - 20 : i]))
            sma50 = float(np.mean(close[i - 50 : i]))
            sma200 = float(np.mean(close[max(0, i - 200) : i]))
            tr = np.maximum(
                high[i - 14 : i] - low[i - 14 : i],
                np.maximum(abs(high[i - 14 : i] - close[i - 15 : i - 1]), abs(low[i - 14 : i] - close[i - 15 : i - 1])),
            )
            atr = max(float(np.mean(tr)), px * 0.0025)
            adv20 = float(np.mean(vol[i - 20 : i]))
            surge = float(vol[i]) / (adv20 + 1e-9)
            std5 = float(np.std(close[i - 5 : i]))
            std20 = float(np.std(close[i - 20 : i])) + 1e-9
            squeeze = std5 / std20
            diffs = np.diff(close[i - 15 : i])
            gains = float(np.sum(diffs[diffs > 0]))
            losses = float(np.sum(-diffs[diffs < 0]))
            rsi = 50.0 if (gains + losses) == 0 else 100.0 * gains / (gains + losses)
            ret_1 = (px - float(close[i - 1])) / float(close[i - 1])
            ret_6 = (px - float(close[i - 6])) / float(close[i - 6])
            ret_24 = (px - float(close[i - 24])) / float(close[i - 24])

            s = neural_trader.extract_features(
                px=px, sma20=sma20, sma50=sma50, sma200=sma200,
                atr14=atr, rsi14=rsi, squeeze_ratio=squeeze, vol_surge=surge,
                bb_width=std20 * 2.0 / (sma20 + 1e-5),
                ret_1d=ret_1, ret_5d=ret_6, ret_20d=ret_24,
            )
            px_n = float(close[i + 6])
            sma20_n = float(np.mean(close[i - 14 : i + 6]))
            s_next = neural_trader.extract_features(
                px=px_n, sma20=sma20_n, sma50=sma50, sma200=sma200,
                atr14=atr, rsi14=rsi, squeeze_ratio=squeeze, vol_surge=surge,
                bb_width=std20 * 2.0 / (sma20 + 1e-5),
                ret_1d=(px_n - float(close[i + 5])) / float(close[i + 5]),
                ret_5d=(px_n - px) / px,
                ret_20d=ret_24,
            )

            fwd_ret = (px_n - px) / px
            # Risk unit normalized to ~1.0% intraday stop; subtract full 0.19% round-trip cost
            r_long = float(np.clip((fwd_ret - 0.0019) / 0.01, -2.0, 2.5))
            r_short = float(np.clip((-fwd_ret - 0.0019) / 0.01, -2.0, 2.5))
            ts_key = times[i].timestamp()
            # Finite 30-min episode horizon (done=True keeps Bellman Q calibrated directly in Net R)
            timed_transitions.append((ts_key, s, 1, r_long, s_next, True))
            timed_transitions.append((ts_key, s, 2, r_short, s_next, True))
            timed_transitions.append((ts_key, s, 0, 0.0, s_next, True))

    # Sort chronologically so train/val split respects time ordering strictly
    timed_transitions.sort(key=lambda x: x[0])
    pure_transitions = [(t[1], t[2], t[3], t[4], t[5]) for t in timed_transitions]
    neural_trader.save_real_replay_buffer(pure_transitions)
    n_loaded = neural_trader.load_real_replay_buffer()
    print(f"\n[STEP 1] Saved & Loaded {n_loaded:,} chronological 5-min transitions (Train: {len(neural_trader.replay_buffer):,}, Holdout Val: {len(neural_trader.val_buffer):,})")

    # 3. Reset & Train DuelingQNetwork on CUDA GPU with Holdout Validation Checkpointing
    print("\n[STEP 2] Training DuelingQNetwork on CUDA GPU...")
    neural_trader.policy_net = DuelingQNetwork(16, 3).to(DEVICE)
    neural_trader.target_net = DuelingQNetwork(16, 3).to(DEVICE)
    neural_trader.target_net.load_state_dict(neural_trader.policy_net.state_dict())
    neural_trader.optimizer = optim.AdamW(neural_trader.policy_net.parameters(), lr=5e-4, weight_decay=1e-4)
    neural_trader.epochs_completed = 0
    neural_trader.batch_size = 256
    neural_trader.best_val_loss = neural_trader.evaluate_validation_loss()
    start_t = time.time()
    best_dqn_state = {k: v.cpu().clone() for k, v in neural_trader.policy_net.state_dict().items()}
    best_val = float("inf")

    for ep in range(1, 3001):
        loss = neural_trader.train_step(allow_checkpoint=False)
        if ep % 500 == 0:
            vloss = neural_trader.evaluate_validation_loss()
            if vloss < best_val:
                best_val = vloss
                best_dqn_state = {k: v.cpu().clone() for k, v in neural_trader.policy_net.state_dict().items()}
            print(f"  Epoch {ep:>4}/3000 | TrainLoss: {loss:.5f} | ValLoss: {vloss:.5f} | BestVal: {best_val:.5f} | {time.time()-start_t:.1f}s")

    for pg in neural_trader.optimizer.param_groups:
        pg["lr"] = 1e-4

    neural_trader.policy_net.load_state_dict({k: v.to(DEVICE) for k, v in best_dqn_state.items()})
    neural_trader.target_net.load_state_dict(neural_trader.policy_net.state_dict())
    neural_trader.best_val_loss = best_val
    torch.save(neural_trader.policy_net.state_dict(), MODEL_SAVE_PATH)
    print(f"  Saved best validation DuelingQNetwork -> {MODEL_SAVE_PATH}")

    # 4. Extract C6 Intraday VWAP-Pullback Setups across 50 NSE Institutional Equities & Train IntradayAlphaCritic
    print("\n[STEP 3] Training 3-Seed IntradayAlphaCritic Ensemble on C6 Institutional Setups...")
    c6_events = []
    for sym, df in inst_nse.items():
        da = daily_aggs[sym]
        for d, day_df in df.groupby("date"):
            if len(day_df) < 35 or d not in da.index:
                continue
            di = da.loc[d]
            da_pct = float(di["atr_pct"]) if not pd.isna(di["atr_pct"]) else 0.022
            if da_pct < 0.020:
                continue
            pc = float(di["prev_close"]) if not pd.isna(di["prev_close"]) else float(day_df["Open"].iloc[0])
            sm10 = float(di["sma10d"]) if not pd.isna(di["sma10d"]) else pc
            opens = day_df["Open"].values
            highs = day_df["High"].values
            lows = day_df["Low"].values
            closes = day_df["Close"].values
            vols = day_df["Volume"].values
            med_vols = day_df["med_vol"].values
            vwap = np.cumsum(((highs + lows + closes) / 3.0) * vols) / (np.cumsum(vols) + 1e-9)

            o0 = float(opens[0])
            gap_pct = (o0 - pc) / pc
            mv = (float(closes[5]) - o0) / o0
            mv_prev = (float(closes[5]) - pc) / pc
            if abs(mv) < 0.013:
                continue
            hi_open = float(np.max(highs[:6]))
            lo_open = float(np.min(lows[:6]))

            in_pos = False
            side = ""
            entry_px = sl = tp_px = 0.0
            r_pct = 0.01
            rec = None
            last_ex = -20
            for i in range(6, len(day_df) - 1):
                c = float(closes[i])
                c_prev = float(closes[i - 1])
                h = float(highs[i])
                l = float(lows[i])
                vw = float(vwap[i])
                if in_pos:
                    closed = False
                    ex_px = c
                    if side == "BUY":
                        if l <= sl:
                            closed = True; ex_px = sl
                        elif h >= tp_px:
                            closed = True; ex_px = tp_px
                    else:
                        if h >= sl:
                            closed = True; ex_px = sl
                        elif l <= tp_px:
                            closed = True; ex_px = tp_px
                    if not closed and i >= len(day_df) - 2:
                        closed = True; ex_px = c
                    if closed:
                        ret_g = (ex_px - entry_px) / entry_px if side == "BUY" else (entry_px - ex_px) / entry_px
                        rec["r_net"] = (ret_g - 0.0019) / r_pct
                        c6_events.append(rec)
                        in_pos = False
                        last_ex = i
                elif i <= 48 and i - last_ex >= 6:
                    sig = (
                        "BUY" if (mv > 0 and l <= vw * 1.0025 and c > vw and c > c_prev)
                        else ("SELL" if (mv < 0 and h >= vw * 0.9975 and c < vw and c < c_prev) else None)
                    )
                    if sig:
                        mv_v = med_vols[i]
                        rvol = float(vols[i]) / float(mv_v) if (mv_v and not np.isnan(mv_v) and mv_v > 0) else 1.0
                        n_ret = float(nifty_map[d][min(i, len(nifty_map[d]) - 1)]) if d in nifty_map else 0.0
                        sma20 = float(np.mean(closes[max(0, i - 20) : i + 1]))
                        diffs = np.diff(closes[max(0, i - 14) : i + 1])
                        g = np.sum(diffs[diffs > 0]); ls = np.sum(-diffs[diffs < 0])
                        rsi = 50.0 if (g + ls) == 0 else float(100.0 * g / (g + ls))
                        std5 = float(np.std(closes[max(0, i - 5) : i + 1]))
                        std20 = float(np.std(closes[max(0, i - 20) : i + 1]))
                        feats = neural_trader.extract_intraday_features(
                            side=sig, abs_mv=abs(mv), mv_prev=mv_prev, gap_pct=gap_pct,
                            c=c, vw=vw, o0=o0, pc=pc, sm10=sm10, da_pct=da_pct, sma20=sma20,
                            rsi=rsi, std5=std5, std20=std20, rvol=rvol,
                            day_hi=float(np.max(highs[: i + 1])), day_lo=float(np.min(lows[: i + 1])),
                            bar_open=float(opens[i]), bar_high=h, bar_low=l,
                            hi_open=hi_open, lo_open=lo_open, n_ret=n_ret, bar_idx=i,
                        )
                        sl_p = float(np.clip(da_pct * 1.2 * 0.45, 0.008, 0.022))
                        tp_p = sl_p * (3.0 / 1.2)
                        r_pct = sl_p
                        entry_px = c
                        side = sig
                        in_pos = True
                        sl = round(entry_px * (1 - sl_p), 2) if sig == "BUY" else round(entry_px * (1 + sl_p), 2)
                        tp_px = round(entry_px * (1 + tp_p), 2) if sig == "BUY" else round(entry_px * (1 - tp_p), 2)
                        rec = {"date": d, "feats": feats}

    c6_events.sort(key=lambda x: x["date"])
    X_all = torch.tensor(np.array([e["feats"] for e in c6_events]), dtype=torch.float32, device=DEVICE)
    R_all = torch.tensor(np.array([np.clip(e["r_net"], -1.5, 2.5) for e in c6_events]), dtype=torch.float32, device=DEVICE)
    y_all = torch.stack([torch.zeros_like(R_all), R_all], dim=1)

    critic_states = []
    critic_models = []
    loss_fn = nn.SmoothL1Loss()
    for seed in [11, 42, 99]:
        torch.manual_seed(seed)
        m = IntradayAlphaCritic(16).to(DEVICE)
        opt = optim.AdamW(m.parameters(), lr=8e-4, weight_decay=3e-2)
        for _ in range(240):
            m.train()
            opt.zero_grad()
            l = loss_fn(m(X_all), y_all)
            l.backward()
            opt.step()
        m.eval()
        critic_models.append(m)
        critic_states.append({k: v.cpu() for k, v in m.state_dict().items()})

    torch.save({
        "models": critic_states,
        "trained_through": str(all_dates[-1]),
        "num_c6_events": len(c6_events),
    }, INTRADAY_CRITIC_PATH)
    neural_trader.critic_models = critic_models

    with torch.no_grad():
        sample_q = neural_trader.score_intraday_setup(c6_events[-1]["feats"])
    print(f"  Trained 3-seed IntradayAlphaCritic on {len(c6_events)} C6 events through {all_dates[-1]}")
    print(f"  Sample Live Critic Edge Score (last setup): {sample_q:+.4f}R post-fee advantage")
    print("==========================================================================================")

    return {
        "status": "SUCCESS",
        "dqn_transitions": n_loaded,
        "c6_events": len(c6_events),
        "best_dqn_val_loss": round(best_val, 5),
        "device": str(DEVICE),
    }


if __name__ == "__main__":
    train_neural_network_on_universe()
