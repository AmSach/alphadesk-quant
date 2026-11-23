# AlphaDesk

One Streamlit app that combines: a bull-vs-bear LLM debate on real filings, an insider cluster-buy radar, an earnings-call stress detector, volatility-based targets/stop-losses, a walk-forward ML check, an options (Black-Scholes) toolkit, and an alerts subsystem. Works for US tickers and Indian NSE/BSE tickers.

**Read this first:** nothing here guarantees profit. The tool is built to *measure* its own accuracy (walk-forward test + live scorecard) and to refuse to trust itself when the numbers don't support it. Paper-trade it before risking money. Not financial advice.

---

## 1. Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # fill it in

# Launch the clean, beautiful native Python-only UI (PySide6 / Qt 6, no HTML):
python gui.py
# or:
python app.py

# Run the test suite:
python test_quant.py

# Optional: Streamlit web UI
streamlit run app.py
```

`.env` settings:

| Variable | Meaning |
|---|---|
| `LLM_KEYS` | Comma-separated API keys (as many as you want). Rotated automatically. |
| `LLM_URL`, `LLM_MODEL` | Any OpenAI-compatible endpoint. Default is Groq + `llama-3.3-70b-versatile`. |
| `SEC_UA` | **Required by SEC**: `Your Name your@email.com`. Without a real one EDGAR may block you. |
| `DATABASE_URL` | `sqlite:///quant.db` (default) or `postgresql+psycopg2://user:pass@host/db`. |
| `TELEGRAM_TOKEN`, `TELEGRAM_CHAT` | Optional alert delivery. |
| `WATCHLIST` | Default tickers for scanning. |
| `DATA_SOURCE` | `yahoo` (default) or `bloomberg`. |
| `BBG_HOST`, `BBG_PORT` | Bloomberg Desktop API location (default localhost:8194). |
| `RISK_FREE_USD`, `RISK_FREE_INR` | Used in Black-Scholes. **Update to current rates**; defaults are guesses. |

**Ticker format:** `AAPL` (US), `NSE:RELIANCE` or `RELIANCE.NS`, `BSE:500325` or `500325.BO`, any Yahoo symbol (`VOD.L`, `7203.T`). Spaces and case are cleaned up.

**Bloomberg:** run the app on the terminal machine with the Terminal logged in; install blpapi with the pip line in `requirements.txt`; set `DATA_SOURCE=bloomberg`. If it fails, price data falls back to Yahoo and the sidebar/source note says so. Check your college's data-use rules before automated pulls.

---

## 2. File map

| File | Job |
|---|---|
| `gui.py` | Clean, beautiful native Python-only UI (PySide6 / Qt 6, zero HTML, 6 tabs, interactive chart) |
| `app.py` | Launcher (`python app.py` launches the native Python UI; `streamlit run app.py` runs Streamlit web) |
| `test_quant.py` | Quantitative verification test suite (Black-Scholes, Greeks, IV, ATR, Monte Carlo, walk-forward, DB) |
| `data.py` | Ticker cleanup, price history (Yahoo/Bloomberg), indicators, SEC filing text, evidence bundle for the debate, TTL cache |
| `llm.py` | API-key pool with rotation, cooldowns and retries |
| `debate.py` | Bull / Bear / Judge agents with persistent memory |
| `insiders.py` | Insider cluster-buy detection (SEC Form 4, NSE disclosures) |
| `stress.py` | Earnings-call stress score + check against next-day move |
| `quant.py` | ATR levels, Monte Carlo, barrier probabilities, GARCH, walk-forward ML, live scorecard |
| `options.py` | Black-Scholes, Greeks, implied vol, chain analytics |
| `bbg.py` | Bloomberg adapter (prices, option chain) |
| `alerts.py` | Combines signals into BUY/HOLD/SELL, logs them, sends alerts |
| `db.py` | Thin pandas/SQLAlchemy storage (works on SQLite and PostgreSQL) |

Tables: `messages` (debate), `verdicts`, `signals` (every logged call with entry/stop/target), `alerts`, `calls` (earnings-call stress results).

---

## 3. The algorithms, what each does and why I chose it

### 3.1 Key rotation (`llm.py`)
Round-robins through your keys. HTTP 429 (rate limit) puts that key on cooldown for the server's `retry-after`; 401/403 (bad key) benches it for 24h; network errors/empty replies bench it briefly. It retries on the next key. *Why:* free-tier keys are rate-limited per key, so rotating multiplies throughput. Works with any OpenAI-style provider because that is the common protocol.

### 3.2 Bull vs Bear vs Judge (`debate.py`, `data.py`)
- **Evidence block:** Yahoo fundamentals (P/E, margins, growth, debt...), technicals, news headlines, income statement, and for US stocks the Risk Factors and MD&A sections of the latest 10-K and 10-Q from EDGAR (the second occurrence of the heading is used to skip the table of contents; this heuristic can miss).
- **Grounding:** agents are told to use only the evidence, cite source tags, and say "not in data" otherwise. This reduces hallucination; it does not eliminate it.
- **Memory:** every message is saved to the database; each turn the last 24 messages are fed back, so you can interrupt, ask questions, and both agents answer you and each other. Evidence is capped at 14,000 characters to fit free-tier token limits.
- **Judge:** outputs `WINNER: BULL/BEAR/DRAW` and reasoning. The winner is stored and counts as a +1/-1 vote in the call for 7 days.
- *Why two opposing agents:* forcing each side to argue the best case surfaces risks a single "balanced" answer glosses over. It is a research aid, not a predictor.

### 3.3 Insider cluster-buy radar (`insiders.py`)
- **Signal:** several different insiders buying with their own money at the same time. Selling has many innocent reasons (taxes, diversification); open-market buying has basically one.
- **US:** Form 4 filings from EDGAR; only transaction code `P` (open-market purchase). **India:** NSE's SEBI insider-trading (PIT) disclosure feed; buys only. NSE blocks scripts sometimes, so this fails soft ("insider data unavailable") instead of breaking.
- **Cluster rule:** at least 3 distinct insiders inside any rolling 14-day window, ignoring tiny buys (under $5,000 / ₹1 lakh) and removing duplicate/amended filings.
- Thresholds (3 insiders, 14 days) are judgment calls, not optimized.

### 3.4 Earnings-call stress detector (`stress.py`)
1. **Whisper** (faster-whisper "base") transcribes audio.
2. **librosa** extracts voice features: pitch variability (coefficient of variation of the fundamental frequency, via pYIN) and pause ratio (share of the clip that is silence).
3. **FinBERT** (finance-tuned sentiment model) gives the average "negative" probability over 1,500-character chunks.
4. **Score (0-100)** = 45% negativity + 30% pitch variability + 25% pauses, each normalized. **These weights are my heuristic, not fitted to data.**
5. It records the next-day stock return and the Stress tab shows the correlation across all calls you have analyzed. You need dozens of calls before that number means anything. Until then treat the score as descriptive.

### 3.5 Technical indicators (`data.py`, `quant.py`)
- **SMA 50/200:** average close over 50/200 days. 50 above 200 = uptrend ("golden cross" regime).
- **RSI(14):** momentum oscillator 0-100; <30 oversold, >70 overbought.
- **ATR(14):** average true range, the typical daily price swing in currency units. Used to size stops and targets so they adapt to each stock's volatility.

### 3.6 Entry / stop / target (`quant.levels`)
- Entry = last close. **Stop = 2 ATR** away, **target = 3 ATR** away (risk:reward ≈ 1:1.5). Shorts mirror. ATR-based stops are a standard approach: a fixed 5% stop is too tight for a volatile stock and too loose for a calm one.
- **Position size** = (capital × risk%) ÷ (entry − stop), capped by capital. This is "fixed fractional" sizing: a stop-out loses only your chosen % (default 1%).
- **"R"** = the amount risked. Target win = +1.5R, stop = -1R.

### 3.7 Monte Carlo odds (`quant.levels`)
Simulates 5,000 price paths for your horizon by **block bootstrap**: it resamples 5-day blocks of the last 3 years of the stock's *own* returns (blocks keep volatility clustering and short-term autocorrelation), removes the average drift (so it assumes **no edge**), and rescales to current volatility. For each path it checks whether target or stop is hit first. Output: P(target first), P(stop first), P(neither), expectancy in R = P(win)×1.5 − P(loss)×1, plus 5th/50th/95th percentile prices and a 1-sigma range. Seeded (42) so results are repeatable. *Why bootstrap instead of a normal distribution:* real returns have fat tails and clustering; bootstrap keeps them. *Limit:* uses closing prices only, so real intraday touches happen slightly more often.

### 3.8 Barrier-touch probability (`quant.touch`)
Closed-form probability that a geometric Brownian motion touches a level at any time within the horizon (reflection principle). Shown next to the Monte Carlo as a cross-check; it assumes continuous monitoring so it is a bit higher than the closes-only simulation. (Verified against a fine-step simulation.)

### 3.9 GARCH(1,1) with Student-t (`quant.garch_sigma`)
Volatility clusters (calm periods and wild periods persist). GARCH models tomorrow's variance as a base level + yesterday's shock + yesterday's variance, so the forecast reacts to recent turbulence and decays back to normal. Student-t errors allow fat tails. The averaged forecast over your horizon sets the volatility for the Monte Carlo and for the options comparison. Falls back to a simple exponentially-weighted estimate if the `arch` package or the fit fails.

### 3.10 Walk-forward ML (`quant.walk_forward`, `model_signal`)
- **Question:** will the stock be higher in 5 days?
- **Features:** 1/5/10/21/63-day returns, short- and relative volatility, RSI, distance from the 50/200-day averages, ATR%, volume z-score. All use only data available at that date (no look-ahead).
- **Model:** average of a regularized logistic regression (stable, hard to overfit) and a shallow gradient-boosted tree (captures non-linear interactions). An ensemble of a simple and a flexible model is a conservative choice for noisy financial data.
- **Walk-forward validation:** train on the past only, predict the next 21 days, roll forward, retrain. A purge gap equal to the horizon prevents training labels from overlapping test periods. **Every reported probability is truly out-of-sample.**
- **Trading simulation:** go long when P(up) > 55%, hold 5 days, no overlapping trades, **20 bps round-trip cost deducted** (a flat guess; Indian retail costs including STT, brokerage and GST can be higher, so raise `cost_bps` in `walk_forward` for realism).
- **Trust gate:** the ML vote is used only if there are ≥30 trades, positive average net return, positive total return, and the win rate beats the stock's base up-rate with p < 0.01 (binomial test). Otherwise the call says "ML has no proven edge" and ignores it. *Why so strict:* if you test many stocks, some look good by luck; the strict gate plus the live scorecard guard against that. Even so, expect occasional false positives.
- **Honest test results I ran (synthetic data):** on a pure random walk it found no edge and lost money after costs, which is the correct answer.

### 3.11 The combined call (`alerts.make_call`)
A transparent vote count, not a black box:

| Vote | Points |
|---|---|
| RSI < 30 / > 70 | +1 / -1 |
| 50DMA above / below 200DMA | +1 / -1 |
| Price above / below 200DMA | +1 / -1 |
| Insider cluster | +2 |
| Debate winner (last 7 days) | +1 / -1 |
| ML up-prob > 55% / < 45% (only if trusted) | +1 / -1 |

Score ≥ 3 = BUY, ≤ -2 = SELL, else HOLD. A BUY is downgraded to HOLD if the Monte Carlo expectancy is ≤ 0. **These point weights are judgment, not optimized or backtested.** Note the RSI vote is mean-reversion while the trend votes are trend-following; they can pull against each other, which is intentional dampening but worth knowing. Calls return `NO DATA`/`STALE` instead of guessing when history is too short (<260 days) or the last price is over 5 days old.

### 3.12 Live scorecard (`quant.scorecard`)
Every BUY is logged (entry, stop, target, horizon). The scorecard replays later price bars: whichever of target/stop is touched first decides the outcome (if both in one bar, stop wins, the conservative assumption); otherwise it expires at the horizon. **This is your real accuracy number.** One logged BUY per ticker per day.

### 3.13 Black-Scholes toolkit (`options.py`)
- **Black-Scholes-Merton** prices European options from spot, strike, time, rate, volatility and dividend yield. Verified against the textbook value (10.4506) and put-call parity.
- **Greeks:** delta (price change per 1 move in the stock), gamma (change in delta), vega (per 1 vol point), theta (per day), rho (per 1% rate). Verified against finite differences.
- **Implied volatility:** the vol that makes the model price equal the market price, solved numerically (Brent). I compute IVs myself from mid-quotes rather than trusting Yahoo's field, which is unreliable.
- **Chain analytics:** ATM IV; **straddle-implied move** (ATM call + put price ÷ spot ≈ market's expected absolute move); **25-delta skew** (put IV − call IV, a fear gauge); put/call ratios; and **vol risk premium** = implied vol − GARCH forecast (positive = options expensive relative to forecast, negative = cheap). Illiquid or wide-spread quotes are filtered out.
- *Limits:* constant-volatility, European-exercise model. US equity options are American; NSE equity-derivative options are European-style (verify contract specs), so BS fits those better. Outside market hours quotes are stale. Yahoo has no NSE/BSE options, so use Bloomberg for those. It is a pricing and risk tool, not a price predictor.

### 3.14 Alerts (`alerts.py`)
The scanner runs on demand or every N minutes. It alerts on: a change in call (BUY→HOLD etc.), a new insider cluster (max once per 24h per ticker), no-data/stale tickers, errors. Identical messages are suppressed for 6 hours. Delivery: UI table plus Telegram if configured.

---

## 4. Design decisions

- **Measure, don't claim:** walk-forward test, trust gate and live scorecard exist because most retail "prediction" systems are overfit. A tool that says "no edge here" is more useful than one that is always confident.
- **Zero-drift simulation:** assumes no expected return so the risk numbers don't smuggle in optimism.
- **Transparent scoring** over a single opaque model so you can see why a call was made (the `why` column).
- **pandas-based storage** so the same code runs on SQLite and PostgreSQL.
- **Fail soft:** bad ticker, blocked feed, empty LLM reply, missing library: each degrades to a clear message instead of a crash.
- **Caching + fixed seeds:** same inputs give the same outputs within a session, so calls don't flicker.

## 5. Known limitations (be aware)

- Yahoo data is unofficial and often delayed ~15 minutes; true real-time Indian data needs a broker feed or Bloomberg.
- The rule score weights, stress score weights and cluster thresholds are untuned heuristics.
- Running the walk-forward on many tickers produces some false "trusted" results by chance.
- Monte Carlo and the scorecard check daily closes/bars, not tick data; gaps can blow through stops (slippage is not modeled).
- LLM agents can still misread filings; check the "Evidence" expander.
- The NSE feed and Bloomberg field names may need adjustment; I could not test them from my environment.
- I tested everything on synthetic data and with mocks. Live Yahoo, SEC, NSE, Groq, Bloomberg and the Whisper/FinBERT pipeline are untested by me.

## 6. Suggested workflow

1. Add your keys and a watchlist; click **Scan now**.
2. Open **Levels & Model** for a ticker to see entry/stop/target, odds, position size, and the ML track record.
3. Use **Bull vs Bear** to stress-test the thesis; **Options** to see what the market implies.
4. Let the scorecard accumulate (weeks). Trade real money only if closed BUYs show a positive average return and a sensible hit rate, and then at small size.

## 7. Glossary

- **ATR:** typical daily price range. **SMA/DMA:** simple moving average. **RSI:** momentum gauge.
- **R:** your risked amount per trade. **Expectancy:** average outcome per trade in R.
- **Implied vol (IV):** volatility the options market is pricing in. **Realized vol:** what actually happened.
- **Walk-forward / out-of-sample:** testing only on data the model never saw in training.
- **Overfitting:** a model memorizing noise; looks great in the past, fails live.
- **Base rate:** how often a stock simply goes up over the horizon, the number a model must beat.
