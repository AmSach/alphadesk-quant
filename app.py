import os, sys, time, threading, uuid, pandas as pd
from dotenv import load_dotenv; load_dotenv()

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# If run directly with `python app.py` (not via `streamlit run`), launch the clean, beautiful native Python-only UI (no HTML)
import streamlit as st
if not (hasattr(st, "runtime") and st.runtime.exists()):
    import gui
    gui.main()
    sys.exit(0)

import db, alerts, stress, insiders, quant, data, options
from llm import KeyPool
from debate import Debate
st.set_page_config("AlphaDesk",layout="wide")
sb=st.sidebar; kt=sb.text_area("API keys (one per line / comma)",os.getenv("LLM_KEYS","").replace(",","\n"),height=110)
keys=[k for k in kt.replace(",","\n").split("\n") if k.strip()]
if "pool" not in st.session_state or st.session_state.get("kt")!=kt: st.session_state.pool=KeyPool(keys); st.session_state.kt=kt
pool=st.session_state.pool; sb.caption(f"{len(pool.keys)} keys in rotation")
watch=[w.strip() for w in sb.text_area("Watchlist (AAPL, NSE:RELIANCE, BSE:500325, TCS.NS)",os.getenv("WATCHLIST","AAPL,NVDA,NSE:RELIANCE,NSE:TCS")).replace("\n",",").split(",") if w.strip()]
cap=sb.number_input("Capital",1000.0,1e10,100000.0,1000.0); risk=sb.slider("Risk per trade %",.1,3.0,1.0,.1); hor=sb.slider("Horizon (trading days)",3,30,10)
mins=sb.number_input("Auto-scan every N min (0=off)",0,240,15)
@st.cache_resource
def bg():
    s={"w":[],"m":0,"p":(100000,1.0,10)}
    def loop():
        while True:
            if s["m"]>0 and s["w"]:
                try: alerts.scan(s["w"],*s["p"])
                except Exception: pass
                time.sleep(s["m"]*60)
            else: time.sleep(5)
    threading.Thread(target=loop,daemon=True).start(); return s
b=bg(); b["w"],b["m"],b["p"]=watch,mins,(cap,risk,hor)
t1,t2,t3,t4,t5,t6=st.tabs(["Calls & Alerts","Levels & Model","Bull vs Bear","Insider radar","Call stress","Options / Black-Scholes"])
sb.caption("Data source: "+os.getenv("DATA_SOURCE","yahoo"))
with t1:
    if st.button("Scan now"):
        with st.spinner("Scanning (first run trains models, can take a while)"): st.session_state.calls=alerts.scan(watch,cap,risk,hor)
    if st.session_state.get("calls"):
        d=pd.DataFrame(st.session_state.calls); st.dataframe(d[["ticker","call","score","entry","stop","target","qty","p_win","p_loss","exp_R","model_prob","asof","why"]],use_container_width=True)
    st.subheader("Live scorecard of past BUY calls (target vs stop)")
    try:
        sc=quant.scorecard()
        if len(sc):
            dn=sc[sc.status.isin(["target","stop","expired"])]; st.dataframe(sc,use_container_width=True)
            if len(dn): st.write(f"Closed: {len(dn)} | hit target: {(dn.status=='target').mean():.0%} | avg return: {dn.ret_pct.mean():.2f}%")
    except Exception as e: st.warning(f"scorecard: {e}")
    a=db.load("alerts"); st.subheader("Alerts")
    if len(a): st.dataframe(a.sort_values("ts",ascending=False).assign(ts=lambda d:pd.to_datetime(d.ts,unit="s")),use_container_width=True)
with t2:
    lt=st.text_input("Ticker","NSE:RELIANCE",key="lt")
    if st.button("Compute levels + model"):
        try:
            with st.spinner("Working"): st.session_state.lv=(lt,alerts.make_call(lt,cap,risk,hor),quant.model_signal(quant.clean(data.hist(lt)),key=data.norm(lt)) if quant.clean(data.hist(lt)) is not None else {})
        except Exception as e: st.error(str(e))
    if "lv" in st.session_state:
        n,r,m=st.session_state.lv; st.subheader(f"{r['ticker']}: {r['call']}"); st.caption(r["why"])
        if "entry" in r:
            u=r["ccy"]; c=st.columns(4); c[0].metric("Entry",f"{r['entry']:.2f} {u}"); c[1].metric("Stop",f"{r['stop']:.2f}"); c[2].metric("Target",f"{r['target']:.2f}"); c[3].metric("Qty (risk-sized)",r["qty"])
            st.write(f"Risk:reward 1:{r['rr']:.2f} | P(target first) {r['p_win']:.0%} | P(stop first) {r['p_loss']:.0%} | P(neither in {r['horizon']}d) {r['p_open']:.0%} | expectancy {r['exp_R']:+.2f}R")
            st.write(f"{r['horizon']}-day 1-sigma range {r['sigma1_lo']:.2f} to {r['sigma1_hi']:.2f}; simulated 5/50/95th pct: {r['p5']:.2f} / {r['p50']:.2f} / {r['p95']:.2f}. Zero-drift simulation: this is risk sizing, not a forecast.")
        if m.get("ok"):
            st.write({k:(round(v,4) if isinstance(v,float) else v) for k,v in m.items() if k!="curve"}); st.line_chart(m["curve"])
            st.caption("Out-of-sample, after costs. 'trusted' only if statistically significant; otherwise the ML vote is ignored.")
        elif m: st.info(m.get("reason",""))
with t3:
    tk=st.text_input("Ticker","AAPL"); nk=data.norm(tk); sid=st.session_state.setdefault("sid",{}).setdefault(nk,str(uuid.uuid4())); c=st.columns(4)
    if c[0].button("Load data"):
        with st.spinner("Fetching filings/data"): st.session_state[f"ev_{nk}"]=data.evidence(tk)
    ev=st.session_state.get(f"ev_{nk}")
    if ev:
        try:
            d=Debate(sid,nk,ev,pool)
            if c[1].button("Next round"): d.round()
            if c[2].button("Judge"): d.verdict()
            if c[3].button("New debate"): st.session_state.sid[nk]=str(uuid.uuid4()); st.rerun()
            for r in d.hist().itertuples():
                with st.chat_message({"BULL":"👍","BEAR":"👎","JUDGE":"⚖️"}.get(r.role,"user")): st.markdown(f"**{r.role}** {r.content}")
            if q:=st.chat_input("Jump into the debate"): d.user(q); st.rerun()
        except Exception as e: st.error(str(e))
        with st.expander("Evidence fed to agents"): st.text(ev)
with t4:
    ct=st.text_input("Ticker for insider buys","AAPL")
    if st.button("Check"): r=insiders.cluster(ct); st.write({k:v for k,v in r.items() if k!="df"}); st.dataframe(r["df"])
with t5:
    f=st.file_uploader("Earnings call audio",["mp3","wav","m4a"]); tt=st.text_input("Ticker","AAPL",key="s"); dt=st.date_input("Call date")
    if f and st.button("Analyze"):
        try: open("/tmp/call","wb").write(f.read()); st.json(stress.analyze("/tmp/call",tt,str(dt)))
        except Exception as e: st.error(str(e))
    st.write(stress.predictive_power())
with t6:
    ot=st.text_input("Underlying","AAPL",key="ot")
    if st.button("Load expiries"):
        try: st.session_state.exps=options.expiries(ot)
        except Exception as e: st.error(f"{e}  (Yahoo has no NSE/BSE options; use DATA_SOURCE=bloomberg for those)")
    if st.session_state.get("exps"):
        ex=st.selectbox("Expiry",st.session_state.exps)
        if st.button("Analyze chain"):
            try:
                h=quant.clean(data.hist(ot)); sp=float(data.hist(ot).Close.iloc[-1]); res=options.analyze(options.get_chain(ot,ex),sp,options.rf(ot),h=h)
                if not res["ok"]: st.warning(res["reason"])
                else:
                    c=st.columns(4); c[0].metric("ATM IV",f"{res['atm_iv']:.1%}"); c[1].metric("Straddle-implied move",f"{res.get('implied_move_pct',float('nan')):.1%}"); c[2].metric("25d skew (put-call IV)",f"{res.get('skew_25d',float('nan')):+.1%}"); c[3].metric("Put/call OI",f"{res['pcr_oi']:.2f}" if res["pcr_oi"] else "n/a")
                    if "vol_risk_premium" in res: st.write(f"GARCH vol forecast {res['garch_vol_ann']:.1%} vs implied {res['atm_iv']:.1%}: vol risk premium {res['vol_risk_premium']:+.1%} (positive = options rich vs forecast, negative = cheap).")
                    st.dataframe(res["chain"],use_container_width=True)
            except Exception as e: st.error(str(e))
    with st.expander("Black-Scholes calculator"):
        c=st.columns(6); S=c[0].number_input("Spot",1.0,1e7,100.0); K=c[1].number_input("Strike",1.0,1e7,100.0); T=c[2].number_input("Days",1,1500,30)/365; r_=c[3].number_input("Rate",0.0,.5,.05); s_=c[4].number_input("Vol",.01,3.0,.25); cp=c[5].selectbox("Type",["c","p"])
        st.write({"price":round(options.bs(S,K,T,r_,s_,cp),4),**{k:round(v,5) for k,v in options.greeks(S,K,T,r_,s_,cp).items()},"risk-neutral P(S_T>K)":round(options.rn_prob_above(S,K,T,r_,s_),4)})
