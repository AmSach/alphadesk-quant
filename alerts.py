import os, time, requests, pandas as pd, db, data, quant
from datetime import date
from insiders import cluster
from debate import last_winner
def notify(msg,ticker=""):
    a=db.load("alerts")
    if len(a) and ((a.msg==msg)&(a.ts>time.time()-21600)).any(): return
    db.save("alerts",dict(ts=time.time(),ticker=ticker,msg=msg))
    if os.getenv("TELEGRAM_TOKEN"):
        try: requests.post(f"https://api.telegram.org/bot{os.getenv('TELEGRAM_TOKEN')}/sendMessage",json=dict(chat_id=os.getenv("TELEGRAM_CHAT"),text=msg),timeout=15)
        except Exception: pass
def make_call(t,capital=100000,risk=1.0,horizon=10):
    t=data.norm(t); h=quant.clean(data.hist(t))
    if h is None: return dict(ticker=t,call="NO DATA",why="no or <260 days of price history (bad symbol, new listing, delisted)")
    k=data.tech(t)
    if not k or k["stale"]: return dict(ticker=t,call="STALE",why=f"last price bar {k.get('asof')} is old (halted/delisted?)")
    sc,why=0,[]
    if k["rsi"]<30: sc+=1; why.append(f"RSI {k['rsi']:.0f} oversold")
    elif k["rsi"]>70: sc-=1; why.append(f"RSI {k['rsi']:.0f} overbought")
    if k["sma50"] and k["sma200"]: sc+=1 if k["sma50"]>k["sma200"] else -1; why.append("50DMA>200DMA" if k["sma50"]>k["sma200"] else "50DMA<200DMA")
    if k["sma200"]: sc+=1 if k["price"]>k["sma200"] else -1; why.append("above 200DMA" if k["price"]>k["sma200"] else "below 200DMA")
    c=cluster(t)
    if c.get("err"): why.append("insider data unavailable")
    elif c["flag"]: sc+=2; why.append(f"insider cluster: {c['n']} insiders in 14d, {c['total']:,.0f} {data.ccy(t)}")
    w=last_winner(t)
    if w in("BULL","BEAR"): sc+=1 if w=="BULL" else -1; why.append(f"debate (<7d): {w.lower()} won")
    m=quant.model_signal(h,key=t); mp=None
    if m.get("ok") and m.get("prob") is not None:
        mp=m["prob"]
        if m["trusted"]:
            if mp>.55: sc+=1; why.append(f"ML up-prob {mp:.0%} (OOS-validated)")
            elif mp<.45: sc-=1; why.append(f"ML up-prob {mp:.0%} (OOS-validated)")
        else: why.append(f"ML has no proven edge here (OOS p={m['p_value']:.2f}) - ignored")
    call="BUY" if sc>=3 else "SELL" if sc<=-2 else "HOLD"
    lv=quant.levels(h,capital,risk,horizon,direction=-1 if call=="SELL" else 1,sigma=quant.garch_sigma(h,horizon))
    if lv is None: return dict(ticker=t,call="NO DATA",why="could not compute volatility/levels")
    if call=="BUY" and lv["exp_R"]<=0: call="HOLD"; why.append("downgraded: expected R<=0")
    return dict(ticker=t,call=call,score=sc,asof=k["asof"],why="; ".join(why),cluster=bool(c.get("flag")),model_prob=mp,model_trusted=bool(m.get("trusted")),ccy=data.ccy(t),**lv)
def scan(watch,capital=100000,risk=1.0,horizon=10):
    out=[]; seen=set()
    for t in [data.norm(x) for x in watch if str(x).strip()]:
        if t in seen: continue
        seen.add(t)
        try:
            r=make_call(t,capital,risk,horizon)
            if r["call"] in("NO DATA","STALE"): notify(f"{t}: {r['call']} - {r['why']}",t); continue
            p=db.load("signals"); prev=p[p.ticker==t].sort_values("ts").call.iloc[-1] if len(p) and (p.ticker==t).any() else None
            db.save("signals",dict(ts=time.time(),day=str(date.today()),ticker=t,call=r["call"],score=r["score"],entry=r["entry"],stop=r["stop"],target=r["target"],horizon=horizon,qty=r["qty"],p_win=r["p_win"],why=r["why"]))
            al=db.load("alerts"); rec=len(al)>0 and ((al.ticker==t)&(al.ts>time.time()-86400)&al.msg.str.contains("insider cluster")).any()
            if (r["cluster"] and not rec) or (prev and prev!=r["call"]):
                notify(f"{t}: {r['call']} (was {prev}) entry {r['entry']:.2f} stop {r['stop']:.2f} target {r['target']:.2f} | P(target first) {r['p_win']:.0%} | {r['why']}",t)
            out.append(r)
        except Exception as e: notify(f"{t}: scan error {str(e)[:100]}",t)
    return out
