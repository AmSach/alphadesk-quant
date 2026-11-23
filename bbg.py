"""Bloomberg Desktop API adapter (blpapi). Needs a logged-in Terminal on this machine.
Field names/overrides below follow Bloomberg docs but are UNTESTED here; verify any rejected field with FLDS<GO>."""
import os, datetime as dt, pandas as pd
try: import blpapi
except ImportError: blpapi=None
def available(): return blpapi is not None
def sym(n):
    """n = normalized yahoo-style symbol (TCS.NS, 500325.BO, AAPL, VOD.L)."""
    n=n.upper(); suf={"NS":"IN","BO":"IB","L":"LN","T":"JT","HK":"HK","DE":"GR","PA":"FP","TO":"CT","AX":"AU"}
    if "." in n: b,s=n.rsplit(".",1); return f"{b} {suf.get(s,'US')} Equity"
    return f"{n} US Equity"
def _sess():
    if blpapi is None: raise RuntimeError("blpapi not installed")
    o=blpapi.SessionOptions(); o.setServerHost(os.getenv("BBG_HOST","localhost")); o.setServerPort(int(os.getenv("BBG_PORT",8194)))
    s=blpapi.Session(o)
    if not s.start(): raise RuntimeError("Bloomberg session failed: is the Terminal logged in on this machine?")
    if not s.openService("//blp/refdata"): s.stop(); raise RuntimeError("could not open //blp/refdata")
    return s
def _send(s,req):
    s.sendRequest(req); out=[]
    while True:
        ev=s.nextEvent(15000)
        for m in ev:
            if m.hasElement("responseError"): raise RuntimeError(str(m.getElement("responseError")))
            out.append(m)
        if ev.eventType()==blpapi.Event.RESPONSE: return out
def hist(n,days=1825):
    s=_sess()
    try:
        r=s.getService("//blp/refdata").createRequest("HistoricalDataRequest"); r.getElement("securities").appendValue(sym(n))
        F={"PX_OPEN":"Open","PX_HIGH":"High","PX_LOW":"Low","PX_LAST":"Close","PX_VOLUME":"Volume"}
        for f in F: r.getElement("fields").appendValue(f)
        r.set("startDate",(dt.date.today()-dt.timedelta(days=days)).strftime("%Y%m%d")); r.set("endDate",dt.date.today().strftime("%Y%m%d"))
        r.set("periodicitySelection","DAILY"); r.set("adjustmentSplit",True); r.set("adjustmentNormal",True); r.set("adjustmentAbnormal",True)
        rows=[]
        for m in _send(s,r):
            if not m.hasElement("securityData"): continue
            fd=m.getElement("securityData").getElement("fieldData")
            for i in range(fd.numValues()):
                x=fd.getValueAsElement(i); rows.append({"date":pd.Timestamp(x.getElementAsDatetime("date")),**{v:(x.getElementAsFloat(k) if x.hasElement(k) else None) for k,v in F.items()}})
        if not rows: return None
        return pd.DataFrame(rows).set_index("date").sort_index()
    finally: s.stop()
def ref(secs,fields):
    s=_sess()
    try:
        r=s.getService("//blp/refdata").createRequest("ReferenceDataRequest")
        for x in secs: r.getElement("securities").appendValue(x)
        for f in fields: r.getElement("fields").appendValue(f)
        out={}
        for m in _send(s,r):
            if not m.hasElement("securityData"): continue
            arr=m.getElement("securityData")
            for i in range(arr.numValues()):
                sd=arr.getValueAsElement(i); fd=sd.getElement("fieldData"); row={}
                for f in fields:
                    if not fd.hasElement(f): continue
                    e=fd.getElement(f); row[f]=[e.getValueAsElement(j).getElement(0).getValueAsString() for j in range(e.numValues())] if e.isArray() else e.getValueAsString()
                out[sd.getElementAsString("security")]=row
        return out
    finally: s.stop()
def option_chain(n,max_n=400):
    """Returns DataFrame cp,strike,expiry,bid,ask,last,volume,oi using OPT_CHAIN bulk field."""
    u=sym(n); secs=ref([u],["OPT_CHAIN"]).get(u,{}).get("OPT_CHAIN",[])[:max_n]
    if not secs: raise RuntimeError("no OPT_CHAIN returned (check entitlements)")
    F=["PX_BID","PX_ASK","PX_LAST","OPT_STRIKE_PX","OPT_PUT_CALL","OPT_EXPIRE_DT","OPEN_INT","PX_VOLUME"]; rows=[]
    for sec,v in ref(secs,F).items():
        rows.append(dict(cp="c" if str(v.get("OPT_PUT_CALL","")).lower().startswith("c") else "p",strike=pd.to_numeric(v.get("OPT_STRIKE_PX"),errors="coerce"),expiry=pd.Timestamp(v.get("OPT_EXPIRE_DT")),
            bid=pd.to_numeric(v.get("PX_BID"),errors="coerce"),ask=pd.to_numeric(v.get("PX_ASK"),errors="coerce"),last=pd.to_numeric(v.get("PX_LAST"),errors="coerce"),volume=pd.to_numeric(v.get("PX_VOLUME"),errors="coerce"),oi=pd.to_numeric(v.get("OPEN_INT"),errors="coerce")))
    return pd.DataFrame(rows).dropna(subset=["strike","expiry"])
