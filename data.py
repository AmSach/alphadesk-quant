import os, re, time, requests, yfinance as yf, pandas as pd
UA={"User-Agent":os.getenv("SEC_UA","AlphaDesk you@example.com")}
_C={}
def cached(key,ttl,fn):
    v=_C.get(key)
    if v and time.time()-v[0]<ttl: return v[1]
    r=fn(); _C[key]=(time.time(),r); return r
_BSE_MAP = {
    "500325": "RELIANCE.NS", "532540": "TCS.BO", "500209": "INFY.BO", "500180": "HDFCBANK.BO",
    "532667": "SUZLON.NS", "544026": "IREDA.NS", "540691": "CDSL.NS", "543235": "ANGELONE.NS",
    "543664": "KAYNES.NS", "543237": "MAZDOCK.NS", "540776": "BSE.NS", "532523": "BIOCON.NS"
}
def norm(t):
    t=re.sub(r"\s+","",str(t)).upper()
    if t.startswith("NSE:"): t=t[4:]; return t if t.endswith(".NS") else t+".NS"
    if t.startswith("BSE:"):
        t=t[4:]
        if t in _BSE_MAP: return _BSE_MAP[t]
        return t if t.endswith(".BO") else t+".BO"
    if t in _BSE_MAP: return _BSE_MAP[t]
    return t
def is_us(t): n=norm(t); return "." not in n and not n.startswith("^")
def ccy(t): n=norm(t); return "INR" if n.endswith((".NS",".BO")) else "local/USD"
SRC={}
def hist(t,period="5y"):
    def f():
        h=None
        if os.getenv("DATA_SOURCE","yahoo").lower()=="bloomberg":
            try:
                import bbg; h=bbg.hist(norm(t),365*int(re.sub(r"\D","",period) or 5)); SRC[norm(t)]="bloomberg"
            except Exception as e: h=None; SRC[norm(t)]=f"yahoo (bloomberg failed: {str(e)[:60]})"
        if h is None:
            try: h=yf.Ticker(norm(t)).history(period=period,auto_adjust=True); SRC.setdefault(norm(t),"yahoo")
            except Exception: return None
        if h is None or len(h)==0 or "Close" not in h: return None
        i=pd.DatetimeIndex(h.index); i=i.tz_localize(None) if i.tz is not None else i
        h=h.copy(); h.index=i.normalize(); h=h[~h.index.duplicated(keep="last")].dropna(subset=["Close"])
        return h if len(h) else None
    return cached(("h",norm(t),period),300,f)
def tech(t):
    h=hist(t)
    if h is None or len(h)<30: return {}
    c=h.Close; d=c.diff(); g=d.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); l=(-d.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rsi=50.0 if l.iloc[-1]==0 and g.iloc[-1]==0 else 100.0 if l.iloc[-1]==0 else float(100-100/(1+g.iloc[-1]/l.iloc[-1]))
    s50=c.rolling(50).mean().iloc[-1]; s200=c.rolling(200).mean().iloc[-1]
    return dict(price=float(c.iloc[-1]),asof=str(h.index[-1].date()),stale=(pd.Timestamp.now().normalize()-h.index[-1]).days>5,rsi=rsi,
        sma50=None if pd.isna(s50) else float(s50),sma200=None if pd.isna(s200) else float(s200),
        chg_1m=float(c.iloc[-1]/c.iloc[-22]-1) if len(c)>22 else None,chg_1y=float(c.iloc[-1]/c.iloc[max(-252,-len(c))]-1))
_cik=None
def cik(t):
    global _cik
    if _cik is None: _cik={v["ticker"]:str(v["cik_str"]).zfill(10) for v in requests.get("https://www.sec.gov/files/company_tickers.json",headers=UA,timeout=30).json().values()}
    return _cik.get(norm(t))
def _txt(html): return re.sub(r"\s+"," ",re.sub(r"<[^>]+>"," ",re.sub(r"(?s)<(script|style).*?</\1>","",html)))
def sec_text(t):
    c=cik(t)
    if not c: return ""
    s=requests.get(f"https://data.sec.gov/submissions/CIK{c}.json",headers=UA,timeout=30).json()["filings"]["recent"]
    out=[]
    for form in("10-K","10-Q"):
        for i,f in enumerate(s["form"]):
            if f==form:
                a=s["accessionNumber"][i].replace("-",""); u=f"https://www.sec.gov/Archives/edgar/data/{int(c)}/{a}/{s['primaryDocument'][i]}"
                time.sleep(.15); x=_txt(requests.get(u,headers=UA,timeout=60).text)
                for key in("Risk Factors","Management’s Discussion","Management's Discussion"):
                    m=[q.start() for q in re.finditer(key,x)]
                    if len(m)>1: out.append(f"[SEC {form} {s['filingDate'][i]} | {key}] "+x[m[1]:m[1]+3500]); break
                break
    return "\n".join(out)
def evidence(t):
    k=yf.Ticker(norm(t)); i=k.info or {}; keys=["longName","sector","industry","marketCap","trailingPE","forwardPE","priceToBook","profitMargins","revenueGrowth","earningsGrowth","debtToEquity","returnOnEquity","freeCashflow","dividendYield","fiftyTwoWeekHigh","fiftyTwoWeekLow","targetMeanPrice","recommendationKey"]
    ev=f"[YF fundamentals] "+"; ".join(f"{q}={i.get(q)}" for q in keys)+f"\n[YF technicals] {tech(t)}\n"
    try: ev+="[YF news] "+" | ".join(n.get("content",n).get("title","") for n in (k.news or [])[:8])+"\n"
    except Exception: pass
    try: ev+="[YF income stmt, latest cols]\n"+k.income_stmt.iloc[:12,:2].to_string()+"\n"
    except Exception: pass
    if is_us(t):
        try: ev+=sec_text(t)
        except Exception as e: ev+=f"[SEC unavailable: {e}]"
    else: ev+="[NOTE] No SEC filings for non-US listing; BSE/NSE filings not ingested, rely on YF data above."
    return ev
