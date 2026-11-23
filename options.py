import os, numpy as np, pandas as pd, yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import data, quant
def rf(t): return float(os.getenv("RISK_FREE_INR",.065) if data.norm(t).endswith((".NS",".BO")) else os.getenv("RISK_FREE_USD",.045))
def _d(S,K,T,r,q,s): d1=(np.log(S/K)+(r-q+.5*s*s)*T)/(s*np.sqrt(T)); return d1,d1-s*np.sqrt(T)
def bs(S,K,T,r,s,cp="c",q=0.0):
    """Black-Scholes-Merton (European, continuous dividend yield q)."""
    if T<=0 or s<=0: return max(S-K,0) if cp=="c" else max(K-S,0)
    d1,d2=_d(S,K,T,r,q,s)
    return S*np.exp(-q*T)*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2) if cp=="c" else K*np.exp(-r*T)*norm.cdf(-d2)-S*np.exp(-q*T)*norm.cdf(-d1)
def greeks(S,K,T,r,s,cp="c",q=0.0):
    d1,d2=_d(S,K,T,r,q,s); n1=norm.pdf(d1); eq,er=np.exp(-q*T),np.exp(-r*T); c=cp=="c"
    th=(-S*eq*n1*s/(2*np.sqrt(T))-r*K*er*norm.cdf(d2)+q*S*eq*norm.cdf(d1)) if c else (-S*eq*n1*s/(2*np.sqrt(T))+r*K*er*norm.cdf(-d2)-q*S*eq*norm.cdf(-d1))
    return dict(delta=eq*(norm.cdf(d1) if c else norm.cdf(d1)-1),gamma=eq*n1/(S*s*np.sqrt(T)),vega=S*eq*n1*np.sqrt(T)/100,theta=th/365,rho=(K*T*er*(norm.cdf(d2) if c else -norm.cdf(-d2)))/100)
def iv(price,S,K,T,r,cp="c",q=0.0):
    if T<=0 or price<=0: return np.nan
    lo=max((S*np.exp(-q*T)-K*np.exp(-r*T)) if cp=="c" else (K*np.exp(-r*T)-S*np.exp(-q*T)),0)
    if price<=lo+1e-8: return np.nan
    try: return brentq(lambda s:bs(S,K,T,r,s,cp,q)-price,1e-4,5.0,xtol=1e-8)
    except Exception: return np.nan
def rn_prob_above(S,K,T,r,s,q=0.0): return float(norm.cdf(_d(S,K,T,r,q,s)[1]))
def expiries(t):
    if os.getenv("DATA_SOURCE","yahoo").lower()=="bloomberg":
        c=_bbg(t); return sorted({str(x.date()) for x in c.expiry})
    return list(yf.Ticker(data.norm(t)).options)
def _bbg(t):
    import bbg; return data.cached(("oc",data.norm(t)),600,lambda:bbg.option_chain(data.norm(t)))
def get_chain(t,expiry):
    if os.getenv("DATA_SOURCE","yahoo").lower()=="bloomberg":
        c=_bbg(t); return c[c.expiry.dt.strftime("%Y-%m-%d")==expiry]
    oc=yf.Ticker(data.norm(t)).option_chain(expiry); fr=[]
    for cp,d in(("c",oc.calls),("p",oc.puts)):
        d=d.rename(columns={"lastPrice":"last","openInterest":"oi"}); 
        for col in("bid","ask","last","volume","oi"):
            if col not in d: d[col]=0
        fr.append(d[["strike","bid","ask","last","volume","oi"]].assign(cp=cp,expiry=pd.Timestamp(expiry)))
    return pd.concat(fr,ignore_index=True)
def analyze(df,spot,r,q=0.0,h=None,now=None):
    """Chain analytics: own IVs (not the vendor's), Greeks, ATM IV, straddle-implied move, skew, put/call ratios, vol risk premium."""
    now=now or pd.Timestamp.now(); d=df.copy()
    for c in("bid","ask","last","volume","oi"): d[c]=pd.to_numeric(d.get(c,0),errors="coerce").fillna(0)
    d["T"]=((d.expiry+pd.Timedelta(hours=10))-now).dt.total_seconds()/(365*86400); d=d[d["T"]>0.5/365]
    two=(d.bid>0)&(d.ask>=d.bid); d["mid"]=np.where(two,(d.bid+d.ask)/2,d["last"]); d=d[(d.mid>0)&(~two|((d.ask-d.bid)/d.mid.replace(0,np.nan)<.5))]
    d["iv"]=[iv(m,spot,k,T,r,cp,q) for m,k,T,cp in zip(d.mid,d.strike,d["T"],d.cp)]; d=d.dropna(subset=["iv"]); d=d[(d.iv>.02)&(d.iv<4)]
    if d.empty: return dict(ok=False,reason="no usable quotes after filtering (illiquid/expired/closed market)")
    g=pd.DataFrame([greeks(spot,k,T,r,s,cp,q) for k,T,s,cp in zip(d.strike,d["T"],d.iv,d.cp)],index=d.index); d=pd.concat([d,g],axis=1)
    T=float(d["T"].median()); k0=d.iloc[(d.strike-spot).abs().argsort()].strike.iloc[0]; a=d[d.strike==k0]; ca,pa=a[a.cp=="c"],a[a.cp=="p"]
    out=dict(ok=True,spot=spot,expiry=str(d.expiry.iloc[0].date()),T_years=T,atm_strike=float(k0),atm_iv=float(a.iv.mean()))
    if len(ca) and len(pa): out["straddle"]=float(ca.mid.iloc[0]+pa.mid.iloc[0]); out["implied_move_pct"]=out["straddle"]/spot
    out["sigma_move_pct"]=out["atm_iv"]*np.sqrt(T)
    p,c=d[(d.cp=="p")],d[(d.cp=="c")]
    if len(p) and len(c): out["skew_25d"]=float(p.iloc[(p.delta+.25).abs().argsort()].iv.iloc[0]-c.iloc[(c.delta-.25).abs().argsort()].iv.iloc[0])
    pv,cv,po,co=p.volume.sum(),c.volume.sum(),p.oi.sum(),c.oi.sum(); out["pcr_volume"]=float(pv/cv) if cv else None; out["pcr_oi"]=float(po/co) if co else None
    if h is not None:
        sg=quant.garch_sigma(h,int(np.clip(T*252,1,60)))
        if sg: out["garch_vol_ann"]=sg*np.sqrt(252); out["vol_risk_premium"]=out["atm_iv"]-out["garch_vol_ann"]
    out["chain"]=d[["cp","strike","mid","iv","delta","gamma","vega","theta","volume","oi"]].sort_values(["cp","strike"]).reset_index(drop=True); return out
