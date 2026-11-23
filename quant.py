import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np, pandas as pd, db, data
from scipy.stats import binomtest, norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
def clean(h):
    if h is None or len(h)<260: return None
    h=h[~h.index.duplicated()].dropna(subset=["High","Low","Close"]); h=h[h.Close>0]
    return h if len(h)>=260 else None
def rsi(c,n=14):
    d=c.diff(); g=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); l=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    o=100-100/(1+g/l.replace(0,np.nan)); o[(l==0)&(g>0)]=100; return o.fillna(50)
def atr(h,n=14):
    pc=h.Close.shift(); return pd.concat([h.High-h.Low,(h.High-pc).abs(),(h.Low-pc).abs()],axis=1).max(axis=1).ewm(alpha=1/n,adjust=False).mean()
def touch(S,B,sig,H,nu=None):
    """Exact GBM probability of touching barrier B at any time within H days (continuous monitoring, zero arithmetic drift by default). sig = daily vol."""
    nu=-.5*sig**2 if nu is None else nu; sd=sig*np.sqrt(H); b=abs(np.log(B/S))
    if b==0: return 1.0
    if B>S: return float(norm.cdf((-b+nu*H)/sd)+np.exp(2*nu*b/sig**2)*norm.cdf((-b-nu*H)/sd))
    return float(norm.cdf((-b-nu*H)/sd)+np.exp(-2*nu*b/sig**2)*norm.cdf((-b+nu*H)/sd))
def garch_sigma(h,horizon=10):
    """GARCH(1,1)-t average daily vol forecast over the horizon; None if arch missing or fit fails."""
    def go():
        try:
            from arch import arch_model
            r=100*np.log(h.Close).diff().dropna().tail(1000); f=arch_model(r,mean="Zero",vol="GARCH",p=1,q=1,dist="t").fit(disp="off",show_warning=False).forecast(horizon=horizon).variance.iloc[-1].values
            v=float(np.sqrt(f.mean())/100); return v if np.isfinite(v) and v>0 else None
        except Exception: return None
    return data.cached(("g",str(h.index[-1]),len(h),horizon),21600,go)
def levels(h,capital=100000,risk_pct=1.0,horizon=10,stop_atr=2.0,target_atr=3.0,direction=1,n=5000,seed=42,sigma=None):
    """ATR stop/target + block-bootstrap Monte Carlo (zero drift, current EWMA vol). Closes only, so intraday touches are understated."""
    c=h.Close; r=np.log(c).diff().dropna()
    if len(r)<250 or r.tail(756).std()==0: return None
    a=float(atr(h).iloc[-1]); entry=float(c.iloc[-1]); sig=float(sigma) if sigma else float(np.sqrt(r.ewm(alpha=.06).var().iloc[-1]))
    if not np.isfinite(a) or a<=0 or not np.isfinite(sig) or sig<=0: return None
    stop=entry-direction*stop_atr*a; target=entry+direction*target_atr*a
    if direction>0: stop=max(stop,entry*.01)
    else: target=max(target,entry*.01)
    rr=r.tail(756).values; rr=rr-rr.mean(); rr=rr*(sig/rr.std()); B=5; nb=-(-horizon//B)
    rng=np.random.default_rng(seed); idx=(rng.integers(0,len(rr)-B,(n,nb))[:,:,None]+np.arange(B)).reshape(n,-1)[:,:horizon]
    P=entry*np.exp(np.cumsum(rr[idx],1)); H=horizon
    ht,hs=(P>=target,P<=stop) if direction>0 else (P<=target,P>=stop)
    ti=np.where(ht.any(1),ht.argmax(1),H+1); si=np.where(hs.any(1),hs.argmax(1),H+1)
    pw=float((ti<si).mean()); pl=float(((si<=ti)&(si<=H)).mean()); RR=abs(target-entry)/max(abs(entry-stop),1e-9)
    risk_amt=capital*risk_pct/100; qty=int(max(0,min(risk_amt/max(abs(entry-stop),1e-9),capital/entry)))
    fin=P[:,-1]; mv=sig*np.sqrt(H)
    return dict(entry=entry,stop=float(stop),target=float(target),rr=RR,atr=a,daily_vol=sig,horizon=H,p_win=pw,p_loss=pl,p_open=1-pw-pl,exp_R=pw*RR-pl,
        touch_target=touch(entry,target,sig,H),touch_stop=touch(entry,stop,sig,H),qty=qty,risk_amt=qty*abs(entry-stop),sigma1_lo=entry*np.exp(-mv),sigma1_hi=entry*np.exp(mv),p5=float(np.percentile(fin,5)),p50=float(np.percentile(fin,50)),p95=float(np.percentile(fin,95)))
def feats(h):
    c=h.Close; r=c.pct_change(); f=pd.DataFrame(index=h.index)
    for k in(1,5,10,21,63): f[f"m{k}"]=c.pct_change(k)
    f["vol21"]=r.rolling(21).std(); f["volr"]=f.vol21/r.rolling(126).std(); f["rsi"]=rsi(c); f["d50"]=c/c.rolling(50).mean()-1; f["d200"]=c/c.rolling(200).mean()-1; f["atrp"]=atr(h)/c
    if "Volume" in h and (h.Volume>0).mean()>.9:
        v=np.log(h.Volume.replace(0,np.nan)); f["vz"]=(v-v.rolling(63).mean())/v.rolling(63).std()
    f=f.replace([np.inf,-np.inf],np.nan); return f.dropna(axis=1,how="all")
class Ens:
    def __init__(s): s.a=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500)); s.b=HistGradientBoostingClassifier(max_depth=3,learning_rate=.05,max_iter=150,l2_regularization=1.0,random_state=0)
    def fit(s,X,y): s.a.fit(X,y); s.b.fit(X,y); return s
    def predict_proba(s,X): return (s.a.predict_proba(X)+s.b.predict_proba(X))/2
def walk_forward(h,horizon=5,min_train=504,step=21,cost_bps=20,thr=.55):
    """Expanding-window, purged walk-forward. Every probability is out-of-sample. Costs deducted. Trusted only if statistically significant."""
    c=h.Close; f=feats(h); y=c.shift(-horizon)/c-1; ok=f.notna().all(axis=1)&y.notna(); X,R=f[ok],y[ok]; Y=(R>0).astype(int); n=len(X)
    if n<min_train+step*4: return dict(ok=False,reason="not enough history")
    pr=pd.Series(np.nan,index=X.index)
    for s in range(min_train,n,step):
        te=s-horizon; e=min(s+step,n)
        if Y.iloc[:te].nunique()<2: continue
        pr.iloc[s:e]=Ens().fit(X.iloc[:te],Y.iloc[:te]).predict_proba(X.iloc[s:e])[:,1]
    o=pr.dropna()
    if len(o)<252: return dict(ok=False,reason="not enough out-of-sample days")
    pos={d:j for j,d in enumerate(X.index)}; net=[]; dts=[]; nxt=-1
    for d,p in o.items():
        if pos[d]>=nxt and p>thr: net.append(float(R[d])-cost_bps/1e4); dts.append(d); nxt=pos[d]+horizon
    nt=len(net); base=float(Y[o.index].mean()); acc=float(((o>.5).astype(int)==Y[o.index]).mean())
    k=sum(x>0 for x in net); pv=float(binomtest(k,nt,base,alternative="greater").pvalue) if nt else 1.0
    eq=np.cumprod(1+np.array(net)) if nt else np.array([1.0]); avg=float(np.mean(net)) if nt else 0.0
    dd=float((eq/np.maximum.accumulate(eq)-1).min()); sh=float(np.mean(net)/np.std(net)*np.sqrt(252/horizon)) if nt>2 and np.std(net)>0 else 0.0
    curve=pd.DataFrame({"strategy":pd.Series(eq,index=dts if nt else [o.index[0]]).reindex(o.index,method="ffill").fillna(1.0),"buy_hold":c[o.index]/c[o.index[0]]})
    return dict(ok=True,trusted=bool(nt>=30 and avg>0 and pv<.01 and eq[-1]>1),n_trades=nt,win_rate=k/nt if nt else 0,base_rate=base,accuracy=acc,avg_net_ret=avg,
        total_ret=float(eq[-1]-1),buy_hold_ret=float(c[o.index[-1]]/c[o.index[0]]-1),max_dd=dd,sharpe=sh,p_value=pv,oos_days=len(o),curve=curve)
def model_signal(h,key="x",horizon=5):
    def go():
        w=walk_forward(h,horizon)
        if not w.get("ok"): return w
        f=feats(h); y=h.Close.shift(-horizon)/h.Close-1; ok=f.notna().all(axis=1)&y.notna()
        if f.iloc[-1].isna().any() or (y[ok]>0).nunique()<2: return dict(w,prob=None,trusted=False)
        return dict(w,prob=float(Ens().fit(f[ok],(y[ok]>0).astype(int)).predict_proba(f.iloc[[-1]])[:,1][0]))
    return data.cached(("m",key,str(h.index[-1]),len(h),horizon),21600,go)
def scorecard():
    """Live, honest tally of logged BUY calls: did price hit target or stop first?"""
    d=db.load("signals")
    if not len(d): return pd.DataFrame()
    d=d[d.call=="BUY"].drop_duplicates(["ticker","day"]); rows=[]
    for r in d.itertuples():
        h=data.hist(r.ticker)
        if h is None: continue
        ed=pd.Timestamp(r.ts,unit="s").normalize(); w=h[h.index>ed].iloc[:int(r.horizon)]; st,ex="open",None
        for _,b in w.iterrows():
            if b.Low<=r.stop: st,ex="stop",r.stop; break
            if b.High>=r.target: st,ex="target",r.target; break
        else:
            if len(w): ex=float(w.Close.iloc[-1]); st="expired" if len(w)>=r.horizon else "open"
        rows.append(dict(ticker=r.ticker,date=str(ed.date()),entry=r.entry,stop=r.stop,target=r.target,status=st,ret_pct=None if ex is None else (ex/r.entry-1)*100))
    return pd.DataFrame(rows)
