import time, requests, pandas as pd, xml.etree.ElementTree as ET
from datetime import date, timedelta
from data import UA, cik, norm, is_us
def parse4(x):
    r=ET.fromstring(x); n=r.findtext(".//rptOwnerName"); ti=r.findtext(".//officerTitle") or ("Director" if r.findtext(".//isDirector") in("1","true") else "Other"); o=[]
    for tx in r.findall(".//nonDerivativeTransaction"):
        if tx.findtext(".//transactionCode")!="P": continue
        sh=float(tx.findtext(".//transactionShares/value") or 0); px=float(tx.findtext(".//transactionPricePerShare/value") or 0)
        o.append(dict(insider=n,title=ti,date=tx.findtext(".//transactionDate/value")[:10],shares=sh,price=px,value=sh*px))
    return o
def sec_buys(t,days=30):
    c=cik(t)
    if not c: return pd.DataFrame()
    s=requests.get(f"https://data.sec.gov/submissions/CIK{c}.json",headers=UA,timeout=30).json()["filings"]["recent"]; cut=str(date.today()-timedelta(days)); rows=[]
    for i,f in enumerate(s["form"]):
        if f=="4" and s["filingDate"][i]>=cut:
            a=s["accessionNumber"][i].replace("-",""); time.sleep(.15)
            try: rows+=parse4(requests.get(f"https://www.sec.gov/Archives/edgar/data/{int(c)}/{a}/{s['primaryDocument'][i].split('/')[-1]}",headers=UA,timeout=30).content)
            except Exception: pass
    return pd.DataFrame(rows)
def nse_buys(t,days=30):
    """NSE SEBI-PIT disclosures. NSE blocks bots at times / changes schema; fails soft."""
    sym=norm(t).split(".")[0]; s=requests.Session(); h={"User-Agent":"Mozilla/5.0","Accept":"application/json","Referer":"https://www.nseindia.com/"}
    s.get("https://www.nseindia.com",headers=h,timeout=20)
    f=(date.today()-timedelta(days)).strftime("%d-%m-%Y"); to=date.today().strftime("%d-%m-%Y")
    j=s.get("https://www.nseindia.com/api/corporates-pit",params=dict(index="equities",symbol=sym,from_date=f,to_date=to),headers=h,timeout=20).json().get("data",[])
    rows=[dict(insider=r.get("acqName"),title=r.get("personCategory"),date=str(r.get("acqfromDt"))[:11],shares=float(r.get("secAcq") or 0),value=float(r.get("secVal") or 0))
          for r in j if "buy" in str(r.get("tdpTransactionType","")).lower() or "purchase" in str(r.get("acqMode","")).lower()]
    return pd.DataFrame(rows)
def max_window(df,w=14,min_value=0):
    if df is None or df.empty: return 0,0.0
    d=df.copy(); d["dt"]=pd.to_datetime(d.date,errors="coerce"); d=d.dropna(subset=["dt","insider"]); d=d[d.value>=min_value].sort_values("dt"); best=(0,0.0)
    for dt in d.dt:
        win=d[(d.dt<=dt)&(d.dt>dt-pd.Timedelta(days=w))]; n=win.insider.nunique()
        if n>best[0]: best=(n,float(win.value.sum()))
    return best
def cluster(t,days=30,min_insiders=3,window=14):
    try: df=sec_buys(t,days) if is_us(t) else nse_buys(t,days)
    except Exception as e: return dict(ticker=t,flag=False,n=0,total=0.0,err=str(e)[:120],df=pd.DataFrame())
    if len(df): df=df.drop_duplicates([c for c in("insider","date","shares","price") if c in df])
    n,tot=max_window(df,window,5000 if is_us(t) else 100000)
    return dict(ticker=t,flag=n>=min_insiders,n=n,total=tot,df=df)
