import numpy as np, yfinance as yf, pandas as pd, db
from data import norm
_fb=None
def analyze(path,ticker,call_date):
    """call_date: 'YYYY-MM-DD' of the call. Returns stress score 0-100 and next-day return."""
    global _fb
    import librosa
    from faster_whisper import WhisperModel
    from transformers import pipeline
    txt=" ".join(s.text for s in WhisperModel("base",compute_type="int8").transcribe(path)[0])
    y,sr=librosa.load(path,sr=16000); f0=librosa.pyin(y,fmin=70,fmax=400,sr=sr)[0]; f0=f0[~np.isnan(f0)]
    pitch_cv=float(np.std(f0)/np.mean(f0)) if len(f0)>10 else 0
    voiced=sum(b-a for a,b in librosa.effects.split(y,top_db=30))/len(y); pause=1-voiced; rate=len(txt.split())/(len(y)/sr)
    _fb=_fb or pipeline("text-classification",model="ProsusAI/finbert",top_k=None)
    ch=[txt[i:i+1500] for i in range(0,len(txt),1500)]; neg=float(np.mean([next(x["score"] for x in r if x["label"]=="negative") for r in _fb(ch,truncation=True)]))
    score=round(100*min(1,.45*neg*2+.3*min(pitch_cv/.35,1)+.25*min(pause/.4,1)),1)
    h=yf.Ticker(norm(ticker)).history(start=call_date,period="10d"); nxt=float(h.Close.iloc[1]/h.Close.iloc[0]-1) if len(h)>1 else None
    r=dict(ticker=ticker,call_date=call_date,stress=score,neg=neg,pitch_cv=pitch_cv,pause=pause,wpm=rate*60,next_day_ret=nxt); db.save("calls",r); return r
def predictive_power():
    d=db.load("calls")
    d=d.dropna(subset=["next_day_ret"]) if "next_day_ret" in d else d.iloc[0:0]
    return dict(n=len(d),corr=float(d.stress.corr(d.next_day_ret))) if len(d)>2 else dict(n=len(d),corr=None)
