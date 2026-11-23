import time, re, db
RULE=("\nUse ONLY the EVIDENCE for facts/numbers. If it's not there, say 'not in data'. Cite source tags like [SEC 10-K] or [YF fundamentals]. "
      "Max 170 words. Never promise returns.")
ROLES={"BULL":"You are the BULL analyst arguing why this stock is a buy. Rebut the BEAR directly.",
 "BEAR":"You are the BEAR analyst arguing why this stock is a sell/avoid. Rebut the BULL directly.",
 "JUDGE":"You are the neutral JUDGE. Start with 'WINNER: BULL', 'WINNER: BEAR' or 'WINNER: DRAW', then 'CONFIDENCE: x%', then a short reasoning scoring evidence quality, and flag unsupported claims."}
class Debate:
    def __init__(s,sid,ticker,evidence,pool):
        if not evidence or len(evidence)<150: raise ValueError('Not enough data for this ticker to debate')
        s.sid,s.ticker,s.ev,s.pool=sid,ticker,evidence,pool
    def hist(s):
        d=db.load("messages"); return d[d.sid==s.sid].sort_values("ts") if len(d) else d
    def _say(s,role,instr):
        h=s.hist().tail(24); tr="\n".join(f"{r.role}: {r.content}" for r in h.itertuples()) if len(h) else "(start)"
        m=[{"role":"system","content":ROLES[role]+RULE+f"\nTICKER: {s.ticker}\nEVIDENCE:\n{s.ev[:14000]}"},
           {"role":"user","content":f"DEBATE SO FAR:\n{tr}\n\n{instr}"}]
        o=s.pool.chat(m); db.save("messages",dict(sid=s.sid,ts=time.time(),role=role,content=o)); return o
    def round(s): s._say("BULL","Make or continue your case."); s._say("BEAR","Rebut the bull and press your case.")
    def user(s,text):
        db.save("messages",dict(sid=s.sid,ts=time.time(),role="USER",content=text))
        s._say("BULL",f"The USER said: {text}\nRespond to them."); s._say("BEAR",f"The USER said: {text}\nRespond to them and to the bull.")
    def verdict(s):
        o=s._say("JUDGE","Judge the debate."); m=re.search(r"WINNER:\W*(BULL|BEAR|DRAW)",o.upper()); w=m.group(1) if m else "DRAW"
        db.save("verdicts",dict(ticker=s.ticker,ts=time.time(),winner=w)); return o
def last_winner(t,max_age_days=7):
    d=db.load("verdicts")
    if not len(d): return None
    d=d[(d.ticker==t)&(d.ts>time.time()-max_age_days*86400)]
    return d.sort_values("ts").winner.iloc[-1] if len(d) else None
