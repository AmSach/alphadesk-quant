import os, time, threading, requests
class KeyPool:
    """Round-robins keys; cools down rate-limited keys, drops bad ones."""
    def __init__(s,keys,url=None,model=None):
        s.keys=list(dict.fromkeys(k.strip() for k in keys if k.strip())); s.i=0; s.cool={}; s.lock=threading.Lock()
        s.url=url or os.getenv("LLM_URL","https://api.groq.com/openai/v1/chat/completions")
        s.model=model or os.getenv("LLM_MODEL","llama-3.3-70b-versatile")
    def chat(s,messages,temperature=0.3,max_tokens=700):
        if not s.keys: raise RuntimeError("No API keys set")
        last=None
        for _ in range(len(s.keys)*3+3):
            with s.lock:
                live=[k for k in s.keys if s.cool.get(k,0)<time.time()]
                if not live: wait=min(s.cool.values())-time.time(); live=None
                else: s.i+=1; k=live[s.i%len(live)]
            if live is None: time.sleep(max(wait,1)); continue
            try:
                r=requests.post(s.url,headers={"Authorization":f"Bearer {k}"},timeout=90,
                    json=dict(model=s.model,messages=messages,temperature=temperature,max_tokens=max_tokens))
                if r.status_code==429: s.cool[k]=time.time()+float(r.headers.get("retry-after",20)); continue
                if r.status_code in(401,403): s.cool[k]=time.time()+86400; continue
                r.raise_for_status(); o=r.json()["choices"][0]["message"]["content"]
                if o and o.strip(): return o
                s.cool[k]=time.time()+3; continue
            except (requests.RequestException,KeyError,IndexError,ValueError,TypeError) as e: last=e; s.cool[k]=time.time()+5
        raise RuntimeError(f"All keys failed: {last}")
