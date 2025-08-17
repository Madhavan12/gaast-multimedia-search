import json
from pathlib import Path
from urllib.parse import urlencode
from app import DATA, EMBEDDINGS, semantic_scores, CFG, merge_context_indices, early_bonus, highlight, MEDIA_BASE_URL

def search_once(query: str):
    docs = DATA.get("docs", [])
    qnorm = query.lower()
    # keyword
    kw_scores = []
    for d in docs:
        tn = d["text_norm"]; pos = tn.find(qnorm)
        kw_scores.append(max(0.6, 1.0/(1.0+pos)) if pos!=-1 else 0.0)
    # semantic
    import numpy as np
    sem = semantic_scores(query, EMBEDDINGS)
    if sem is None: sem = np.zeros(len(docs))
    # combine
    scored=[]
    for i,d in enumerate(docs):
        kw=float(kw_scores[i]); s=float(sem[i])
        if kw==0.0 and s < CFG["MIN_SEM"]: continue
        total=CFG["SEM_WEIGHT"]*s + CFG["KW_WEIGHT"]*kw + early_bonus(float(d["start"]))
        if total<CFG["MIN_TOTAL"] and kw==0.0: continue
        scored.append((total,i,d))
    scored.sort(key=lambda x:x[0], reverse=True)
    # take top K
    return scored[:CFG["TOP_K"]]

def main():
    qfile = Path("eval/queries.json")
    if not qfile.exists():
        print("eval/queries.json not found"); return
    tests = json.loads(qfile.read_text(encoding="utf-8"))
    total=0; hits=0
    for t in tests:
        q=t["q"]; expect=t.get("expect",[])
        res=search_once(q)
        found=False
        for _,i,d in res:
            key=f'{d["media_file"]}#{int(d["start"])}-{int(d["end"])}'
            if any(key.startswith(e.split("#")[0]) for e in expect):
                found=True; break
        total+=1; hits+=1 if found else 0
        print(f'Q: {q:20s} -> {"OK" if found else "MISS"}  ({len(res)} returned)')
    print(f"Precision@{CFG['TOP_K']}: {hits}/{total} = {hits/total:.2f}")

if __name__=="__main__":
    main()
