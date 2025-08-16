from flask import Flask, render_template_string, request
import orjson, html
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

TOP_K = 4           # max results to show
WIN_SECONDS = 20.0   # suppress near-duplicates within this window per file
MIN_TOTAL = 0.18     # overall score floor
MIN_SEM = 0.30       # semantic floor when there’s no keyword hit
KW_WEIGHT = 0.35     # weight of keyword score
SEM_WEIGHT = 0.65    # weight of semantic score



INDEX_PATH = Path("index/search_index.json")
MEDIA_BASE_URL = "/media/"  # served by Flask static route below

app = Flask(__name__, static_folder="data/media", static_url_path="/media")

def load_resources():
    data = {"docs": []}
    if INDEX_PATH.exists():
        data = orjson.loads(INDEX_PATH.read_bytes())

    emb_path = Path("index/embeddings.npz")
    embs = None
    if emb_path.exists():
        npz = np.load(emb_path)
        embs = npz["embeddings"]

    # lazy-load model if embeddings exist (for query encoding)
    model = None
    if embs is not None:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return data, embs, model

DATA, EMBEDDINGS, SEM_MODEL = load_resources()

HTML_TMPL = """
<!doctype html>
<title>GAAST Multimedia Search (MVP)</title>
<style>
  body{font-family:system-ui,Arial,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}
  form{display:flex;gap:.5rem;margin-bottom:1rem}
  input[type=text]{flex:1;padding:.6rem;border:1px solid #ccc;border-radius:.5rem}
  button{padding:.6rem 1rem;border:0;background:#111;color:#fff;border-radius:.5rem;cursor:pointer}
  .item{padding:1rem;border:1px solid #eee;border-radius:.75rem;margin:.5rem 0}
  .ts{font-size:.9rem;color:#666}
  .hit{background:#fffbcc}
</style>
<h1>GAAST Multimedia Search — MVP</h1>
<form method="GET">
  <input type="text" name="q" placeholder="Try: emergency landing, pre-flight checks" value="{{q}}"/>
  <button>Search</button>
</form>
<p>Media folder: <code>data/media</code>. Results link to local files with start offsets.</p>
{% if results is not none %}
  <p><strong>{{results|length}}</strong> matches</p>
  {% for r in results %}
    <div class="item">
      <div><strong>{{r.media_file}}</strong></div>
      <div class="ts">{{r.start}}s → {{r.end}}s</div>
      <div>{{r.snippet|safe}}</div>
      <div style="margin-top:.5rem">
        <a href="{{r.play_url}}" target="_blank">▶ Play from {{r.start}}s</a>
      </div>
    </div>
  {% endfor %}
{% endif %}
"""

def highlight(text: str, q: str) -> str:
    safe = html.escape(text)
    if not q:
        return safe
    try:
        import re
        pattern = re.compile(re.escape(q), re.IGNORECASE)
        return pattern.sub(lambda m: f"<mark class='hit'>{m.group(0)}</mark>", safe)
    except Exception:
        return safe
    
def semantic_scores(query: str, docs, embs):
    if not query or embs is None or SEM_MODEL is None:
        return None  # semantic disabled if no embeddings
    q = SEM_MODEL.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = (embs @ q)  # cosine because both normalized
    # return list of (score, index_in_docs)
    return sims

@app.route("/")
def home():
    q = (request.args.get("q") or "").strip()
    docs = DATA.get("docs", [])
    results = None
    if q:
        qnorm = q.lower()

        # --- keyword scores (0..1) ---
        kw_scores = []
        for d in docs:
            tn = d["text_norm"]
            pos = tn.find(qnorm)
            if pos != -1:
                # exact keyword hit → stronger score if earlier in segment text
                kw = max(0.6, 1.0 / (1.0 + pos))  # clamp min to boost matches
            else:
                kw = 0.0
            kw_scores.append(kw)

        # --- semantic scores (normalize to 0..1) ---
        sem_sims = semantic_scores(q, docs, EMBEDDINGS)
        sem_scores = []
        if sem_sims is not None:
            # sem_sims is cosine in [-1,1] → map to [0,1]
            sem_scores = ((sem_sims + 1.0) * 0.5).tolist()
        else:
            sem_scores = [0.0] * len(docs)

        # --- combine & filter with thresholds ---
        combined = []
        for i, d in enumerate(docs):
            kw = kw_scores[i]
            sem = sem_scores[i]
            # if there's no keyword match, require a higher semantic minimum
            if kw == 0.0 and sem < MIN_SEM:
                continue
            total = SEM_WEIGHT * sem + KW_WEIGHT * kw
            if total < MIN_TOTAL and kw == 0.0:
                continue
            combined.append((total, i, d))

        # sort best-first
        combined.sort(key=lambda x: x[0], reverse=True)

        # --- windowed suppression: keep best hit per 20s window per file ---
        kept = []
        last_kept_by_file = {}  # file -> list of (start_time, score)
        for total, i, d in combined:
            fname = d["media_relpath"]
            start = float(d["start"])
            ok = True
            if fname in last_kept_by_file:
                for s_prev, sc_prev in last_kept_by_file[fname]:
                    if abs(start - s_prev) <= WIN_SECONDS:
                        # already have a hit in this window; keep the stronger one
                        if total <= sc_prev:
                            ok = False
                        else:
                            # replace the weaker one
                            last_kept_by_file[fname].remove((s_prev, sc_prev))
                        break
            if ok:
                last_kept_by_file.setdefault(fname, []).append((start, total))
                kept.append((total, i, d))
            if len(kept) >= TOP_K:
                break

        # build UI results
        results = []
        for total, i, d in kept:
            media_rel = d["media_relpath"].replace("\\", "/")
            start = round(float(d["start"]), 2)
            play_url = f"{MEDIA_BASE_URL}{media_rel}#t={start}"
            results.append({
                "media_file": d["media_file"],
                "start": start,
                "end": round(float(d["end"]), 2),
                "snippet": highlight(d["text"], q),
                "play_url": play_url,
            })

    return render_template_string(HTML_TMPL, q=q, results=results)

if __name__ == "__main__":
    app.run(debug=True)
