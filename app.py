from flask import Flask, render_template_string, request, send_from_directory
import orjson, html
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# ---------- Config ----------
CFG_PATH = Path("config.json")
CFG = {
    "TOP_K": 5,
    "WIN_SECONDS": 30.0,
    "MIN_TOTAL": 0.25,
    "MIN_SEM": 0.40,
    "KW_WEIGHT": 0.45,
    "SEM_WEIGHT": 0.55,
    "CONTEXT_BEFORE": 1,
    "CONTEXT_AFTER": 1,
    "EARLY_TIME_BONUS_SEC": 60.0,
    "EARLY_BONUS": 0.05,
}
if CFG_PATH.exists():
    CFG.update(json.loads(CFG_PATH.read_text(encoding="utf-8")))

# ---------- Paths ----------
INDEX_PATH = Path("index/search_index.json")
EMB_PATH = Path("index/embeddings.npz")
MEDIA_BASE_URL = "/media/"  # served by Flask static route below
CAPTIONS_DIR = Path("data/captions")

# ---------- Flask ----------
app = Flask(__name__, static_folder="data/media", static_url_path="/media")

# Serve captions (WebVTT) if present
@app.route("/captions/<path:filename>")
def captions(filename):
    return send_from_directory(CAPTIONS_DIR, filename)

# ---------- HTML ----------
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
  .row{display:flex;gap:1rem;align-items:center;flex-wrap:wrap}
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
      <div class="row" style="margin-top:.5rem">
        <a href="{{r.play_url}}" target="_blank">▶ Play from {{r.start}}s</a>
        <a href="{{r.deep_link}}" target="_blank">Open player</a>
        <button onclick="navigator.clipboard.writeText(window.location.origin + '{{r.deep_link}}'); this.innerText='Copied!'; setTimeout(()=>this.innerText='Copy link',1200);">Copy link</button>
        {% if r.captions %}
          <a href="{{r.captions}}" target="_blank">Captions (VTT)</a>
        {% endif %}
      </div>
    </div>
  {% endfor %}
{% endif %}
"""

PLAYER_TMPL = """
<!doctype html>
<title>Player</title>
<style>
  body{font-family:system-ui,Arial,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}
  video,audio{width:100%;max-width:900px}
</style>
<h2>{{fname}}</h2>
{% if is_video %}
<video controls autoplay src="{{src}}#t={{t}}">
  <track kind="subtitles" src="{{vtt}}" default>
</video>
{% else %}
<audio controls autoplay src="{{src}}#t={{t}}"></audio>
{% endif %}
<p><a href="/">← Back to search</a></p>
"""

# ---------- Helpers ----------
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

def is_video_file(path: str) -> bool:
    return path.lower().endswith((".mp4", ".mov", ".mkv", ".webm"))

def load_resources():
    data = {"docs": []}
    if INDEX_PATH.exists():
        data = orjson.loads(INDEX_PATH.read_bytes())
    embs = None
    if EMB_PATH.exists():
        npz = np.load(EMB_PATH)
        embs = npz["embeddings"]
    model = None
    if embs is not None:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return data, embs, model

DATA, EMBEDDINGS, SEM_MODEL = load_resources()

def semantic_scores(query: str, embs):
    if not query or embs is None or SEM_MODEL is None:
        return None
    q = SEM_MODEL.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = (embs @ q)  # cosine in [-1,1]
    return ((sims + 1.0) * 0.5)  # map to [0,1]

def early_bonus(start_sec: float) -> float:
    return CFG["EARLY_BONUS"] if start_sec <= CFG["EARLY_TIME_BONUS_SEC"] else 0.0

def merge_context_indices(docs, hit_idx, before, after):
    """
    Return indices around hit within the same file, ordered by start time.
    """
    base = docs[hit_idx]
    fname = base["media_relpath"]
    s = max(0, hit_idx - before)
    e = min(len(docs) - 1, hit_idx + after)
    chosen = [i for i in range(s, e + 1) if docs[i]["media_relpath"] == fname]
    chosen.sort(key=lambda i: float(docs[i]["start"]))
    return chosen

# ---------- Routes ----------
@app.route("/")
def home():
    q = (request.args.get("q") or "").strip()
    docs = DATA.get("docs", [])
    results = None
    if q:
        qnorm = q.lower()

        # Keyword scoring (0..1)
        kw_scores = []
        for d in docs:
            tn = d["text_norm"]
            pos = tn.find(qnorm)
            kw = max(0.6, 1.0 / (1.0 + pos)) if pos != -1 else 0.0
            kw_scores.append(kw)

        # Semantic scoring (0..1)
        sem_arr = semantic_scores(q, EMBEDDINGS)
        if sem_arr is None:
            sem_scores = [0.0] * len(docs)
        else:
            sem_scores = sem_arr.tolist()

        # Combine, threshold, and apply early bonus
        combined = []
        for i, d in enumerate(docs):
            kw = float(kw_scores[i])
            sem = float(sem_scores[i])
            if kw == 0.0 and sem < CFG["MIN_SEM"]:
                continue
            total = CFG["SEM_WEIGHT"] * sem + CFG["KW_WEIGHT"] * kw + early_bonus(float(d["start"]))
            if total < CFG["MIN_TOTAL"] and kw == 0.0:
                continue
            combined.append((total, i, d))

        combined.sort(key=lambda x: x[0], reverse=True)

        # Window suppression per file
        kept = []
        last_by_file = {}  # file -> list[(start, score)]
        for total, i, d in combined:
            fname = d["media_relpath"]
            start = float(d["start"])
            ok = True
            if fname in last_by_file:
                for s_prev, sc_prev in list(last_by_file[fname]):
                    if abs(start - s_prev) <= CFG["WIN_SECONDS"]:
                        if total <= sc_prev:
                            ok = False
                        else:
                            last_by_file[fname].remove((s_prev, sc_prev))
                        break
            if ok:
                last_by_file.setdefault(fname, []).append((start, total))
                kept.append((total, i, d))
            if len(kept) >= CFG["TOP_K"]:
                break

        # Build UI results with context merge
        results = []
        for total, i, d in kept:
            idxs = merge_context_indices(docs, i, CFG["CONTEXT_BEFORE"], CFG["CONTEXT_AFTER"])
            media_rel = d["media_relpath"].replace("\\", "/")
            start = round(float(docs[idxs[0]]["start"]), 2)
            end = round(float(docs[idxs[-1]]["end"]), 2)
            play_url = f"{MEDIA_BASE_URL}{media_rel}#t={start}"
            # captions path (optional)
            vtt_name = Path(d["media_file"]).stem + ".vtt"
            vtt_path = CAPTIONS_DIR / vtt_name
            results.append({
                "media_file": d["media_file"],
                "start": start,
                "end": end,
                "snippet": highlight(" … ".join([docs[j]["text"] for j in idxs]), q),
                "play_url": play_url,
                "deep_link": f"/play?f={media_rel}&t={start}",
                "captions": f"/captions/{vtt_name}" if vtt_path.exists() else None,
            })

    return render_template_string(HTML_TMPL, q=q, results=results)

@app.route("/play")
def play():
    fname = request.args.get("f")
    t = request.args.get("t", "0")
    if not fname:
        return "Missing f= (file path)", 400
    src = f"{MEDIA_BASE_URL}{fname}"
    vtt = f"/captions/{Path(fname).stem}.vtt"
    return render_template_string(
        PLAYER_TMPL,
        fname=fname,
        t=t,
        src=src,
        is_video=is_video_file(fname),
        vtt=vtt,
    )

# ---------- Main ----------
if __name__ == "__main__":
    app.run(debug=True)
