import json, re
from pathlib import Path
from rich import print

TRANS_DIR = Path("data/transcripts")
INDEX_DIR = Path("index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = INDEX_DIR / "search_index.json"

def normalize(txt: str) -> str:
    return re.sub(r"\s+", " ", txt.lower()).strip()

def main():
    docs = []
    for p in TRANS_DIR.glob("*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        media_file = data["media_file"]
        relpath = data["media_relpath"]
        for seg in data.get("segments", []):
            docs.append({
                "media_file": media_file,
                "media_relpath": relpath,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "text_norm": normalize(seg["text"])
            })
    INDEX_PATH.write_text(json.dumps({"docs": docs}, ensure_ascii=False), encoding="utf-8")
    print(f"[green]Built index →[/green] {INDEX_PATH}  (segments: {len(docs)})")

if __name__ == "__main__":
    main()
