import json
from pathlib import Path

TRANS_DIR = Path("data/transcripts")
OUT_DIR = Path("data/captions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def to_timestamp(s: float) -> str:
    # WebVTT uses HH:MM:SS.mmm (with dot or comma; dot is fine)
    h = int(s // 3600); s -= h*3600
    m = int(s // 60); s -= m*60
    ms = int(round((s - int(s)) * 1000))
    s = int(s)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def export_one(p: Path):
    data = json.loads(p.read_text(encoding="utf-8"))
    name = Path(data["media_file"]).stem
    out = ["WEBVTT", ""]
    for seg in data.get("segments", []):
        start = to_timestamp(float(seg["start"]))
        end = to_timestamp(float(seg["end"]))
        text = seg["text"].strip()
        out.append(f"{start} --> {end}")
        out.append(text)
        out.append("")  # blank line
    out_path = OUT_DIR / f"{name}.vtt"
    out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"Saved {out_path}")

def main():
    for p in TRANS_DIR.glob("*.json"):
        export_one(p)

if __name__ == "__main__":
    main()
