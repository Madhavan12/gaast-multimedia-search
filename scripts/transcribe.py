import argparse, json, os, sys
from pathlib import Path
from rich import print
from tqdm import tqdm
import whisper

MEDIA_DIR = Path("data/media")
OUT_DIR = Path("data/transcripts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTS = {'.mp3', '.wav', '.m4a', '.aac', '.flac', '.mp4', '.mov', '.mkv'}

def list_media():
    return [p for p in MEDIA_DIR.glob("**/*") if p.suffix.lower() in SUPPORTED_EXTS]

def transcribe_file(model, path: Path, language: str | None):
    print(f"[bold green]Transcribing:[/bold green] {path}")
    result = model.transcribe(str(path), language=language)
    out = {
        "media_file": str(path.name),
        "media_relpath": str(path.relative_to(MEDIA_DIR)),
        "language": result.get("language"),
        "duration": result.get("duration"),
        "text": result.get("text"),
        "segments": [
            {
                "id": s.get("id"),
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": s.get("text", "").strip()
            }
            for s in result.get("segments", [])
        ],
    }
    out_path = OUT_DIR / f"{path.stem}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[cyan]Saved →[/cyan] {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", help="Whisper model: tiny|base|small|medium|large")
    ap.add_argument("--language", default=None, help="Hint language code (e.g., en)")
    args = ap.parse_args()

    if not MEDIA_DIR.exists():
        print(f"[red]Media folder not found:[/red] {MEDIA_DIR}")
        sys.exit(1)

    print("[bold]Loading Whisper model…[/bold]")
    model = whisper.load_model(args.model)

    media = list_media()
    if not media:
        print(f"[yellow]No media files found in {MEDIA_DIR}. Add .mp3/.mp4 etc.[/yellow]")
        return

    for p in tqdm(media):
        transcribe_file(model, p, args.language)

if __name__ == "__main__":
    main()
