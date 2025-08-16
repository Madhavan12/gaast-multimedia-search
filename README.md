# GAAST Multimedia Search

Local, zero-cost prototype to **search inside audio/video** and jump to exact timestamps.

## Features
- Offline transcription with Whisper (no cloud fees)
- Keyword + **semantic** search (Sentence-Transformers)
- Tiny Flask UI to play media from the matching timestamp

## Tech
Python, Flask, Whisper, sentence-transformers, NumPy, orjson

## Quickstart
```bash
python -m venv .venv
# Windows PowerShell
. .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
pip install sentence-transformers torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Add a test file to data/media/
python scripts/transcribe.py --model base --language en
python scripts/build_index.py
python scripts/build_embeddings.py

python app.py
# open http://127.0.0.1:5000
