# 📷 Photo-Indexer

**Photo-Indexer** is an offline pipeline that walks through a folder of DSLR RAW photos (`*.NEF`), creates lightweight thumbnails, runs three vision models, and stores rich metadata in an embedded database.
Designed for a **CPU-only MacBook with 32 GB RAM (Metal/MPS available)**, but scales up without code changes.

| Stage                        | Model / Tool                   | Output                                   |
| ---------------------------- | ------------------------------ | ---------------------------------------- |
| Thumbnail + EXIF             | `rawpy`, `piexif`              | 512 px RGB JPEG, `DateTimeOriginal`, GPS |
| Scene label & indoor/outdoor | ResNet-18 Places365            | e.g. `outdoor`, `mountain`               |
| Person detection & count     | YOLOv8-n                       | `people =true`, `count = 4`              |
| Caption & fallback location  | LLaVA-7B-Q4 (served by Ollama) | One-sentence caption                     |
| Fusion & storage             | Python                         | One row per photo in SQLite + FTS5       |

---

## ✨ Features

* **100 % open source** – only PyPI wheels and model checkpoints.
* **Offline** – pull models once, index with zero network.
* **Thread-pooled** – keeps all CPU cores busy; overlaps disk I/O and REST calls.
* **Pluggable** – swap SQLite for DuckDB or LLaVA for any GGUF vision-LLM via config.
* **Thumbnail cache** – first pass builds `.jpg` thumbs; re-runs skip RAW decoding.

---

## ⏩ Quick start

Tested on **macOS 12 / 14, Python 3.11, Ollama 0.1.34**.

```bash
# 1 — clone the repo
git clone https://github.com/your-handle/photo-indexer.git
cd photo-indexer

# 2 — system dependencies (Homebrew)
/bin/bash scripts/prepare_env.sh    # optional helper
# or follow docs/setup_mac.md

# 3 — Python environment
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 4 — fetch model weights
bash scripts/download_weights.sh    # ResNet-Places files, sample NEFs
ollama pull llava:7b-q4_k_m         # caption head
# YOLOv8-n downloads automatically on first run

# 5 — run!
pi index /Volumes/DSLR_backup --workers 8
```

First run compiles Torch-Metal kernels; expect \~2 min start-up lag.

---

## 🏗️ Project layout

```
photo-indexer/
├─ scripts/               helper shell scripts
├─ data/                  thumbnails & databases (git-ignored)
├─ src/photo_indexer/     main package
│  ├─ cli.py              entry-point (`pi`)
│  ├─ pipelines/          preprocess, vision, fusion, db
│  ├─ models/             model adapters (ResNet, YOLO, LLaVA)
│  ├─ utils/              exif, geo, logging, etc.
│  └─ workers.py          thread orchestration
└─ tests/                 pytest suite
```

A deeper diagram lives in `docs/ARCHITECTURE.md`.

---

## 🔧 CLI overview

```
Usage: pi index [OPTIONS] PHOTO_ROOT

  Index all .NEF files under PHOTO_ROOT.

Options:
  --workers INTEGER      Concurrent threads (default: CPU count)
  --db {sqlite,duckdb}   Storage backend (default: sqlite)
  --thumb-size INTEGER   Long-edge pixels for JPEG cache (default: 512)
  --help                 Show this message and exit
```

---

## 🚀 Performance tips

* Use `--workers 6-8` on a 10-core Apple Silicon; higher counts seldom help.
* Keep the laptop on mains – Metal can fall back to CPU on low battery.
* To re-caption with a new LLM, reuse cached thumbnails:

  ```bash
  pi index ~/photos --skip-preprocess --caption-only
  ```

---

## 📄 License

Code is MIT-licensed (see `LICENSE`).
Model checkpoints retain their original licenses (MIT for Places365, GPL-v3 for YOLOv8 weights, LLaMA Community License for LLaVA).

Happy indexing!
