import os
import re
import requests
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
URLS_PATH = BASE_DIR / "data" / "urls.txt"
OUT_DIR = BASE_DIR / "data" / "pdfs"

def safe_filename(url: str) -> str:
    name = url.strip().split("/")[-1]
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name or "file.pdf"

def download(url: str, out_path: str, timeout=60) -> None:
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(out_path)) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 128):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(URLS_PATH, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    for url in urls:
        filename = safe_filename(url)
        out_path = os.path.join(OUT_DIR, filename)
        if os.path.exists(out_path):
            print(f"Skip (exists): {out_path}")
            continue
        print(f"Downloading: {url}")
        download(url, out_path)

if __name__ == "__main__":
    main()
