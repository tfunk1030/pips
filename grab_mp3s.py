import re
import sys
from pathlib import Path
from urllib.parse import urlparse
import urllib.request

MP3_RE = re.compile(r'https?://[^\s"^]+?\.mp3')

def filename_from_url(url: str) -> str:
    p = urlparse(url)
    name = Path(p.path).name
    return name or "audio.mp3"

def download(url: str, out_dir: Path, cookie: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename_from_url(url)

    req = urllib.request.Request(
        url,
        headers={
            # mimic a browser enough for most static asset hosts
            "User-Agent": "Mozilla/5.0",
            **({"Cookie": cookie} if cookie else {}),
        },
        method="GET",
    )

    with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
        f.write(resp.read())

    return out_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python grab_mp3s.py <curl_dump.txt> [output_dir]")
        sys.exit(1)

    dump_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("mp3s")

    text = dump_path.read_text(encoding="utf-8", errors="ignore")
    urls = sorted(set(MP3_RE.findall(text)))

    if not urls:
        print("No .mp3 URLs found.")
        return

    # Optional cookie if the asset is gated (avoid hardcoding secrets)
    cookie = None
    cookie_path = Path("cookie.txt")
    if cookie_path.exists():
        cookie = cookie_path.read_text(encoding="utf-8").strip() or None

    print(f"Found {len(urls)} mp3 URL(s). Downloading to: {out_dir.resolve()}")
    for i, url in enumerate(urls, 1):
        try:
            out = download(url, out_dir, cookie=cookie)
            print(f"[{i}/{len(urls)}] OK  {out.name}")
        except Exception as e:
            print(f"[{i}/{len(urls)}] FAIL {url}  ({e})")

if __name__ == "__main__":
    main()
