"""Download the Kokoro text-to-speech model and voices (Apache-2.0) from the project's
GitHub release into a cache folder, and verify them."""

import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
FILES = {
    "kokoro-v1.0.onnx": "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    "voices-v1.0.bin": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def fetch(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        path = dest / name
        if not path.exists() or sha256(path) != digest:
            print(f"downloading {name} ...", file=sys.stderr)
            urllib.request.urlretrieve(BASE + name, path)
        if sha256(path) != digest:
            raise SystemExit(f"error: {path} does not match the expected checksum")
    return dest


if __name__ == "__main__":
    print(fetch(Path(sys.argv[1] if len(sys.argv) > 1 else "data/cache/media/models")))
