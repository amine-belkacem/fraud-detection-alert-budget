"""Download the credit card fraud dataset and check that it is the expected file.

Usage:
    python scripts/download_data.py --out data/raw

The CSV (about 150 MB) comes from the copy that Google hosts for the TensorFlow
imbalanced-data tutorial. Its content matches the Kaggle release
(mlg-ulb/creditcardfraud): same 284,807 rows, 492 frauds and 1,081 duplicate
rows. I have not compared the two files byte for byte, because Kaggle needs an
account to download. I first tried OpenML (dataset 1597) but its API was
returning HTTP 503 at the time, so the script does not depend on it.

The file is verified twice: SHA-256 of the bytes, then row and fraud counts
after parsing. The SHA-256 pinned below is the one of the Google copy. If you
put a copy downloaded from Kaggle at data/raw/creditcard.csv, the script skips
the download; should its bytes differ, the SHA-256 check will say so even if
the counts match.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraudbudget.data import EXPECTED_FRAUDS, EXPECTED_ROWS, load_dataset  # noqa: E402

URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
FILENAME = "creditcard.csv"
CHUNK_BYTES = 1 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(CHUNK_BYTES):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r{done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    # Rename only once the transfer is complete, so an interrupted run never
    # leaves a truncated file under the final name.
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / FILENAME

    if dest.exists():
        print(f"{dest} already present, skipping download")
    else:
        print(f"downloading {URL}")
        download(URL, dest)

    digest = sha256_of(dest)
    if digest != SHA256:
        print(f"ECHEC SHA-256 : obtenu {digest}, attendu {SHA256}", file=sys.stderr)
        return 1

    df = load_dataset(dest)  # raises with a clear message if counts or schema are off
    print(f"OK  sha256={digest[:16]}...  rows={len(df)} (expected {EXPECTED_ROWS})  "
          f"frauds={int(df['Class'].sum())} (expected {EXPECTED_FRAUDS})  "
          f"prevalence={df['Class'].mean():.4%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
