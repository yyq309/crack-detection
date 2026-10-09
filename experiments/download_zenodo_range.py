"""Resume a large Zenodo file using HTTP byte ranges."""

import concurrent.futures
import hashlib
import sys
from pathlib import Path

import requests


def main():
    record, name, size_text, md5_expected, output = sys.argv[1:]
    size = int(size_text)
    url = f"https://zenodo.org/api/records/{record}/files/{name}/content"
    target = Path(output)
    parts = target.parent / (target.name + ".parts")
    parts.mkdir(exist_ok=True)
    chunk = 16 * 1024 * 1024

    def fetch(index):
        lo, hi = index * chunk, min(size, (index + 1) * chunk) - 1
        path = parts / f"{index:04d}"
        expected = hi - lo + 1
        if path.exists() and path.stat().st_size == expected:
            return index, expected
        with requests.get(url, headers={"Range": f"bytes={lo}-{hi}"}, stream=True, timeout=60) as response:
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError(f"Range {index}: HTTP {response.status_code}")
            with path.open("wb") as stream:
                for data in response.iter_content(1024 * 1024):
                    stream.write(data)
        if path.stat().st_size != expected:
            raise RuntimeError(f"Range {index}: wrong length {path.stat().st_size} != {expected}")
        return index, expected

    count = (size + chunk - 1) // chunk
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for index, bytes_done in pool.map(fetch, range(count)):
            print(f"chunk {index + 1}/{count}: {bytes_done}", flush=True)
    md5 = hashlib.md5()
    with target.open("wb") as out:
        for index in range(count):
            with (parts / f"{index:04d}").open("rb") as stream:
                while data := stream.read(1024 * 1024):
                    out.write(data)
                    md5.update(data)
    if md5.hexdigest() != md5_expected:
        raise RuntimeError(f"MD5 mismatch {md5.hexdigest()} != {md5_expected}")
    print(f"saved {target} ({target.stat().st_size} bytes, md5 {md5.hexdigest()})", flush=True)


if __name__ == "__main__":
    main()
