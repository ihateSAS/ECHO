#!/usr/bin/env python3


from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "benjamin-paine/free-music-archive-small"
OUTPUT_DIR = Path("data/real_stereo")


ROW_OFFSETS = (0, 900, 1800, 2700, 3600, 4500, 5400, 6300, 7100)


def fetch_rows(offset: int, length: int) -> dict:
    url = (
        "https://datasets-server.huggingface.co/rows?dataset="
        + urllib.parse.quote(DATASET, safe="")
        + f"&config=default&split=train&offset={offset}&length={length}"
    )
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def genre_names() -> list[str]:
    for feature in fetch_rows(0, 1)["features"]:
        if feature["name"] == "genres":
            return feature["type"]["feature"]["names"]
    raise RuntimeError("the dataset no longer has a genres feature")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = genre_names()
    seen_artists: set[str] = set()
    manifest: list[dict] = []

    for offset in ROW_OFFSETS:
        for entry in fetch_rows(offset, 12)["rows"]:
            row = entry["row"]
            if row["artist"] in seen_artists:
                continue
            seen_artists.add(row["artist"])
            manifest.append(
                {
                    "file": f"fma_{entry['row_idx']:05d}.mp3",
                    "row_idx": entry["row_idx"],
                    "artist": row["artist"],
                    "title": row["title"],
                    "album": row["album_title"],
                    "genres": [names[index] for index in row["genres"]],
                    "url": row["url"],


                    "_src": row["audio"][0]["src"],
                }
            )
            break

    for record in manifest:
        path = OUTPUT_DIR / record["file"]
        source = record.pop("_src")
        if not path.exists():
            with urllib.request.urlopen(source, timeout=180) as response:
                path.write_bytes(response.read())
        size_mb = path.stat().st_size / 1e6
        print(f"{record['file']}  {size_mb:5.2f} MB  {record['artist']} - {record['title']}")

    (OUTPUT_DIR / "tracks.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n{len(manifest)} tracks in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
