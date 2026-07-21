import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    data = json.load(open(args.manifest, "r", encoding="utf-8"))
    by_video = defaultdict(list)

    for item in data:
        video_key = item.get("video_key") or Path(str(item.get("video", ""))).stem
        if not video_key:
            raise ValueError(f"Cannot infer video_key from item: {item.get('id', '<no id>')}")
        by_video[video_key].append(item)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for video_key, items in sorted(by_video.items()):
        out = out_dir / f"{video_key}.json"
        out.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Write] {out} | {len(items)} items")


if __name__ == "__main__":
    main()
