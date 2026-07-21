import argparse
import json
import os
import time
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_REPO_ID = "lmms-lab/LVBench"
DEFAULT_LOCAL_DIR = Path("/ssdwork/masiqi/qwj/LVBench/raw/lmms-lab-LVBench")
DEFAULT_EXTRACT_DIR = Path("/ssdwork/masiqi/qwj/LVBench/raw/needed_extracted")
DEFAULT_VIDEO_DIR = Path("/ssdwork/masiqi/qwj/LVBench/videos")


def load_needed(manifest_path: str):
    data = json.load(open(manifest_path, "r", encoding="utf-8"))
    needed = []

    for item in data:
        v = item.get("video") or item.get("video_path") or item.get("video_key")
        if not v:
            continue

        v = Path(str(v)).name
        if not v.lower().endswith((".mp4", ".mkv", ".webm", ".avi", ".mov")):
            v = v + ".mp4"

        needed.append(v)

    return sorted(set(needed))


def already_ready(video_dir: Path, video_name: str) -> bool:
    return (video_dir / video_name).exists()


def link_video(src: Path, video_dir: Path, video_name: str):
    dst = video_dir / video_name

    if dst.exists() or dst.is_symlink():
        return

    os.symlink(str(src.resolve()), str(dst))
    print(f"[Link] {dst} -> {src}")


def safe_remove(path: Path):
    try:
        if path.exists():
            path.unlink()
            print(f"[Remove] {path}")
    except Exception as exc:
        print(f"[Warn] failed to remove {path}: {exc}")


def cleanup_partial_files(zip_path: Path):
    if zip_path.exists():
        return

    parent = zip_path.parent
    if not parent.exists():
        return

    for tmp in parent.glob(zip_path.name + "*"):
        if tmp.exists():
            safe_remove(tmp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/dev_sanity.json")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL_DIR))
    parser.add_argument("--extract-dir", default=str(DEFAULT_EXTRACT_DIR))
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=14)
    parser.add_argument("--sleep-after-fail", type=int, default=10)
    parser.add_argument("--keep-hit-zip", action="store_true")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    extract_dir = Path(args.extract_dir)
    video_dir = Path(args.video_dir)

    local_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    needed = load_needed(args.manifest)

    print("[Manifest]", args.manifest)
    print("[Needed videos]")
    for v in needed:
        print(" ", v)

    failed_chunks = []
    checked_chunks = []
    hit_chunks = []

    for idx in range(args.start, args.end + 1):
        missing = [v for v in needed if not already_ready(video_dir, v)]

        if not missing:
            print("[Success] All needed videos are ready.")
            print(" ", video_dir)
            return

        filename = f"video_chunks/videos_chunk_{idx:03d}.zip"

        print("=" * 80)
        print(f"[Chunk] {filename}")
        print("[Missing]", missing)

        try:
            zip_path = Path(
                hf_hub_download(
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    filename=filename,
                    local_dir=str(local_dir),
                )
            )
        except Exception as exc:
            print(f"[Download failed] {filename}")
            print(repr(exc))
            failed_chunks.append(idx)

            # Remove possible broken temp files.
            expected_zip = local_dir / filename
            cleanup_partial_files(expected_zip)

            if args.sleep_after_fail > 0:
                print(f"[Sleep] {args.sleep_after_fail}s before next chunk")
                time.sleep(args.sleep_after_fail)

            continue

        checked_chunks.append(idx)
        print(f"[Zip] {zip_path}")

        found_any = False

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                hits = []

                for name in names:
                    if Path(name).name in missing:
                        hits.append(name)

                if not hits:
                    print("[Hit] none")
                else:
                    found_any = True
                    hit_chunks.append(idx)

                    print("[Hit]")
                    for h in hits:
                        print(" ", h)

                    for h in hits:
                        video_name = Path(h).name
                        out_path = extract_dir / video_name

                        if not out_path.exists():
                            print(f"[Extract only needed] {h} -> {out_path}")
                            with zf.open(h) as src, open(out_path, "wb") as dst:
                                while True:
                                    chunk = src.read(1024 * 1024)
                                    if not chunk:
                                        break
                                    dst.write(chunk)
                        else:
                            print(f"[Skip extract] already exists: {out_path}")

                        link_video(out_path, video_dir, video_name)

        except zipfile.BadZipFile as exc:
            print(f"[BadZipFile] {zip_path}: {exc}")
            failed_chunks.append(idx)
            safe_remove(zip_path)
            continue

        # Low-disk behavior: remove zip after checking.
        if found_any and args.keep_hit_zip:
            print(f"[Keep zip] {zip_path}")
        else:
            safe_remove(zip_path)

        cleanup_partial_files(zip_path)

    missing = [v for v in needed if not already_ready(video_dir, v)]

    print("=" * 80)
    print("[Finished scanning chunks]")
    print("[Checked chunks]", checked_chunks)
    print("[Hit chunks]", hit_chunks)
    print("[Failed chunks]", failed_chunks)

    if missing:
        print("[Warning] Still missing:")
        for v in missing:
            print(" ", v)
    else:
        print("[Success] All needed videos are ready.")
        print(" ", video_dir)

    if failed_chunks:
        print()
        print("[Retry suggestion]")
        print(
            "python scripts/download_lvbench_needed_videos_resumable.py "
            f"--manifest {args.manifest} "
            f"--start {min(failed_chunks)} --end {max(failed_chunks)}"
        )


if __name__ == "__main__":
    main()
