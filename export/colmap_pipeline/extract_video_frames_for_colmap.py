"""Extract sharp, native-resolution video frames for a COLMAP image set.

Exact duplicate videos are skipped.  Within each sampling interval the script
keeps the sharpest decoded frame instead of blindly taking the first frame.
Selected frames are written as lossless PNGs without resizing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sharpness_score(frame: Any) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Scoring at half resolution is substantially faster while preserving the
    # ranking between adjacent video frames.  The saved frame remains full-res.
    if gray.shape[1] > 960:
        scale = 960.0 / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in path.stem)


def extract_video(
    video: Path,
    video_index: int,
    output_dir: Path,
    target_fps: float,
    png_compression: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid FPS reported for {video}: {source_fps}")

    interval = max(1, round(source_fps / target_fps))
    rows: list[dict[str, Any]] = []
    frame_index = 0
    selected_index = 0
    best_frame = None
    best_frame_index = -1
    best_score = -1.0
    prefix = f"v{video_index:02d}_{safe_stem(video)}"

    def save_best() -> None:
        nonlocal best_frame, best_frame_index, best_score, selected_index
        if best_frame is None:
            return
        timestamp_ms = round(best_frame_index * 1000.0 / source_fps)
        image_name = (
            f"{prefix}_{selected_index:06d}_"
            f"f{best_frame_index:07d}_t{timestamp_ms:09d}ms.png"
        )
        image_path = output_dir / image_name
        ok = cv2.imwrite(
            str(image_path),
            best_frame,
            [cv2.IMWRITE_PNG_COMPRESSION, png_compression],
        )
        if not ok:
            raise RuntimeError(f"Failed to write image: {image_path}")
        rows.append(
            {
                "image": image_name,
                "source_video": video.name,
                "source_frame": best_frame_index,
                "timestamp_seconds": f"{best_frame_index / source_fps:.6f}",
                "sharpness": f"{best_score:.6f}",
                "width": width,
                "height": height,
            }
        )
        selected_index += 1
        best_frame = None
        best_frame_index = -1
        best_score = -1.0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        score = sharpness_score(frame)
        if score > best_score:
            best_frame = frame.copy()
            best_frame_index = frame_index
            best_score = score
        frame_index += 1
        if frame_index % interval == 0:
            save_best()
        if frame_index % 300 == 0:
            print(
                f"[{video_index}] {video.name}: decoded {frame_index}/{source_frames}, "
                f"selected {selected_index}",
                flush=True,
            )

    save_best()
    capture.release()
    summary = {
        "source_video": video.name,
        "source_fps": source_fps,
        "source_frames_reported": source_frames,
        "source_frames_decoded": frame_index,
        "resolution": [width, height],
        "sampling_interval_frames": interval,
        "selected_frames": selected_index,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=3.0)
    parser.add_argument("--png-compression", type=int, choices=range(0, 10), default=1)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.target_fps <= 0:
        raise ValueError("--target-fps must be positive")
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)

    videos = sorted(
        path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise RuntimeError(f"No video files found in {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_pngs = list(output_dir.glob("*.png"))
    if existing_pngs:
        raise RuntimeError(
            f"Output directory already contains {len(existing_pngs)} PNG files: {output_dir}"
        )

    cv2.setNumThreads(max(1, args.threads))
    unique_videos: list[Path] = []
    duplicates: list[dict[str, str]] = []
    seen_hashes: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for video in videos:
        digest = sha256_file(video)
        hashes[video.name] = digest
        if digest in seen_hashes:
            duplicates.append(
                {
                    "duplicate": video.name,
                    "same_as": seen_hashes[digest].name,
                    "sha256": digest,
                }
            )
            print(f"Skipping exact duplicate: {video.name} == {seen_hashes[digest].name}", flush=True)
        else:
            seen_hashes[digest] = video
            unique_videos.append(video)

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for video_index, video in enumerate(unique_videos, start=1):
        rows, summary = extract_video(
            video,
            video_index,
            output_dir,
            args.target_fps,
            args.png_compression,
        )
        all_rows.extend(rows)
        summary["sha256"] = hashes[video.name]
        summaries.append(summary)

    manifest_path = output_dir.parent / "frames_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = output_dir.parent / "extraction_summary.json"
    summary_payload = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "target_fps": args.target_fps,
        "format": "lossless PNG, native resolution",
        "unique_videos": summaries,
        "exact_duplicates_skipped": duplicates,
        "total_selected_frames": len(all_rows),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"Finished: {len(all_rows)} images in {output_dir}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
