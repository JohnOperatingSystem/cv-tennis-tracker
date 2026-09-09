"""Collect diverse, source-traceable wheelchair-tennis training images.

The script reads public YouTube videos, seeks to evenly spaced timestamps,
uses the project's court-keypoint model to prefer elevated full-court views,
and writes only JPEG frames plus source/timestamp metadata. Temporary source
downloads are removed after sampling each match.
"""

import argparse
import csv
import math
from pathlib import Path
import sys

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
VENDOR_DIR = BASE_DIR / ".vendor"
sys.path.insert(0, str(BASE_DIR))
if VENDOR_DIR.is_dir():
    sys.path.insert(0, str(VENDOR_DIR))

from court_line_detector import CourtLineDetector

try:
    from yt_dlp import YoutubeDL
except ImportError as error:
    raise SystemExit(
        "yt-dlp is required. Install it with: pip install yt-dlp"
    ) from error


SOURCES = [
    {
        "id": "Eztpzmeo7mM",
        "channel": "Paralympic Games",
        "event": "Paris 2024 men's singles semifinal",
    },
    {
        "id": "ic1Nf4cBjc0",
        "channel": "Paralympic Games",
        "event": "Tokyo 2020 quad singles gold",
    },
    {
        "id": "AeeNepmMkzI",
        "channel": "Paralympic Games",
        "event": "Rio 2016 quad singles semifinal",
    },
    {
        "id": "oRkHzVTINEo",
        "channel": "Paralympic Games",
        "event": "Rio 2016 men's singles bronze",
    },
    {
        "id": "ZBH9JPACUc8",
        "channel": "Paralympic Games",
        "event": "Paris 2024 quad singles gold",
    },
    {
        "id": "0Vk8xqt4Voo",
        "channel": "Paralympic Games",
        "event": "Paris 2024 women's singles gold",
    },
    {
        "id": "xvIcKh11F_c",
        "channel": "US Open Tennis Championships",
        "event": "US Open 2020 men's singles final",
    },
    {
        "id": "CBO9hMKeBU0",
        "channel": "US Open Tennis Championships",
        "event": "US Open 2018 men's singles final",
    },
    {
        "id": "3A1OkxjzM7o",
        "channel": "US Open Tennis Championships",
        "event": "US Open 2018 quad doubles final",
    },
    {
        "id": "wrhkPMfoGpw",
        "channel": "USTA",
        "event": "US Open 2017 women's doubles semifinal",
    },
    {
        "id": "U25W58LYBaI",
        "channel": "Australian Open",
        "event": "Australian Open 2016 men's singles final",
    },
    {
        "id": "qmAFbStBW84",
        "channel": "Australian Open",
        "event": "Australian Open 2016 quad singles final",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect unlabelled wheelchair-tennis frames from YouTube."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "datasets" / "wheelchair_tennis_youtube",
    )
    parser.add_argument("--target-count", type=int, default=500)
    parser.add_argument("--source-limit", type=int, default=len(SOURCES))
    parser.add_argument("--candidates-per-source", type=int, default=150)
    parser.add_argument("--max-per-source", type=int, default=60)
    parser.add_argument("--maximum-height", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--minimum-court-score", type=float, default=0.72)
    parser.add_argument(
        "--stream-direct",
        action="store_true",
        help="Seek the remote stream instead of using one temporary download.",
    )
    return parser.parse_args()


def _format_timestamp(seconds):
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _court_view_score(frame, detector):
    """Score whether a frame contains a complete elevated court view."""
    try:
        points = np.asarray(detector.predict(frame), dtype=float).reshape(-1, 2)
    except (ValueError, cv2.error):
        return 0.0
    if len(points) < 4 or not np.isfinite(points).all():
        return 0.0

    height, width = frame.shape[:2]
    inside = (
        (points[:, 0] >= -0.05 * width)
        & (points[:, 0] <= 1.05 * width)
        & (points[:, 1] >= -0.05 * height)
        & (points[:, 1] <= 1.05 * height)
    )
    inside_ratio = float(np.mean(inside))
    corners = points[:4]
    polygon = np.float32(corners[[0, 1, 3, 2]])
    area_ratio = abs(float(cv2.contourArea(polygon))) / (width * height)
    far_y = float(np.mean(corners[:2, 1]))
    near_y = float(np.mean(corners[2:, 1]))
    depth_ratio = max(0.0, (near_y - far_y) / height)
    far_width = float(np.linalg.norm(corners[1] - corners[0]))
    near_width = float(np.linalg.norm(corners[3] - corners[2]))
    court_center_x = float(np.mean(corners[:, 0]))
    width_ratio = near_width / max(1.0, far_width)

    # The regressor always emits a court-shaped quadrilateral, even for
    # close-ups and title cards. Require the strong perspective geometry of an
    # elevated broadcast view before considering its softer confidence score.
    if (
        inside_ratio < 0.85
        or not 0.12 <= area_ratio <= 0.55
        or depth_ratio < 0.32
        or far_width > 0.46 * width
        or near_width < 0.52 * width
        or width_ratio < 1.75
        or abs(court_center_x - 0.5 * width) > 0.18 * width
        or far_y > 0.43 * height
        or near_y < 0.70 * height
    ):
        return 0.0
    perspective_score = float(np.clip(
        width_ratio / 2.5,
        0.0,
        1.0,
    ))
    area_score = float(np.clip(area_ratio / 0.22, 0.0, 1.0))
    depth_score = float(np.clip(depth_ratio / 0.45, 0.0, 1.0))
    sharpness = float(cv2.Laplacian(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F
    ).var())
    sharpness_score = float(np.clip(sharpness / 180.0, 0.0, 1.0))
    return (
        0.35 * inside_ratio
        + 0.30 * area_score
        + 0.20 * depth_score
        + 0.10 * perspective_score
        + 0.05 * sharpness_score
    )


def _get_video_info(source, maximum_height, temporary_dir=None):
    url = f"https://www.youtube.com/watch?v={source['id']}"
    options = {
        "format": (
            f"bestvideo[height<={maximum_height}][ext=mp4][protocol=https]/"
            f"bestvideo[height<={maximum_height}][protocol=https]"
        ),
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
    }
    if temporary_dir is not None:
        temporary_dir.mkdir(parents=True, exist_ok=True)
        options["outtmpl"] = str(temporary_dir / f"{source['id']}.%(ext)s")
        print(f"  downloading temporary {maximum_height}p source...", flush=True)
    with YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=temporary_dir is not None)
        media_path = (
            None
            if temporary_dir is None
            else Path(downloader.prepare_filename(info))
        )
    result = {
        "url": url,
        "stream_url": info["url"],
        "duration": float(info["duration"]),
        "title": info.get("title") or source["event"],
        "channel": info.get("channel") or source["channel"],
        "width": info.get("width"),
        "height": info.get("height"),
    }
    if media_path is not None:
        result["media_path"] = media_path
    return result


def _collect_source_candidates(
    source,
    detector,
    candidate_count,
    maximum_height,
    minimum_court_score,
    jpeg_quality,
    temporary_dir,
    stream_direct,
):
    info = _get_video_info(
        source,
        maximum_height,
        None if stream_direct else temporary_dir,
    )
    # Avoid introductions, ceremonies, and final interviews. Even spacing is
    # intentionally much wider than adjacent video frames.
    timestamps = np.linspace(
        0.08 * info["duration"],
        0.92 * info["duration"],
        candidate_count,
    )
    input_location = str(info.get("media_path") or info["stream_url"])
    capture = cv2.VideoCapture(input_location, cv2.CAP_FFMPEG)
    if not capture.isOpened():
        media_path = info.get("media_path")
        if media_path is not None and media_path.is_file():
            media_path.unlink()
        raise OSError(f"Could not open YouTube stream {source['id']}")

    candidates = []
    try:
        for candidate_index, timestamp in enumerate(timestamps, start=1):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp * 1000.0))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            score = _court_view_score(frame, detector)
            if score < minimum_court_score:
                continue
            encoded, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            if not encoded:
                continue
            candidates.append(
                {
                    "score": score,
                    "timestamp_seconds": float(timestamp),
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "jpeg": buffer.tobytes(),
                    "info": info,
                }
            )
            if candidate_index % 25 == 0:
                print(
                    f"  {source['id']}: checked {candidate_index}/"
                    f"{candidate_count}, retained {len(candidates)}",
                    flush=True,
                )
    finally:
        capture.release()
        media_path = info.get("media_path")
        if media_path is not None and media_path.is_file():
            media_path.unlink()
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def _write_readme(output_dir, count):
    content = f"""# Wheelchair Tennis YouTube Frames

This folder contains {count} unlabelled JPEG frames selected for training a
`wheelchair_player` detector. Frames favor elevated, complete-court broadcast
views and are sampled from multiple official tournament channels.

## Layout

- `images/unlabelled/`: images awaiting bounding-box annotation
- `manifest.csv`: exact source URL, video ID, title, timestamp, and view score
- `sources.csv`: one row per source video
- `contact_sheet.jpg`: an evenly spaced visual-QA sample (not training input)

## Annotation rule

Draw one tight box around each complete athlete-and-wheelchair combination.
Label every visible wheelchair player, including distant players. Do not label
spectators, officials, ordinary chairs, or able-bodied players as
`wheelchair_player`.

Split train/validation/test by `video_id`, never by random images, to prevent
near-identical frames from the same broadcast appearing in multiple splits.

## Rights

The image files retain source attribution in `manifest.csv`. Verify the source
license and obtain any required permission before redistribution or commercial
use. YouTube/source terms still apply.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def _write_contact_sheet(images_dir, output_path, maximum_images=48):
    """Write a compact visual-QA sheet without adding files to training data."""
    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        return
    sample_indices = np.linspace(
        0,
        len(image_paths) - 1,
        min(maximum_images, len(image_paths)),
        dtype=int,
    )
    tile_width, tile_height = 256, 162
    columns = 6
    rows = math.ceil(len(sample_indices) / columns)
    sheet = np.full(
        (rows * tile_height, columns * tile_width, 3),
        24,
        dtype=np.uint8,
    )
    for tile_number, image_index in enumerate(sample_indices):
        frame = cv2.imread(str(image_paths[image_index]))
        if frame is None:
            continue
        frame = cv2.resize(
            frame,
            (tile_width, tile_height - 18),
            interpolation=cv2.INTER_AREA,
        )
        row, column = divmod(tile_number, columns)
        x, y = column * tile_width, row * tile_height
        sheet[y : y + tile_height - 18, x : x + tile_width] = frame
        cv2.putText(
            sheet,
            image_paths[image_index].stem.split("_")[1],
            (x + 5, y + tile_height - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), sheet)


def main():
    args = parse_args()
    if args.target_count < 1:
        raise ValueError("Target count must be positive")
    if args.candidates_per_source < 1 or args.max_per_source < 1:
        raise ValueError("Candidate and per-source counts must be positive")

    sources = SOURCES[: max(1, min(args.source_limit, len(SOURCES)))]
    images_dir = args.output / "images" / "unlabelled"
    temporary_dir = args.output / ".temporary_sources"
    images_dir.mkdir(parents=True, exist_ok=True)
    detector = CourtLineDetector(BASE_DIR / "models" / "keypoints_model.pth")

    manifest_rows = []
    source_rows = []
    image_number = 1
    for source_index, source in enumerate(sources):
        if len(manifest_rows) >= args.target_count:
            break
        sources_remaining = len(sources) - source_index
        images_needed = args.target_count - len(manifest_rows)
        source_target = min(
            args.max_per_source,
            max(1, math.ceil(images_needed / sources_remaining)),
        )
        print(
            f"Source {source_index + 1}/{len(sources)}: "
            f"{source['event']} (target {source_target})",
            flush=True,
        )
        try:
            candidates = _collect_source_candidates(
                source,
                detector,
                max(args.candidates_per_source, source_target * 2),
                args.maximum_height,
                args.minimum_court_score,
                args.jpeg_quality,
                temporary_dir,
                args.stream_direct,
            )
        except Exception as error:
            print(f"  skipped {source['id']}: {error}", flush=True)
            continue

        selected = sorted(
            candidates[:source_target],
            key=lambda item: item["timestamp_seconds"],
        )
        for candidate in selected:
            timestamp_ms = round(candidate["timestamp_seconds"] * 1000)
            filename = (
                f"wheelchair_{image_number:04d}_{source['id']}_"
                f"{timestamp_ms:010d}.jpg"
            )
            (images_dir / filename).write_bytes(candidate["jpeg"])
            manifest_rows.append(
                {
                    "filename": filename,
                    "video_id": source["id"],
                    "source_url": candidate["info"]["url"],
                    "channel": candidate["info"]["channel"],
                    "title": candidate["info"]["title"],
                    "event": source["event"],
                    "timestamp": _format_timestamp(
                        candidate["timestamp_seconds"]
                    ),
                    "timestamp_seconds": round(
                        candidate["timestamp_seconds"], 3
                    ),
                    "width": candidate["width"],
                    "height": candidate["height"],
                    "court_view_score": round(candidate["score"], 4),
                }
            )
            image_number += 1
        source_rows.append(
            {
                "video_id": source["id"],
                "source_url": f"https://www.youtube.com/watch?v={source['id']}",
                "channel": source["channel"],
                "event": source["event"],
                "images_selected": len(selected),
            }
        )
        print(
            f"  selected {len(selected)}; total {len(manifest_rows)}/"
            f"{args.target_count}",
            flush=True,
        )

    manifest_fields = [
        "filename",
        "video_id",
        "source_url",
        "channel",
        "title",
        "event",
        "timestamp",
        "timestamp_seconds",
        "width",
        "height",
        "court_view_score",
    ]
    with (args.output / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    source_fields = [
        "video_id",
        "source_url",
        "channel",
        "event",
        "images_selected",
    ]
    with (args.output / "sources.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=source_fields)
        writer.writeheader()
        writer.writerows(source_rows)
    _write_readme(args.output, len(manifest_rows))
    _write_contact_sheet(
        images_dir,
        args.output / "contact_sheet.jpg",
    )
    if temporary_dir.is_dir() and not any(temporary_dir.iterdir()):
        temporary_dir.rmdir()

    print(f"Finished: {len(manifest_rows)} images in {images_dir}")
    if len(manifest_rows) < args.target_count:
        raise SystemExit(
            f"Only {len(manifest_rows)} of {args.target_count} requested images "
            "passed the full-court filter"
        )


if __name__ == "__main__":
    main()
