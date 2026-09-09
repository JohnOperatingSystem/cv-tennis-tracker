import cv2
import numpy as np
from pathlib import Path

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        raise ValueError(f"Could not determine FPS for video: {video_path}")
    return fps


def find_scene_cuts(
    frames,
    fps,
    minimum_seconds=1.5,
    difference_threshold=30.0,
    correlation_threshold=0.35,
):
    """Return frame indices containing hard broadcast cuts.

    The minimum duration applies to the start of the video so a brief exposure
    change on the opening frames is not treated as a cut.
    """
    if len(frames) < 2:
        return []
    earliest = max(1, round(float(fps) * minimum_seconds))
    cuts = []
    previous = cv2.resize(
        cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY),
        (160, 90),
    )
    for frame_num in range(1, len(frames)):
        current = cv2.resize(
            cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY),
            (160, 90),
        )
        if frame_num >= earliest:
            difference = float(np.mean(cv2.absdiff(current, previous)))
            current_values = current.ravel().astype(float)
            previous_values = previous.ravel().astype(float)
            if np.std(current_values) < 1e-6 or np.std(previous_values) < 1e-6:
                correlation = -1.0 if difference >= difference_threshold else 1.0
            else:
                correlation = float(
                    np.corrcoef(current_values, previous_values)[0, 1]
                )
            if (
                np.isfinite(correlation)
                and difference >= difference_threshold
                and correlation <= correlation_threshold
            ):
                cuts.append(frame_num)
        previous = current
    return cuts


def find_first_scene_cut(
    frames,
    fps,
    minimum_seconds=1.5,
    difference_threshold=30.0,
    correlation_threshold=0.35,
):
    """Return the first hard broadcast cut, or ``len(frames)``."""
    cuts = find_scene_cuts(
        frames,
        fps,
        minimum_seconds=minimum_seconds,
        difference_threshold=difference_threshold,
        correlation_threshold=correlation_threshold,
    )
    return cuts[0] if cuts else len(frames)

def save_video(output_video_frames, output_video_path, fps=30):
    if not output_video_frames:
        raise ValueError("Cannot save a video with no frames")
    if fps <= 0:
        raise ValueError("Video FPS must be greater than zero")

    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        float(fps),
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0]),
    )
    if not out.isOpened():
        raise OSError(f"Could not open video writer for {output_path}")

    for frame in output_video_frames:
        out.write(frame)
    out.release()
