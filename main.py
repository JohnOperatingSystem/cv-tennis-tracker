import argparse
import json
import os
from pathlib import Path
import pickle

import cv2
import numpy as np
import pandas as pd

from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
from trackers import (
    BallTracker,
    BallTrajectoryReconstructor,
    PlayerTracker,
    PointAnalyzer,
    SpeedTracker,
)
from utils import (
    draw_player_stats,
    find_scene_cuts,
    get_video_fps,
    read_video,
    save_video,
)

BASE_DIR = Path(__file__).resolve().parent
DETECTION_CACHE_VERSION = 5


def _detection_cache_key(
    video_path,
    retained_frame_count,
    fps,
    player_detector=None,
    first_source_frame=0,
):
    source = Path(video_path).resolve()
    stat = source.stat()
    return {
        "version": DETECTION_CACHE_VERSION,
        "source": str(source),
        "size": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
        "frames": int(retained_frame_count),
        "fps": round(float(fps), 6),
        "first_source_frame": int(first_source_frame),
        "player_detector": player_detector,
    }


def _select_point_scene(video_frames, fps, court_line_detector):
    """Select the first complete-court shot, skipping broadcast pre-roll."""
    cuts = find_scene_cuts(video_frames, fps)
    if not cuts:
        return 0, len(video_frames)

    boundaries = [0, *cuts, len(video_frames)]
    minimum_segment_frames = max(2, round(0.75 * float(fps)))
    for start, end in zip(boundaries, boundaries[1:]):
        if end - start < minimum_segment_frames:
            continue
        sample_count = min(5, end - start)
        sample_indices = np.unique(
            np.linspace(start, end - 1, sample_count, dtype=int)
        )
        plausible_views = 0
        for frame_num in sample_indices:
            frame = video_frames[frame_num]
            keypoints = court_line_detector.predict(frame)
            if court_line_detector.is_plausible_full_court(
                keypoints,
                frame.shape,
            ):
                plausible_views += 1
        if plausible_views >= max(2, int(np.ceil(0.40 * sample_count))):
            return start, end

    # Preserve the previous behavior for unusual court views the keypoint
    # geometry gate cannot recognize.
    return 0, cuts[0]


def _load_detection_cache(path, expected_key):
    if not path.is_file():
        return None
    try:
        with path.open("rb") as stream:
            cache = pickle.load(stream)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None
    return cache if cache.get("key") == expected_key else None


def _save_detection_cache(
    path,
    key,
    raw_player_detections,
    player_detections,
    keypoints,
    ball_detections,
    player_mode,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(
            {
                "key": key,
                "raw_player_detections": raw_player_detections,
                "player_detections": player_detections,
                "court_keypoints": keypoints,
                "ball_detections": ball_detections,
                "player_mode": player_mode,
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _file_signature(path):
    path = Path(path)
    if not path.is_file():
        return None
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def _load_bounce_cache(path, expected_key):
    if not path.is_file():
        return None
    try:
        with path.open("rb") as stream:
            cache = pickle.load(stream)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None
    if cache.get("key") != expected_key:
        return None
    return cache.get("predictions")


def _save_bounce_cache(path, key, predictions):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(
            {"key": key, "predictions": predictions},
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def calculate_player_stats(
    frame_count,
    ball_events,
    speed_series,
    ball_visibility_states=None,
):
    """Build per-frame display/export statistics from rolling speeds."""
    player_stats = pd.DataFrame(
        {
            "frame_num": np.arange(frame_count),
            "player_1_last_player_speed": speed_series.player_speeds_kmh[1],
            "player_2_last_player_speed": speed_series.player_speeds_kmh[2],
            "ball_speed": speed_series.ball_speeds_kmh,
            "ball_speed_quality": speed_series.ball_speed_quality,
            "ball_visibility": (
                list(ball_visibility_states)
                if ball_visibility_states is not None
                else ["unknown"] * frame_count
            ),
        }
    )
    for player_id in (1, 2):
        speeds = player_stats[f"player_{player_id}_last_player_speed"]
        player_stats[f"player_{player_id}_average_player_speed"] = (
            speeds.expanding().mean()
        )

        shot_speed_updates = np.full(frame_count, np.nan)
        for event in ball_events:
            if event.get("type") != "hit" or event.get("player_id") != player_id:
                continue
            start = int(event["frame"])
            end = min(frame_count, start + speed_series.window_frames * 2 + 1)
            samples = [
                speed
                for speed in speed_series.ball_speeds_kmh[start:end]
                if speed is not None and np.isfinite(speed)
            ]
            if samples:
                shot_speed_updates[start] = max(samples)
        shot_speeds = pd.Series(shot_speed_updates).ffill()
        player_stats[f"player_{player_id}_last_shot_speed"] = shot_speeds
        shot_only = pd.Series(shot_speed_updates)
        cumulative_average = shot_only.expanding().mean().ffill()
        player_stats[f"player_{player_id}_average_shot_speed"] = cumulative_average

    return player_stats


def main(
    input_video_path=None,
    open_output=True,
    event_model_path=None,
    outcome_model_path=None,
    player_mode="auto",
    wheelchair_model_path=None,
    bounce_model_path=None,
    use_bounce_model=True,
    bounce_confidence=0.35,
):
    input_video_path = str(
        Path(input_video_path)
        if input_video_path is not None
        else BASE_DIR / "input_videos" / "input_video4.mp4"
    )
    video_fps = get_video_fps(input_video_path)
    video_frames = read_video(input_video_path)
    if not video_frames:
        raise ValueError(f"Video contains no readable frames: {input_video_path}")
    court_line_detector = CourtLineDetector(
        str(BASE_DIR / "models" / "keypoints_model.pth")
    )
    scene_start, scene_end = _select_point_scene(
        video_frames,
        video_fps,
        court_line_detector,
    )
    if scene_start:
        print(
            f"Ignoring {scene_start} pre-point frames before full-court "
            "play"
        )
    if scene_end < len(video_frames):
        print(
            f"Ignoring {len(video_frames) - scene_end} post-point frames "
            f"after broadcast cut at frame {scene_end}"
        )
    video_frames = video_frames[scene_start:scene_end]

    event_model_path = event_model_path or os.environ.get("TENNIS_EVENT_MODEL")
    outcome_model_path = outcome_model_path or os.environ.get("TENNIS_OUTCOME_MODEL")
    temporal_predictions = []
    temporal_outcome = None
    if event_model_path or outcome_model_path:
        from tenniset.temporal import TemporalPredictor

        if event_model_path:
            event_predictor = TemporalPredictor(event_model_path)
            temporal_predictions = event_predictor.predict_events(video_frames)
            print(
                f"Temporal model supplied {len(temporal_predictions)} "
                "serve/hit observations"
            )
        if outcome_model_path:
            outcome_predictor = TemporalPredictor(outcome_model_path)
            temporal_outcome = outcome_predictor.predict_outcome(video_frames)
            print("Temporal outcome:", temporal_outcome)

    bounce_predictions = []
    configured_bounce_path = (
        bounce_model_path
        or os.environ.get("TENNIS_BOUNCE_MODEL")
        or BASE_DIR / "models" / "tennis_bounce_e2espot.pt"
    )
    configured_bounce_path = Path(configured_bounce_path)
    if use_bounce_model and configured_bounce_path.is_file():
        bounce_cache_path = (
            BASE_DIR
            / "output_videos"
            / f"{Path(input_video_path).stem}_bounce_predictions.pkl"
        )
        source_stat = Path(input_video_path).resolve().stat()
        bounce_cache_key = {
            "version": 1,
            "source": str(Path(input_video_path).resolve()),
            "source_size": source_stat.st_size,
            "source_modified_ns": source_stat.st_mtime_ns,
            "scene_start": scene_start,
            "scene_end": scene_end,
            "fps": round(float(video_fps), 6),
            "model": _file_signature(configured_bounce_path),
            "confidence_threshold": round(float(bounce_confidence), 6),
        }
        bounce_predictions = _load_bounce_cache(
            bounce_cache_path,
            bounce_cache_key,
        )
        if bounce_predictions is None:
            from tenniset.bounce_spotter import TennisBounceSpotter

            print("Running exact-frame learned bounce detector")
            bounce_spotter = TennisBounceSpotter(
                configured_bounce_path,
                confidence_threshold=bounce_confidence,
            )
            bounce_predictions = bounce_spotter.predict(video_frames, video_fps)
            _save_bounce_cache(
                bounce_cache_path,
                bounce_cache_key,
                bounce_predictions,
            )
        else:
            print(f"Using cached learned bounces: {bounce_cache_path.name}")
        print(
            "Learned bounce observations:",
            [
                (
                    prediction["frame"],
                    prediction["label"],
                    round(prediction["confidence"], 3),
                )
                for prediction in bounce_predictions
            ],
        )
    elif use_bounce_model and bounce_model_path is not None:
        raise FileNotFoundError(
            f"Bounce model does not exist: {configured_bounce_path}"
        )
    elif use_bounce_model:
        print("Learned bounce checkpoint not found; using trajectory fallback")

    wheelchair_model_path = (
        wheelchair_model_path
        or os.environ.get("TENNIS_WHEELCHAIR_MODEL")
        or BASE_DIR / "models" / "wheelchair_best.pt"
    )
    player_tracker = PlayerTracker(
        model_path=str(BASE_DIR / "yolov8x.pt"),
        wheelchair_model_path=str(wheelchair_model_path),
        mode=player_mode,
    )
    ball_tracker = BallTracker(model_path=str(BASE_DIR / "models" / "best.pt"))
    ball_tracker.configure_for_video(
        frame_height=video_frames[0].shape[0],
        fps=video_fps,
    )
    cache_path = (
        BASE_DIR
        / "output_videos"
        / f"{Path(input_video_path).stem}_detections.pkl"
    )
    cache_key = _detection_cache_key(
        input_video_path,
        len(video_frames),
        video_fps,
        player_detector=player_tracker.cache_signature(),
        first_source_frame=scene_start,
    )
    cached = _load_detection_cache(cache_path, cache_key)
    if cached is None:
        raw_player_detections = player_tracker.detect_frames(video_frames)
        court_keypoints_per_frame = court_line_detector.predict_frames(video_frames)
        court_keypoints_per_frame = court_line_detector.stabilize_keypoints(
            court_keypoints_per_frame,
            # The detector still runs on every frame. A one-second centered
            # median removes jitter while retaining real pans and zooms.
            assume_fixed_camera=False,
            window_size=max(5, round(1.00 * video_fps) | 1),
        )
        player_detections = player_tracker.choose_and_filter_players(
            court_keypoints_per_frame, raw_player_detections
        )
        detected_ball_detections = ball_tracker.detect_frames(
            video_frames,
            court_keypoints_per_frame=court_keypoints_per_frame,
        )
        _save_detection_cache(
            cache_path,
            cache_key,
            raw_player_detections,
            player_detections,
            court_keypoints_per_frame,
            detected_ball_detections,
            player_tracker.active_mode,
        )
        active_player_mode = player_tracker.active_mode
    else:
        print(f"Using cached neural detections: {cache_path.name}")
        player_detections = cached["player_detections"]
        court_keypoints_per_frame = cached["court_keypoints"]
        detected_ball_detections = cached["ball_detections"]
        active_player_mode = cached["player_mode"]
    allowed_bounces = 2 if active_player_mode == "wheelchair" else 1
    print(
        f"Applying {allowed_bounces}-bounce "
        f"{'wheelchair' if allowed_bounces == 2 else 'standard'} rules"
    )
    ball_detections = ball_tracker.interpolate_ball_positions(
        detected_ball_detections,
        max_gap=ball_tracker.interpolation_max_gap,
        maximum_average_step=ball_tracker.max_jump_per_frame,
    )
    display_ball_detections = ball_tracker.interpolate_ball_positions(
        ball_detections,
        max_gap=ball_tracker.display_interpolation_max_gap,
        maximum_average_step=ball_tracker.max_jump_per_frame,
    )

    mini_court = MiniCourt(video_frames[0])
    player_mini_court_detections, ball_mini_court_detections = (
        mini_court.convert_bounding_boxes_to_mini_court_coordinates(
            player_detections,
            ball_detections,
            court_keypoints_per_frame,
        )
    )
    raw_ball_mini_court_detections = [
        dict(frame_positions)
        for frame_positions in ball_mini_court_detections
    ]
    trajectory_reconstructor = BallTrajectoryReconstructor(
        video_fps,
        frame_width=video_frames[0].shape[1],
        frame_height=video_frames[0].shape[0],
        allowed_bounces=allowed_bounces,
    )
    (
        ball_mini_court_detections,
        ball_heights,
        ball_events,
    ) = trajectory_reconstructor.reconstruct(
        ball_detections,
        player_detections,
        player_mini_court_detections,
        ball_mini_court_detections,
        mini_court,
        temporal_predictions=temporal_predictions,
        bounce_predictions=bounce_predictions,
    )
    print(
        "Ball events:",
        [
            (
                event["frame"],
                event["type"],
                event.get("bounce_number"),
                event.get("event_source")
                or ("inferred" if event.get("inferred", False) else "trajectory"),
                (
                    round(float(event["learned_confidence"]), 3)
                    if event.get("learned_confidence") is not None
                    else None
                ),
            )
            for event in ball_events
        ],
    )
    try:
        trajectory_report = trajectory_reconstructor.validate_trajectory(
            ball_mini_court_detections,
            ball_heights,
            ball_events,
            mini_court,
        )
        trajectory_report["valid"] = True
    except ValueError as error:
        # Preserve usable speeds, raw point evidence, and video output when a
        # noisy clip fails a reconstructed-trajectory sanity check.
        trajectory_report = {
            "valid": False,
            "warning": str(error),
        }
    observed_ball_frames = {
        frame_num
        for frame_num, detection in enumerate(detected_ball_detections)
        if detection.get(1)
    }
    short_interpolated_ball_frames = {
        frame_num
        for frame_num, detection in enumerate(ball_detections)
        if detection.get(1) and frame_num not in observed_ball_frames
    }
    projected_ball_image_centers = ball_tracker.project_trajectory_to_image(
        ball_mini_court_detections,
        ball_heights,
        player_detections,
        player_mini_court_detections,
        court_keypoints_per_frame,
        mini_court,
    )
    projected_ball_image_centers = (
        ball_tracker.estimate_bounded_gap_image_centers(
            detected_ball_detections,
            projected_ball_image_centers,
            maximum_gap_frames=max(3, round(3.0 * video_fps)),
        )
    )
    ball_visibility_states = ball_tracker.classify_visibility(
        detected_ball_detections,
        ball_detections,
        projected_ball_image_centers,
        video_frames[0].shape,
    )
    predicted_high_arc_frames = {
        frame_num
        for frame_num, state in enumerate(ball_visibility_states)
        if state == "out_of_frame"
    }
    visibility_counts = {
        state: ball_visibility_states.count(state)
        for state in (
            "observed",
            "interpolated",
            "out_of_frame",
            "occluded",
            "unknown",
        )
    }
    print("Ball visibility:", visibility_counts)
    inferred_flight_frames = {
        frame_num
        for event, next_event in zip(ball_events, ball_events[1:])
        if event["type"] == "hit"
        and next_event["type"] == "bounce"
        and next_event.get("inferred", False)
        for frame_num in range(event["frame"], next_event["frame"] + 1)
    }
    inferred_prediction_frames = {
        frame_num
        for frame_num in inferred_flight_frames - observed_ball_frames
        if ball_visibility_states[frame_num] == "interpolated"
    }
    display_ball_detections = (
        ball_tracker.align_predicted_boxes_to_trajectory(
            display_ball_detections,
            ball_mini_court_detections,
            ball_heights,
            player_detections,
            player_mini_court_detections,
            court_keypoints_per_frame,
            mini_court,
            inferred_prediction_frames,
        )
    )
    print("Trajectory validation:", trajectory_report)

    motion_candidates = trajectory_reconstructor.find_motion_candidates(
        ball_detections,
        player_detections,
        bounce_predictions=bounce_predictions,
    )
    point_analyzer = PointAnalyzer(
        video_fps,
        singles=True,
        clip_ends_with_point=True,
        allowed_bounces=allowed_bounces,
    )
    point_result = point_analyzer.analyze(
        ball_events,
        motion_candidates,
        raw_ball_mini_court_detections,
        player_mini_court_detections,
        mini_court,
        ball_heights=ball_heights,
        observed_ball_frames=observed_ball_frames,
        temporal_outcome=temporal_outcome,
    )
    print("Point result:", point_result.to_dict())

    measurement_end = len(video_frames) - 1
    if point_result.terminal_frame is not None:
        measurement_end = min(
            measurement_end,
            point_result.terminal_frame + round(0.25 * video_fps),
        )
    speed_ball_positions = [
        dict(position) if frame_num <= measurement_end else {}
        for frame_num, position in enumerate(ball_mini_court_detections)
    ]
    speed_ball_heights = [
        height if frame_num <= measurement_end else None
        for frame_num, height in enumerate(ball_heights)
    ]
    speed_tracker = SpeedTracker(video_fps, measurement_window_seconds=0.1)
    speed_series = speed_tracker.calculate(
        player_mini_court_detections,
        speed_ball_positions,
        speed_ball_heights,
        mini_court,
        observed_ball_frames={
            frame for frame in observed_ball_frames if frame <= measurement_end
        },
        interpolated_ball_frames={
            frame
            for frame in short_interpolated_ball_frames
            if frame <= measurement_end
        },
        ball_visibility_states=ball_visibility_states,
    )
    for frame_num in range(measurement_end + 1, len(display_ball_detections)):
        display_ball_detections[frame_num] = {}
        ball_visibility_states[frame_num] = "unknown"

    predicted_high_arc_frames = {
        frame_num
        for frame_num, state in enumerate(ball_visibility_states)
        if state == "out_of_frame"
    }
    display_ball_mini_court_detections = [
        dict(position)
        if ball_visibility_states[frame_num]
        in ("observed", "interpolated", "out_of_frame")
        else {}
        for frame_num, position in enumerate(ball_mini_court_detections)
    ]

    player_stats = calculate_player_stats(
        len(video_frames),
        ball_events,
        speed_series,
        ball_visibility_states=ball_visibility_states,
    )
    output_video_frames = player_tracker.draw_bboxes(
        video_frames, player_detections
    )
    output_video_frames = ball_tracker.draw_bboxes(
        output_video_frames,
        display_ball_detections,
        observed_frame_numbers=observed_ball_frames,
        short_interpolated_frame_numbers=short_interpolated_ball_frames,
        predicted_high_arc_frame_numbers=predicted_high_arc_frames,
        visibility_states=ball_visibility_states,
        projected_image_centers=projected_ball_image_centers,
    )
    output_video_frames = court_line_detector.draw_keypoints_on_video(
        output_video_frames, court_keypoints_per_frame
    )
    output_video_frames = draw_player_stats(
        output_video_frames,
        player_stats,
        point_result=point_result,
    )
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(
        output_video_frames, player_mini_court_detections
    )
    output_video_frames = mini_court.draw_points_on_mini_court(
        output_video_frames,
        display_ball_mini_court_detections,
        color=(0, 255, 255),
    )
    output_video_frames = trajectory_reconstructor.draw_debug_overlay(
        output_video_frames,
        display_ball_mini_court_detections,
        ball_heights,
        ball_events,
    )
    output_video_frames = trajectory_reconstructor.draw_bounce_markers(
        output_video_frames,
        ball_events,
        court_keypoints_per_frame,
        mini_court,
    )

    for frame_num, frame in enumerate(output_video_frames):
        cv2.putText(
            frame,
            f"Frame: {frame_num}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    output_path = BASE_DIR / "output_videos" / "output_video.avi"
    result_path = BASE_DIR / "output_videos" / "point_result.json"
    stats_path = BASE_DIR / "output_videos" / "tracking_stats.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as result_file:
        json.dump(point_result.to_dict(), result_file, indent=2)
    player_stats.to_csv(stats_path, index=False)
    save_video(
        output_video_frames,
        str(output_path),
        fps=video_fps,
    )
    if open_output:
        os.startfile(output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Track one complete tennis point and classify its outcome."
    )
    parser.add_argument("video", nargs="?", default=None)
    parser.add_argument("--event-model", default=None)
    parser.add_argument("--outcome-model", default=None)
    parser.add_argument(
        "--player-mode",
        choices=("auto", "standard", "wheelchair"),
        default="auto",
        help=(
            "Player detector to use. Auto conservatively distinguishes "
            "wheelchair footage from standard tennis."
        ),
    )
    parser.add_argument(
        "--wheelchair-model",
        default=None,
        help="Custom wheelchair-player weights (default: models/wheelchair_best.pt).",
    )
    parser.add_argument(
        "--bounce-model",
        default=None,
        help=(
            "Exact-frame tennis bounce checkpoint "
            "(default: models/tennis_bounce_e2espot.pt)."
        ),
    )
    parser.add_argument(
        "--no-bounce-model",
        action="store_true",
        help="Disable learned bounce spotting and use trajectory rules only.",
    )
    parser.add_argument(
        "--bounce-confidence",
        type=float,
        default=0.35,
        help="Minimum learned bounce probability (default: 0.35).",
    )
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    main(
        cli_args.video,
        open_output=not cli_args.no_open,
        event_model_path=cli_args.event_model,
        outcome_model_path=cli_args.outcome_model,
        player_mode=cli_args.player_mode,
        wheelchair_model_path=cli_args.wheelchair_model,
        bounce_model_path=cli_args.bounce_model,
        use_bounce_model=not cli_args.no_bounce_model,
        bounce_confidence=cli_args.bounce_confidence,
    )
