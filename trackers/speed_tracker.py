from dataclasses import dataclass
import math

import numpy as np


@dataclass
class SpeedSeries:
    player_speeds_kmh: dict
    ball_speeds_kmh: list
    ball_speed_quality: list
    window_frames: int


class SpeedTracker:
    """Calculate causal rolling speeds at video-frame resolution."""

    def __init__(self, fps, measurement_window_seconds=0.1):
        if fps <= 0:
            raise ValueError("Video FPS must be greater than zero")
        if measurement_window_seconds <= 0:
            raise ValueError("Measurement window must be greater than zero")
        self.fps = float(fps)
        self.measurement_window_seconds = float(measurement_window_seconds)
        self.window_frames = max(
            1,
            int(round(self.fps * self.measurement_window_seconds)),
        )

    def calculate(
        self,
        player_positions,
        ball_positions,
        ball_heights,
        mini_court,
        observed_ball_frames=None,
        interpolated_ball_frames=None,
        ball_visibility_states=None,
    ):
        frame_count = len(player_positions)
        if not (
            len(ball_positions) == len(ball_heights) == frame_count
        ):
            raise ValueError("All speed inputs must contain the same frames")
        if (
            ball_visibility_states is not None
            and len(ball_visibility_states) != frame_count
        ):
            raise ValueError("Ball visibility state is required for every frame")

        player_speeds = {
            player_id: self._player_speed_series(
                player_positions,
                player_id,
                mini_court,
            )
            for player_id in (1, 2)
        }
        ball_speeds, ball_quality = self._ball_speed_series(
            ball_positions,
            ball_heights,
            mini_court,
            set(observed_ball_frames or ()),
            set(interpolated_ball_frames or ()),
            ball_visibility_states,
        )
        return SpeedSeries(
            player_speeds_kmh=player_speeds,
            ball_speeds_kmh=ball_speeds,
            ball_speed_quality=ball_quality,
            window_frames=self.window_frames,
        )

    def _player_speed_series(self, positions, player_id, mini_court):
        coordinates = np.full((len(positions), 2), np.nan, dtype=float)
        for frame_num, frame_positions in enumerate(positions):
            point = frame_positions.get(player_id)
            if point is not None and np.isfinite(point).all():
                coordinates[frame_num] = point

        maximum_gap = max(1, round(0.20 * self.fps))
        coordinates = self._interpolate_short_gaps(coordinates, maximum_gap)
        coordinates = self._rolling_median_coordinates(
            coordinates,
            max(3, self.window_frames | 1),
        )

        speeds = [None] * len(positions)
        for frame_num in range(self.window_frames, len(positions)):
            previous = coordinates[frame_num - self.window_frames]
            current = coordinates[frame_num]
            if not (
                np.isfinite(previous).all() and np.isfinite(current).all()
            ):
                continue
            elapsed = self.window_frames / self.fps
            distance_pixels = float(np.linalg.norm(current - previous))
            speed = (
                mini_court.convert_pixels_to_meters(distance_pixels)
                / elapsed
                * 3.6
            )
            # Values above elite sprint speed are normally foot/keypoint jitter.
            if speed <= 45.0:
                speeds[frame_num] = speed
        return self._rolling_nanmedian(speeds, max(3, self.window_frames | 1))

    def _ball_speed_series(
        self,
        positions,
        heights,
        mini_court,
        observed_frames,
        interpolated_frames,
        visibility_states,
    ):
        coordinates = np.full((len(positions), 3), np.nan, dtype=float)
        quality_rank = np.zeros(len(positions), dtype=int)
        for frame_num, frame_positions in enumerate(positions):
            visibility = (
                None
                if visibility_states is None
                else visibility_states[frame_num]
            )
            if visibility in ("occluded", "unknown"):
                continue
            point = frame_positions.get(1)
            if point is None or not np.isfinite(point).all():
                continue
            coordinates[frame_num, 0] = mini_court.convert_pixels_to_meters(
                float(point[0])
            )
            coordinates[frame_num, 1] = mini_court.convert_pixels_to_meters(
                float(point[1])
            )
            height = heights[frame_num]
            coordinates[frame_num, 2] = (
                float(height) if height is not None else 0.0
            )
            if visibility == "observed" or frame_num in observed_frames:
                quality_rank[frame_num] = 4
            elif visibility == "interpolated" or frame_num in interpolated_frames:
                quality_rank[frame_num] = 3
            elif visibility == "out_of_frame":
                quality_rank[frame_num] = 2
            else:
                quality_rank[frame_num] = 1

        speeds = [None] * len(positions)
        qualities = ["unavailable"] * len(positions)
        rank_names = {
            1: "inferred",
            2: "out_of_frame_predicted",
            3: "interpolated",
            4: "observed",
        }
        for frame_num in range(self.window_frames, len(positions)):
            start = frame_num - self.window_frames
            segment = coordinates[start : frame_num + 1]
            if not np.isfinite(segment).all():
                continue

            # Sum the short 3-D path rather than measuring only its chord.
            distance_meters = float(
                np.linalg.norm(np.diff(segment, axis=0), axis=1).sum()
            )
            elapsed = self.window_frames / self.fps
            speed = distance_meters / elapsed * 3.6
            if speed > 280.0:
                continue
            speeds[frame_num] = speed
            window_quality = quality_rank[start : frame_num + 1]
            minimum_rank = int(np.min(window_quality))
            if minimum_rank > 0:
                qualities[frame_num] = rank_names[minimum_rank]

        speeds = self._rolling_nanmedian(
            speeds,
            max(3, self.window_frames | 1),
        )
        return speeds, qualities

    @staticmethod
    def _interpolate_short_gaps(coordinates, maximum_gap):
        output = coordinates.copy()
        valid_indices = np.flatnonzero(np.isfinite(output).all(axis=1))
        for start, end in zip(valid_indices, valid_indices[1:]):
            gap = int(end - start - 1)
            if gap <= 0 or gap > maximum_gap:
                continue
            for frame_num in range(start + 1, end):
                ratio = (frame_num - start) / (end - start)
                output[frame_num] = (
                    output[start] + ratio * (output[end] - output[start])
                )
        return output

    @staticmethod
    def _rolling_median_coordinates(coordinates, window):
        radius = window // 2
        output = coordinates.copy()
        for frame_num in range(len(coordinates)):
            if not np.isfinite(coordinates[frame_num]).all():
                continue
            start = max(0, frame_num - radius)
            end = min(len(coordinates), frame_num + radius + 1)
            if end - start < window:
                continue
            local = coordinates[start:end]
            if np.isfinite(local).all(axis=1).any():
                output[frame_num] = np.nanmedian(local, axis=0)
        return output

    @staticmethod
    def _rolling_nanmedian(values, window):
        radius = window // 2
        numeric = np.asarray(
            [np.nan if value is None else float(value) for value in values],
            dtype=float,
        )
        output = list(values)
        for frame_num, value in enumerate(numeric):
            if not math.isfinite(value):
                continue
            start = max(0, frame_num - radius)
            end = min(len(numeric), frame_num + radius + 1)
            local = numeric[start:end]
            if np.isfinite(local).any():
                output[frame_num] = float(np.nanmedian(local))
        return output
