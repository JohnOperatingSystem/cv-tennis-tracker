import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter


class BallTrajectoryReconstructor:
    """Reconstruct short ball flights between likely hits and bounces.

    The detector supplies image positions.  Motion changes and player proximity
    identify events; event positions then anchor a continuous trajectory in
    mini-court coordinates.  The reconstruction is deliberately conservative:
    an unbounded or unusually long section is left as originally projected.
    """

    def __init__(
        self,
        fps,
        frame_width=1280,
        frame_height=720,
        allowed_bounces=1,
    ):
        if fps <= 0:
            raise ValueError("Video FPS must be greater than zero")
        self.fps = float(fps)
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Video dimensions must be greater than zero")
        self.coordinate_scale = np.asarray(
            (1280.0 / frame_width, 720.0 / frame_height),
            dtype=float,
        )
        self.minimum_event_acceleration = 1.5
        self.gravity = 9.81
        self.allowed_bounces = int(allowed_bounces)
        if self.allowed_bounces < 1:
            raise ValueError("At least one bounce must be allowed")
        # Long lobs can remain airborne substantially longer than flat rally
        # balls. This is a safety bound, not an expected shot duration.
        self.maximum_flight_seconds = 4.0

    def reconstruct(
        self,
        ball_detections,
        player_detections,
        player_mini_court_detections,
        ball_mini_court_detections,
        mini_court,
        temporal_predictions=None,
        bounce_predictions=None,
    ):
        frame_count = len(ball_detections)
        if not (
            len(player_detections)
            == len(player_mini_court_detections)
            == len(ball_mini_court_detections)
            == frame_count
        ):
            raise ValueError("All trajectory inputs must contain the same frames")

        smoothed_measurements = self._smooth_mini_court_positions(
            ball_mini_court_detections
        )
        output_positions = [dict(frame) for frame in smoothed_measurements]
        ball_heights = [None] * frame_count
        candidates = self._find_motion_candidates(
            ball_detections,
            player_detections,
        )
        self._apply_temporal_support(candidates, temporal_predictions)
        self._apply_learned_bounce_support(
            candidates,
            bounce_predictions,
            ball_detections,
            player_detections,
        )
        hit_frames = self._find_hit_frames(
            candidates,
            ball_detections,
            player_detections,
            player_mini_court_detections,
            smoothed_measurements,
            mini_court,
        )
        events = self._build_events(
            hit_frames,
            candidates,
            ball_detections,
            player_detections,
            player_mini_court_detections,
            smoothed_measurements,
            ball_mini_court_detections,
            mini_court,
        )
        self._annotate_temporal_events(events, candidates)

        for start_event, end_event in zip(events, events[1:]):
            frame_gap = end_event["frame"] - start_event["frame"]
            duration = frame_gap / self.fps
            if (
                frame_gap <= 0
                or duration > self._maximum_segment_duration(
                    start_event,
                    end_event,
                )
            ):
                continue

            self._fit_flight_segment(
                output_positions,
                ball_heights,
                start_event,
                end_event,
            )

        self._fit_open_ended_tail(
            output_positions,
            ball_heights,
            events,
            player_mini_court_detections,
            mini_court,
        )

        return output_positions, ball_heights, events

    def _maximum_segment_duration(self, start_event, end_event):
        if start_event.get("type") == "bounce":
            # The post-bounce rise to a racket is short even on defensive
            # groundstrokes. Longer gaps imply a missed contact/event.
            return min(self.maximum_flight_seconds, 1.8)
        if end_event.get("type") == "bounce":
            return min(self.maximum_flight_seconds, 3.2)
        return min(self.maximum_flight_seconds, 2.8)

    def _apply_temporal_support(self, candidates, predictions):
        """Attach optional TenniSet-model evidence to geometric candidates.

        The learned model never creates a contact without a ball-motion peak.
        It only helps rank ambiguous peaks, keeping the physical trajectory as
        the primary source of truth.
        """
        if not predictions:
            return
        tolerance = max(2, round(0.16 * self.fps))
        usable = []
        for prediction in predictions:
            label = str(prediction.get("label", "")).upper()
            if len(label) != 3 or label[0] not in ("H", "S"):
                continue
            try:
                frame_num = int(prediction["frame"])
                confidence = float(prediction.get("confidence", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(confidence) or confidence <= 0:
                continue
            expected_player_id = {"F": 1, "N": 2}.get(label[1])
            if expected_player_id is None:
                continue
            usable.append(
                {
                    "frame": frame_num,
                    "label": label,
                    "confidence": min(1.0, confidence),
                    "expected_player_id": expected_player_id,
                }
            )

        for candidate in candidates:
            nearby = [
                prediction
                for prediction in usable
                if abs(prediction["frame"] - candidate["frame"]) <= tolerance
            ]
            if not nearby:
                continue
            best = max(
                nearby,
                key=lambda prediction: prediction["confidence"]
                - 0.04 * abs(prediction["frame"] - candidate["frame"]),
            )
            support = best["confidence"]
            if (
                candidate.get("player_id") is not None
                and candidate["player_id"] != best["expected_player_id"]
            ):
                support *= 0.25
            candidate["temporal_confidence"] = support
            candidate["temporal_label"] = best["label"]
            candidate["temporal_frame"] = best["frame"]

    @staticmethod
    def _annotate_temporal_events(events, candidates):
        supported = {
            candidate["frame"]: candidate
            for candidate in candidates
            if candidate.get("temporal_label")
        }
        for event in events:
            if event.get("type") != "hit":
                continue
            candidate = supported.get(event.get("frame"))
            if candidate is None:
                continue
            event["temporal_label"] = candidate["temporal_label"]
            event["temporal_confidence"] = round(
                float(candidate.get("temporal_confidence", 0.0)), 4
            )

    def find_motion_candidates(
        self,
        ball_detections,
        player_detections,
        bounce_predictions=None,
    ):
        """Expose unclassified impact candidates for downstream rules logic.

        These candidates retain possible out and net impacts that the display
        reconstruction intentionally rejects when fitting legal rally arcs.
        """
        candidates = self._find_motion_candidates(
            ball_detections,
            player_detections,
        )
        self._apply_learned_bounce_support(
            candidates,
            bounce_predictions,
            ball_detections,
            player_detections,
        )
        return candidates

    def _apply_learned_bounce_support(
        self,
        candidates,
        predictions,
        ball_detections,
        player_detections,
    ):
        """Merge exact-frame model evidence into high-recall motion events.

        A learned observation may create a candidate even when the ball tracker
        misses the acceleration peak. Court side, landing position and rally
        order are still checked later, preventing a visual false positive from
        becoming a rules event on its own.
        """
        if not predictions:
            return
        tolerance = max(2, round(0.14 * self.fps))
        for prediction in predictions:
            label = str(prediction.get("label", "")).lower()
            if "bounce" not in label:
                continue
            try:
                learned_frame = int(prediction["frame"])
                confidence = float(prediction.get("confidence", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not 0 <= learned_frame < len(ball_detections)
                or not np.isfinite(confidence)
                or confidence < 0.25
            ):
                continue

            nearby = [
                candidate
                for candidate in candidates
                if abs(candidate["frame"] - learned_frame) <= tolerance
            ]
            if nearby:
                candidate = min(
                    nearby,
                    key=lambda item: (
                        abs(item["frame"] - learned_frame),
                        -item["acceleration"],
                    ),
                )
            else:
                candidate = self._make_learned_bounce_candidate(
                    learned_frame,
                    ball_detections,
                    player_detections,
                    tolerance,
                )
                if candidate is None:
                    continue
                candidates.append(candidate)
            if confidence > candidate.get("learned_bounce_confidence", 0.0):
                candidate["learned_bounce_confidence"] = confidence
                candidate["learned_bounce_frame"] = learned_frame
                candidate["learned_bounce_label"] = label
        candidates.sort(key=lambda item: item["frame"])

    def _make_learned_bounce_candidate(
        self,
        learned_frame,
        ball_detections,
        player_detections,
        tolerance,
    ):
        available = [
            frame_num
            for frame_num in range(
                max(0, learned_frame - tolerance),
                min(len(ball_detections), learned_frame + tolerance + 1),
            )
            if ball_detections[frame_num].get(1) is not None
        ]
        if not available:
            return None
        frame_num = min(available, key=lambda value: abs(value - learned_frame))
        center = self._bbox_center(ball_detections[frame_num][1])
        nearest_player_id = self._nearest_player_id(
            center,
            player_detections[frame_num],
        )
        player_distance = self._distance_to_players(
            center,
            player_detections[frame_num],
        )
        if nearest_player_id is None:
            distance_ratio = float("inf")
        else:
            player_bbox = player_detections[frame_num][nearest_player_id]
            player_height = max(1.0, player_bbox[3] - player_bbox[1])
            distance_ratio = player_distance / player_height

        before = self._nearest_ball_center(
            ball_detections,
            frame_num - 1,
            direction=-1,
            limit=tolerance + 2,
        )
        after = self._nearest_ball_center(
            ball_detections,
            frame_num + 1,
            direction=1,
            limit=tolerance + 2,
        )
        vertical_reversal = bool(
            before is not None
            and after is not None
            and center[1] > before[1]
            and center[1] > after[1]
        )
        acceleration = 0.0
        if before is not None and after is not None:
            acceleration = float(
                np.linalg.norm((after - center) - (center - before))
            )
        return {
            "frame": frame_num,
            "acceleration": acceleration,
            "contact_acceleration": acceleration,
            "player_distance": player_distance,
            "player_distance_ratio": distance_ratio,
            "player_id": nearest_player_id,
            "vertical_reversal": vertical_reversal,
        }

    @staticmethod
    def _nearest_ball_center(
        ball_detections,
        start_frame,
        direction,
        limit,
    ):
        for offset in range(limit + 1):
            frame_num = start_frame + direction * offset
            if not 0 <= frame_num < len(ball_detections):
                break
            bbox = ball_detections[frame_num].get(1)
            if bbox is not None:
                return BallTrajectoryReconstructor._bbox_center(bbox)
        return None

    def _smooth_mini_court_positions(self, positions):
        """Remove isolated projection jumps and smooth short measured runs."""
        coordinates = np.full((len(positions), 2), np.nan, dtype=float)
        for frame_num, frame_positions in enumerate(positions):
            position = frame_positions.get(1)
            if position is not None and np.isfinite(position).all():
                coordinates[frame_num] = position

        # Fill only short, bounded gaps with believable endpoint motion.  Long
        # gaps remain absent rather than drawing an invented flight.
        valid_indices = np.flatnonzero(np.isfinite(coordinates).all(axis=1))
        maximum_gap = max(4, round(0.8 * self.fps))
        for start, end in zip(valid_indices, valid_indices[1:]):
            gap = int(end - start - 1)
            if gap <= 0 or gap > maximum_gap:
                continue
            average_step = np.linalg.norm(coordinates[end] - coordinates[start]) / (
                end - start
            )
            if average_step > 12.0:
                continue
            for frame_num in range(start + 1, end):
                ratio = (frame_num - start) / (end - start)
                coordinates[frame_num] = (
                    coordinates[start]
                    + ratio * (coordinates[end] - coordinates[start])
                )

        for run_start, run_end in self._valid_runs(coordinates):
            values = coordinates[run_start : run_end + 1].copy()
            if len(values) < 5:
                continue

            # A rolling median is used only to replace isolated large jumps;
            # normal fast linear ball motion is retained.
            padded = np.pad(values, ((2, 2), (0, 0)), mode="edge")
            local_median = np.asarray(
                [np.median(padded[index : index + 5], axis=0) for index in range(len(values))]
            )
            outliers = np.linalg.norm(values - local_median, axis=1) > 14.0
            values[outliers] = local_median[outliers]

            window = min(9, len(values) if len(values) % 2 else len(values) - 1)
            if window >= 5:
                values = savgol_filter(
                    values,
                    window_length=window,
                    polyorder=2,
                    axis=0,
                )
            coordinates[run_start : run_end + 1] = values

        return [
            {1: (float(point[0]), float(point[1]))}
            if np.isfinite(point).all()
            else {}
            for point in coordinates
        ]

    def _find_motion_candidates(self, ball_detections, player_detections):
        centers = np.full((len(ball_detections), 2), np.nan, dtype=float)
        for frame_num, detection in enumerate(ball_detections):
            bbox = detection.get(1)
            if bbox:
                centers[frame_num] = self._bbox_center(bbox)

        normalized_centers = centers * self.coordinate_scale

        candidates = []
        for run_start, run_end in self._valid_runs(normalized_centers):
            run_length = run_end - run_start + 1
            if run_length < 7:
                continue

            window = min(11, run_length if run_length % 2 else run_length - 1)
            smoothed = savgol_filter(
                normalized_centers[run_start : run_end + 1],
                window_length=window,
                polyorder=2,
                axis=0,
            )
            velocity = np.gradient(smoothed, axis=0)
            acceleration = np.linalg.norm(np.gradient(velocity, axis=0), axis=1)
            peak_indices, _ = find_peaks(
                acceleration,
                distance=max(4, round(0.28 * self.fps)),
                # Far-court contacts move only a few image pixels. Keep low
                # peaks here and reject them later using racket proximity and
                # rally sequencing rather than an image-scale-only cutoff.
                prominence=0.25,
            )

            for local_index in peak_indices:
                frame_num = run_start + int(local_index)
                velocity_before = velocity[max(0, local_index - 3) : local_index, 1]
                velocity_after = velocity[local_index + 1 : local_index + 4, 1]
                dy_before = float(np.mean(velocity_before))
                dy_after = float(np.mean(velocity_after))
                nearest_player_id = self._nearest_player_id(
                    centers[frame_num], player_detections[frame_num]
                )
                player_distance = self._distance_to_players(
                    centers[frame_num],
                    player_detections[frame_num],
                )
                if nearest_player_id is None:
                    player_distance_ratio = float("inf")
                else:
                    nearest_bbox = player_detections[frame_num][nearest_player_id]
                    player_height = max(1.0, nearest_bbox[3] - nearest_bbox[1])
                    player_distance_ratio = player_distance / player_height
                contact_acceleration = float(acceleration[local_index])
                if nearest_player_id is not None:
                    # Normalize contact evidence to an 80-pixel-tall player so
                    # the same racket action is not suppressed at the far end.
                    contact_acceleration *= float(
                        np.clip(80.0 / player_height, 0.75, 3.5)
                    )
                candidates.append(
                    {
                        "frame": frame_num,
                        "acceleration": float(acceleration[local_index]),
                        "contact_acceleration": contact_acceleration,
                        "player_distance": player_distance,
                        "player_distance_ratio": player_distance_ratio,
                        "player_id": nearest_player_id,
                        # Image y grows downwards.  Down followed by up is the
                        # strongest visual bounce cue in a fixed camera view.
                        "vertical_reversal": dy_before > 0 and dy_after < 0,
                    }
                )
        return candidates

    @staticmethod
    def _valid_runs(centers):
        valid = np.isfinite(centers).all(axis=1)
        start = None
        for index, is_valid in enumerate(valid):
            if is_valid and start is None:
                start = index
            if start is not None and (not is_valid or index == len(valid) - 1):
                end = index if is_valid else index - 1
                yield start, end
                start = None

    def _find_hit_frames(
        self,
        candidates,
        ball_detections,
        player_detections,
        player_mini_positions,
        ball_mini_positions,
        mini_court,
    ):
        """Select the most coherent sequence of racket contacts.

        Individual acceleration peaks are ambiguous: a hit, bounce, detector
        jump, and camera motion can all create one. Dynamic programming scores
        the complete rally sequence, allowing either a direct hit-to-hit
        flight (volley/overhead) or a flight containing a court-valid bounce.
        No particular shot duration or bounce timing is assumed.
        """
        hit_candidates = sorted(
            (
                candidate
                for candidate in candidates
                if (
                    candidate.get(
                        "contact_acceleration", candidate["acceleration"]
                    )
                    >= self.minimum_event_acceleration
                    or (
                        candidate["acceleration"] >= 0.55
                        and candidate["player_distance_ratio"] <= 0.38
                    )
                    or candidate.get("temporal_confidence", 0.0) >= 0.72
                )
                and (
                    candidate["player_distance_ratio"] <= 1.15
                    or (
                        candidate.get("temporal_confidence", 0.0) >= 0.80
                        and candidate["player_distance_ratio"] <= 1.35
                    )
                )
                # A high-confidence bounce model observation should not be
                # promoted to a racket contact merely because the same motion
                # discontinuity is within the broad hit search radius. Keep a
                # genuinely close half-volley/contact eligible.
                and not (
                    candidate.get("learned_bounce_confidence", 0.0) >= 0.70
                    and candidate.get("temporal_confidence", 0.0) < 0.72
                    and candidate["player_distance_ratio"] > 0.45
                )
                and candidate["player_id"] is not None
            ),
            key=lambda item: item["frame"],
        )
        if not hit_candidates:
            return []

        sequence_scores = [self._hit_candidate_score(item) for item in hit_candidates]
        predecessors = [None] * len(hit_candidates)
        minimum_contact_gap = max(3, round(0.14 * self.fps))
        # A long rally must survive an occasional missed racket contact. The
        # physical reconstructor still refuses to fit a flight longer than
        # ``maximum_flight_seconds``; this wider limit only joins independently
        # observed contact sequences on either side of a detector gap.
        maximum_contact_gap = round(12.0 * self.fps)

        for current_index, current in enumerate(hit_candidates):
            current_score = self._hit_candidate_score(current)
            for previous_index in range(current_index):
                previous = hit_candidates[previous_index]
                frame_gap = current["frame"] - previous["frame"]
                if not minimum_contact_gap <= frame_gap <= maximum_contact_gap:
                    continue

                transition_score = self._hit_transition_score(
                    previous,
                    current,
                    candidates,
                    ball_detections,
                    player_detections,
                    player_mini_positions,
                    ball_mini_positions,
                    mini_court,
                )
                proposed_score = (
                    sequence_scores[previous_index]
                    + current_score
                    + transition_score
                )
                if proposed_score > sequence_scores[current_index]:
                    sequence_scores[current_index] = proposed_score
                    predecessors[current_index] = previous_index

        # Prefer a coherent sequence over a single exceptionally large
        # detector discontinuity. A small coverage reward avoids dropping a
        # valid final contact when two paths otherwise score almost equally.
        final_index = max(
            range(len(hit_candidates)),
            key=lambda index: sequence_scores[index]
            + 0.001 * hit_candidates[index]["frame"],
        )
        selected = []
        while final_index is not None:
            selected.append(hit_candidates[final_index]["frame"])
            final_index = predecessors[final_index]
        return list(reversed(selected))

    def _hit_candidate_score(self, candidate):
        acceleration_strength = np.log1p(
            candidate.get("contact_acceleration", candidate["acceleration"])
            / self.minimum_event_acceleration
        )
        proximity = max(
            0.0,
            1.0 - candidate["player_distance_ratio"] / 1.15,
        )
        # Log compression prevents one detector jump from dominating an
        # otherwise coherent sequence of contacts.
        temporal_support = float(candidate.get("temporal_confidence", 0.0))
        return (
            1.25 * acceleration_strength
            + 2.5 * proximity
            + 2.0 * temporal_support
            - 2.5
        )

    def _hit_transition_score(
        self,
        previous,
        current,
        candidates,
        ball_detections,
        player_detections,
        player_mini_positions,
        ball_mini_positions,
        mini_court,
    ):
        frame_gap = current["frame"] - previous["frame"]
        flight_seconds = frame_gap / self.fps
        score = 0.0

        if flight_seconds > self.maximum_flight_seconds:
            # At least one contact was missed. Do not apply alternation or
            # bounce rules across unknown play, but retain the earlier rally
            # as context instead of restarting and discarding it entirely.
            return -min(2.5, 0.35 * (flight_seconds - self.maximum_flight_seconds))

        # In singles, legal consecutive contacts normally alternate sides.
        # Keep this a penalty rather than a hard rule so short-lived tracker ID
        # swaps do not destroy the rally.
        if previous["player_id"] == current["player_id"]:
            score -= 6.0
        else:
            score += 2.0

        if flight_seconds < 0.30:
            score -= 2.0 * (0.30 - flight_seconds) / 0.30
        elif flight_seconds > 3.0:
            score -= flight_seconds - 3.0

        bounce = self._best_bounce_candidate(
            previous["frame"],
            current["frame"],
            current["player_id"],
            candidates,
            ball_detections,
            player_detections,
            player_mini_positions,
            ball_mini_positions,
            mini_court,
        )
        if bounce is not None:
            score += min(5.0, self._bounce_candidate_score(bounce))
        return score

    def _best_bounce_candidate(
        self,
        start_frame,
        end_frame,
        receiver_id,
        candidates,
        ball_detections,
        player_detections,
        player_mini_positions,
        ball_mini_positions,
        mini_court,
        strict_contact_separation=True,
    ):
        plausible = self._bounce_candidates(
            start_frame,
            end_frame,
            receiver_id,
            candidates,
            ball_detections,
            player_detections,
            player_mini_positions,
            ball_mini_positions,
            mini_court,
            strict_contact_separation=strict_contact_separation,
        )
        if not plausible:
            return None
        return max(plausible, key=self._bounce_candidate_score)

    def _bounce_candidates(
        self,
        start_frame,
        end_frame,
        receiver_id,
        candidates,
        ball_detections,
        player_detections,
        player_mini_positions,
        ball_mini_positions,
        mini_court,
        strict_contact_separation=True,
    ):
        """Return distinct observed landings between two racket contacts."""
        margin = max(2, round(0.08 * self.fps))
        if end_frame - start_frame <= 2 * margin:
            return []

        receiver_position = player_mini_positions[end_frame].get(receiver_id)
        if receiver_position is None:
            return []
        net_y = (
            mini_court.drawing_key_points[1]
            + mini_court.drawing_key_points[5]
        ) / 2

        plausible = []
        for candidate in candidates:
            frame_num = candidate["frame"]
            learned_confidence = candidate.get(
                "learned_bounce_confidence", 0.0
            )
            event_frame = int(
                candidate.get("learned_bounce_frame", frame_num)
            )
            if not (
                start_frame + margin <= event_frame <= end_frame - margin
                and (
                    candidate["acceleration"]
                    >= self.minimum_event_acceleration
                    or learned_confidence >= 0.25
                )
            ):
                continue

            if (
                learned_confidence >= 0.50
                and ball_detections[event_frame].get(1) is not None
            ):
                impact_frame = event_frame
            else:
                impact_frame = self._refine_bounce_frame(
                    frame_num,
                    candidate["vertical_reversal"],
                    ball_detections,
                )
            if not start_frame + margin <= impact_frame <= end_frame - margin:
                continue
            position = ball_mini_positions[impact_frame].get(1)
            ball_bbox = ball_detections[impact_frame].get(1)
            players = player_detections[impact_frame]
            if position is None or ball_bbox is None:
                continue

            ball_center = self._bbox_center(ball_bbox)
            nearest_player_id = self._nearest_player_id(ball_center, players)
            if nearest_player_id is None:
                impact_distance_ratio = float("inf")
            else:
                player_bbox = players[nearest_player_id]
                player_height = max(1.0, player_bbox[3] - player_bbox[1])
                impact_distance_ratio = (
                    self._point_to_bbox_distance(ball_center, player_bbox)
                    / player_height
                )

            evaluated = dict(candidate)
            evaluated["bounce_frame"] = impact_frame
            evaluated["bounce_position"] = position
            evaluated["bounce_player_distance_ratio"] = impact_distance_ratio
            if (
                self._is_plausible_bounce_candidate(
                    evaluated,
                    end_frame=end_frame,
                    strict_contact_separation=strict_contact_separation,
                )
                and (
                    self._inside_court(position, mini_court)
                    or self.allowed_bounces >= 2
                )
                and self._on_receiver_side(position, receiver_position, net_y)
            ):
                plausible.append(evaluated)

        plausible.sort(key=lambda item: item["bounce_frame"])
        deduplicated = []
        dedupe_frames = max(2, round(0.16 * self.fps))
        for candidate in plausible:
            if (
                deduplicated
                and candidate["bounce_frame"]
                - deduplicated[-1]["bounce_frame"]
                <= dedupe_frames
            ):
                if self._bounce_candidate_score(
                    candidate
                ) > self._bounce_candidate_score(deduplicated[-1]):
                    deduplicated[-1] = candidate
                continue
            deduplicated.append(candidate)
        return deduplicated

    def _select_distinct_bounces(self, candidates, limit):
        """Select strong, time-separated bounces and return them in order."""
        if limit <= 0 or not candidates:
            return []
        separation = max(3, round(0.22 * self.fps))
        selected = []
        for candidate in sorted(
            candidates,
            key=self._bounce_candidate_score,
            reverse=True,
        ):
            if all(
                abs(candidate["bounce_frame"] - item["bounce_frame"])
                >= separation
                for item in selected
            ):
                selected.append(candidate)
                if len(selected) >= limit:
                    break
        return sorted(selected, key=lambda item: item["bounce_frame"])

    def _bounce_candidate_score(self, candidate):
        acceleration_strength = np.log1p(
            candidate["acceleration"] / self.minimum_event_acceleration
        )
        # Passing the court/receiver-side/contact-separation checks is itself
        # meaningful evidence of an impact. The reversal is an additional cue,
        # not a requirement (perspective and sparse detections can hide it).
        learned_support = 6.0 * float(
            candidate.get("learned_bounce_confidence", 0.0)
        )
        return 2.0 + acceleration_strength + learned_support + (
            3.0 if candidate["vertical_reversal"] else 0.0
        )

    def _is_plausible_bounce_candidate(
        self,
        candidate,
        end_frame=None,
        strict_contact_separation=True,
    ):
        """Keep racket contacts out while preserving true half-volleys."""
        distance_ratio = candidate.get(
            "bounce_player_distance_ratio",
            candidate["player_distance_ratio"],
        )
        if distance_ratio < 0.10:
            return False

        # A court impact this close to a player is plausible only when the
        # racket contact follows almost immediately (a half-volley). Otherwise
        # proximity makes the acceleration peak much more likely to be the
        # contact itself.
        impact_frame = candidate.get("bounce_frame", candidate["frame"])
        contact_delay = (
            None if end_frame is None else end_frame - impact_frame
        )
        immediate_contact = (
            contact_delay is not None
            and candidate["vertical_reversal"]
            and 1 <= contact_delay <= max(3, round(0.24 * self.fps))
        )
        if immediate_contact:
            return True

        if candidate.get("learned_bounce_confidence", 0.0) >= 0.70:
            return True

        if not strict_contact_separation:
            return True
        if self._hit_candidate_score(candidate) >= 1.4:
            return False
        return distance_ratio >= 0.55

    def _should_infer_bounce(
        self,
        first_hit,
        second_hit,
        receiver_position,
        mini_court,
    ):
        """Infer only when court geometry indicates a grounded reception."""
        flight_seconds = (
            second_hit["frame"] - first_hit["frame"]
        ) / self.fps
        if flight_seconds < 0.45:
            return False

        net_y = (
            mini_court.drawing_key_points[1]
            + mini_court.drawing_key_points[5]
        ) / 2
        receiver_y = float(receiver_position[1])
        side_baseline_y = (
            mini_court.court_start_y
            if receiver_y < net_y
            else mini_court.drawing_key_points[5]
        )
        half_court_length = max(1.0, abs(side_baseline_y - net_y))
        receiver_depth = abs(receiver_y - net_y) / half_court_length
        serve_like_contact = (
            first_hit.get("height", 1.0) >= 1.8
            and receiver_depth >= 0.35
        )
        # Player/ball depth is considerably more reliable than a monocular
        # contact-height estimate for a five-pixel ball. A receiver at or
        # behind the baseline is overwhelmingly playing the ball after a
        # bounce, even when projection noise makes the contact look high.
        deep_baseline_reception = receiver_depth >= 0.65
        grounded_reception = (
            receiver_depth >= 0.35
            and second_hit.get("height", 1.0) <= 1.6
        )
        return (
            serve_like_contact
            or deep_baseline_reception
            or grounded_reception
        )

    def _infer_bounce_anchor(
        self,
        first_hit,
        second_hit,
        receiver_position,
        mini_court,
        typical_contact_delay_seconds=None,
        timing_cue_frame=None,
    ):
        """Place a missing landing from court geometry, not fixed timing."""
        start = np.asarray(first_hit["position"], dtype=float)
        receiver = np.asarray(receiver_position, dtype=float)
        net_y = (
            mini_court.drawing_key_points[1]
            + mini_court.drawing_key_points[5]
        ) / 2

        # Land in front of the receiver (toward the net), while retaining the
        # incoming shot's lateral direction. This adapts to deep balls, drop
        # shots, and camera perspective through mini-court coordinates.
        landing = start + 0.82 * (receiver - start)
        landing[1] = receiver[1] + 0.24 * (net_y - receiver[1])
        landing = self._clamp_to_court(landing, mini_court)

        frame_gap = second_hit["frame"] - first_hit["frame"]
        contact_margin = max(2, round(0.10 * self.fps))
        if timing_cue_frame is not None:
            # A motion discontinuity can still locate the impact in image
            # time even when height makes its homography position invalid.
            bounce_frame = timing_cue_frame
        elif typical_contact_delay_seconds is not None:
            # The ball's reappearance/contact phase is much more stable than
            # its hidden high arc. Use observed bounce-to-contact timing from
            # the video to work backwards from the receiving hit.
            contact_delay = round(typical_contact_delay_seconds * self.fps)
            bounce_frame = second_hit["frame"] - contact_delay
        else:
            total_distance = max(
                1e-6, float(np.linalg.norm(receiver - start))
            )
            distance_ratio = (
                float(np.linalg.norm(landing - start)) / total_distance
            )
            time_ratio = float(np.clip(distance_ratio, 0.35, 0.75))
            bounce_frame = round(
                first_hit["frame"] + time_ratio * frame_gap
            )
        horizontal_distance_meters = mini_court.convert_pixels_to_meters(
            float(np.linalg.norm(landing - start))
        )
        maximum_horizontal_speed_mps = 250.0 / 3.6
        physics_minimum_frames = int(np.ceil(
            horizontal_distance_meters
            / maximum_horizontal_speed_mps
            * self.fps
        ))
        takeoff_margin = max(
            round(0.18 * self.fps),
            physics_minimum_frames,
        )
        earliest_bounce = first_hit["frame"] + takeoff_margin
        latest_bounce = second_hit["frame"] - contact_margin
        if earliest_bounce >= latest_bounce:
            # No physically credible bounce fits between these contacts. Treat
            # the segment as a volley instead of manufacturing an extreme arc.
            return None, None
        bounce_frame = int(np.clip(
            bounce_frame,
            earliest_bounce,
            latest_bounce,
        ))
        return bounce_frame, tuple(landing)

    def _find_inferred_bounce_timing_cue(
        self,
        first_hit,
        second_hit,
        candidates,
        ball_detections,
        typical_contact_delay_seconds=None,
    ):
        """Use image motion to time a landing rejected by court projection."""
        start_frame = first_hit["frame"]
        end_frame = second_hit["frame"]
        margin = max(2, round(0.08 * self.fps))
        expected_frame = None
        timing_tolerance = None
        if typical_contact_delay_seconds is not None:
            expected_delay = round(
                typical_contact_delay_seconds * self.fps
            )
            expected_frame = end_frame - expected_delay
            timing_tolerance = max(
                round(0.20 * self.fps),
                round(0.35 * expected_delay),
            )
        timing_candidates = []
        for candidate in candidates:
            if not (
                start_frame + margin
                <= candidate["frame"]
                <= end_frame - margin
                and candidate["acceleration"]
                >= self.minimum_event_acceleration
            ):
                continue
            impact_frame = self._refine_bounce_frame(
                candidate["frame"],
                candidate["vertical_reversal"],
                ball_detections,
            )
            if not start_frame + margin <= impact_frame <= end_frame - margin:
                continue
            if (
                expected_frame is not None
                and abs(impact_frame - expected_frame) > timing_tolerance
            ):
                continue
            timing_candidates.append(
                (
                    self._bounce_candidate_score(candidate)
                    + (
                        0.20 * (impact_frame - expected_frame)
                        if expected_frame is not None
                        else 0.0
                    ),
                    impact_frame,
                )
            )

        if not timing_candidates:
            return None
        return max(timing_candidates, key=lambda item: item[0])[1]

    def _build_events(
        self,
        hit_frames,
        candidates,
        ball_detections,
        player_detections,
        player_mini_positions,
        ball_mini_positions,
        raw_ball_mini_positions,
        mini_court,
    ):
        hits = []
        for frame_num in hit_frames:
            hit = self._make_hit_event(
                frame_num,
                ball_detections,
                player_detections,
                player_mini_positions,
                ball_mini_positions,
            )
            if hit is not None:
                hits.append(hit)

        hits = self._infer_missing_gap_hits(
            hits,
            ball_detections,
            player_detections,
            player_mini_positions,
            ball_mini_positions,
        )

        events = list(hits)
        net_y = (
            mini_court.drawing_key_points[1]
            + mini_court.drawing_key_points[5]
        ) / 2

        # First collect every observed landing. Their bounce-to-contact delays
        # provide a video-specific timing prior for shots that disappear above
        # the frame and reappear only shortly before the receiving contact.
        flight_segments = []
        observed_contact_delays = []
        for first_hit, second_hit in zip(hits, hits[1:]):
            receiver_position = player_mini_positions[second_hit["frame"]].get(
                second_hit.get("player_id")
            )
            if receiver_position is None:
                receiver_position = second_hit["position"]

            observed_bounces = self._select_distinct_bounces(
                self._bounce_candidates(
                    first_hit["frame"],
                    second_hit["frame"],
                    second_hit.get("player_id"),
                    candidates,
                    ball_detections,
                    player_detections,
                    player_mini_positions,
                    ball_mini_positions,
                    mini_court,
                    strict_contact_separation=False,
                ),
                self.allowed_bounces,
            )
            flight_segments.append(
                (
                    first_hit,
                    second_hit,
                    receiver_position,
                    observed_bounces,
                )
            )
            if observed_bounces:
                contact_delay_seconds = (
                    second_hit["frame"]
                    - observed_bounces[-1]["bounce_frame"]
                ) / self.fps
                if 0.12 <= contact_delay_seconds <= 1.5:
                    observed_contact_delays.append(contact_delay_seconds)

        typical_contact_delay_seconds = (
            float(np.median(observed_contact_delays))
            if observed_contact_delays
            else None
        )

        for (
            first_hit,
            second_hit,
            receiver_position,
            observed_bounces,
        ) in flight_segments:

            if observed_bounces:
                for bounce_number, best in enumerate(
                    observed_bounces,
                    start=1,
                ):
                    bounce_frame = best["bounce_frame"]
                    refined_position = raw_ball_mini_positions[
                        bounce_frame
                    ].get(1)
                    allow_outside = (
                        self.allowed_bounces >= 2 and bounce_number >= 2
                    )
                    if (
                        refined_position is None
                        or (
                            not allow_outside
                            and not self._inside_court(
                                refined_position,
                                mini_court,
                            )
                        )
                        or not self._on_receiver_side(
                            refined_position,
                            receiver_position,
                            net_y,
                        )
                    ):
                        refined_position = best["bounce_position"]
                    events.append(
                        {
                            "frame": bounce_frame,
                            "type": "bounce",
                            "position": tuple(refined_position),
                            "height": 0.0,
                            "inferred": False,
                            "bounce_number": bounce_number,
                            "receiver_id": second_hit.get("player_id"),
                            "receiver_position": tuple(receiver_position),
                            "learned_confidence": best.get(
                                "learned_bounce_confidence"
                            ),
                            "event_source": (
                                "e2e_spot+trajectory"
                                if best.get("learned_bounce_confidence")
                                else "trajectory"
                            ),
                        }
                    )
                continue

            if self._should_infer_bounce(
                first_hit,
                second_hit,
                receiver_position,
                mini_court,
            ):
                # Detector gaps are common for a five-pixel ball.  When a full
                # grounded flight is strongly implied by receiver depth (and,
                # away from the baseline, a low contact), retain a
                # conservative inferred landing.
                timing_cue_frame = self._find_inferred_bounce_timing_cue(
                    first_hit,
                    second_hit,
                    candidates,
                    ball_detections,
                    typical_contact_delay_seconds,
                )
                bounce_frame, bounce_position = self._infer_bounce_anchor(
                    first_hit,
                    second_hit,
                    receiver_position,
                    mini_court,
                    typical_contact_delay_seconds,
                    timing_cue_frame,
                )
                if bounce_frame is None:
                    continue
            else:
                # A bounce is not mandatory: volleys, swinging volleys and
                # overheads may travel directly from one racket to the next.
                continue

            events.append(
                {
                    "frame": bounce_frame,
                    "type": "bounce",
                    "position": tuple(bounce_position),
                    "height": 0.0,
                    "inferred": True,
                    "bounce_number": 1,
                    "receiver_id": second_hit.get("player_id"),
                    "receiver_position": tuple(receiver_position),
                }
            )

        ordered_events = sorted(events, key=lambda event: event["frame"])
        if hits:
            last_hit = hits[-1]
            available_players = player_mini_positions[last_hit["frame"]]
            receiver_ids = [
                player_id
                for player_id in available_players
                if player_id != last_hit.get("player_id")
            ]
            if receiver_ids:
                receiver_id = receiver_ids[0]
                receiver_position = available_players[receiver_id]
                prior_delays = [
                    ordered_events[index + 1]["frame"] - event["frame"]
                    for index, event in enumerate(ordered_events[:-1])
                    if event["type"] == "hit"
                    and ordered_events[index + 1]["type"] == "bounce"
                ]
                expected_delay = (
                    float(np.median(prior_delays))
                    if prior_delays
                    else 1.2 * self.fps
                )
                net_y = (
                    mini_court.drawing_key_points[1]
                    + mini_court.drawing_key_points[5]
                ) / 2
                trailing_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate["frame"] >= last_hit["frame"] + max(5, round(0.2 * self.fps))
                    and (
                        0.45 * expected_delay
                        <= candidate["frame"] - last_hit["frame"]
                        <= 1.65 * expected_delay
                        or (
                            candidate.get("learned_bounce_confidence", 0.0)
                            >= 0.50
                            and candidate["frame"] - last_hit["frame"]
                            <= round(4.0 * self.fps)
                        )
                    )
                    and (
                        candidate["acceleration"] >= self.minimum_event_acceleration
                        or candidate.get("learned_bounce_confidence", 0.0)
                        >= 0.25
                    )
                    and self._is_plausible_bounce_candidate(candidate)
                    and ball_mini_positions[candidate["frame"]].get(1) is not None
                    and self._inside_court(
                        ball_mini_positions[candidate["frame"]][1], mini_court
                    )
                    and self._on_receiver_side(
                        ball_mini_positions[candidate["frame"]][1],
                        receiver_position,
                        net_y,
                    )
                ]
                if trailing_candidates:
                    for candidate in trailing_candidates:
                        delay = candidate["frame"] - last_hit["frame"]
                        candidate["trailing_bounce_score"] = (
                            candidate["acceleration"]
                            + (15.0 if candidate["vertical_reversal"] else 0.0)
                            - 4.0 * abs(delay - expected_delay) / expected_delay
                        )
                    best = max(
                        trailing_candidates,
                        key=lambda item: item["trailing_bounce_score"],
                    )
                    bounce_frame = self._refine_bounce_frame(
                        best["frame"],
                        best["vertical_reversal"],
                        ball_detections,
                    )
                    bounce_position = raw_ball_mini_positions[bounce_frame].get(1)
                    if (
                        bounce_position is None
                        or not self._inside_court(bounce_position, mini_court)
                        or not self._on_receiver_side(
                            bounce_position, receiver_position, net_y
                        )
                    ):
                        bounce_frame = best["frame"]
                        bounce_position = ball_mini_positions[bounce_frame][1]
                    current_receiver_position = player_mini_positions[
                        bounce_frame
                    ].get(receiver_id, receiver_position)
                    trailing_event = {
                        "frame": bounce_frame,
                        "type": "bounce",
                        "position": tuple(bounce_position),
                        "height": 0.0,
                        "inferred": False,
                        "trailing": True,
                        "bounce_number": 1,
                        "receiver_id": receiver_id,
                        "receiver_position": tuple(current_receiver_position),
                        "learned_confidence": best.get(
                            "learned_bounce_confidence"
                        ),
                        "event_source": (
                            "e2e_spot+trajectory"
                            if best.get("learned_bounce_confidence")
                            else "trajectory"
                        ),
                    }
                    ordered_events.append(trailing_event)

                    # Once the first legal landing is anchored, look for each
                    # later ground reversal. Wheelchair tennis may continue
                    # through bounce two; bounce three is retained as the
                    # terminal rules event. Standard tennis similarly keeps a
                    # second bounce so the ordinary one-bounce limit can be
                    # decided from direct evidence.
                    previous_frame = bounce_frame
                    used_frames = {bounce_frame}
                    maximum_number = self.allowed_bounces + 1
                    for bounce_number in range(2, maximum_number + 1):
                        minimum_gap = max(3, round(0.18 * self.fps))
                        maximum_gap = max(minimum_gap + 1, round(1.8 * self.fps))
                        later_candidates = [
                            candidate
                            for candidate in candidates
                            if previous_frame + minimum_gap
                            <= candidate["frame"]
                            <= previous_frame + maximum_gap
                            and candidate["frame"] not in used_frames
                            and (
                                candidate["acceleration"]
                                >= self.minimum_event_acceleration
                                or candidate.get(
                                    "learned_bounce_confidence", 0.0
                                )
                                >= 0.25
                            )
                            and (
                                candidate.get("vertical_reversal", False)
                                or candidate.get(
                                    "learned_bounce_confidence", 0.0
                                )
                                >= 0.50
                            )
                            and self._is_plausible_bounce_candidate(
                                candidate,
                                strict_contact_separation=False,
                            )
                            and ball_mini_positions[candidate["frame"]].get(1)
                            is not None
                            and self._on_receiver_side(
                                ball_mini_positions[candidate["frame"]][1],
                                receiver_position,
                                net_y,
                            )
                        ]
                        if not later_candidates:
                            break
                        next_candidate = max(
                            later_candidates,
                            key=self._bounce_candidate_score,
                        )
                        next_frame = self._refine_bounce_frame(
                            next_candidate["frame"],
                            True,
                            ball_detections,
                        )
                        next_position = raw_ball_mini_positions[
                            next_frame
                        ].get(1)
                        if next_position is None:
                            next_frame = next_candidate["frame"]
                            next_position = ball_mini_positions[next_frame].get(1)
                        if next_position is None:
                            break
                        current_receiver_position = player_mini_positions[
                            next_frame
                        ].get(receiver_id, receiver_position)
                        ordered_events.append(
                            {
                                "frame": next_frame,
                                "type": "bounce",
                                "position": tuple(next_position),
                                "height": 0.0,
                                "inferred": False,
                                "trailing": True,
                                "bounce_number": bounce_number,
                                "receiver_id": receiver_id,
                                "receiver_position": tuple(
                                    current_receiver_position
                                ),
                                "learned_confidence": next_candidate.get(
                                    "learned_bounce_confidence"
                                ),
                                "event_source": (
                                    "e2e_spot+trajectory"
                                    if next_candidate.get(
                                        "learned_bounce_confidence"
                                    )
                                    else "trajectory"
                                ),
                            }
                        )
                        previous_frame = next_frame
                        used_frames.add(next_candidate["frame"])

        return sorted(ordered_events, key=lambda event: event["frame"])

    def validate_trajectory(self, positions, heights, events, mini_court):
        """Fail fast if a reconstructed trajectory violates basic physics."""
        if not events:
            raise ValueError("No ball events were found")

        first_frame = events[0]["frame"]
        reconstructed_frames = [
            frame_num
            for frame_num, height in enumerate(heights)
            if height is not None
        ]
        if not reconstructed_frames:
            raise ValueError("No frames were reconstructed")
        last_frame = max(reconstructed_frames)
        missing_frames = [
            frame_num
            for frame_num in range(first_frame, last_frame + 1)
            if positions[frame_num].get(1) is None
        ]
        if missing_frames:
            raise ValueError(
                f"Reconstructed trajectory contains {len(missing_frames)} gaps"
            )

        for event in events:
            output_position = np.asarray(
                positions[event["frame"]].get(1), dtype=float
            )
            if np.linalg.norm(output_position - event["position"]) > 1e-3:
                raise ValueError(f"Trajectory misses its {event['type']} anchor")
            if event["type"] == "bounce":
                outside_allowed = (
                    self.allowed_bounces >= 2
                    and event.get("bounce_number", 1) >= 2
                )
                if (
                    not outside_allowed
                    and not self._inside_court(event["position"], mini_court)
                ):
                    raise ValueError("A bounce anchor is outside the court")
                net_y = (
                    mini_court.drawing_key_points[1]
                    + mini_court.drawing_key_points[5]
                ) / 2
                if not self._on_receiver_side(
                    event["position"], event["receiver_position"], net_y
                ):
                    raise ValueError("A bounce anchor is on the hitter's court half")
                if abs((heights[event["frame"]] or 0.0)) > 1e-6:
                    raise ValueError("Ball height must be zero at a bounce")

        maximum_height = max(heights[frame_num] for frame_num in reconstructed_frames)
        if maximum_height > 12.0:
            raise ValueError(
                f"Implausible reconstructed ball height: {maximum_height:.1f} m"
            )

        maximum_step = 0.0
        for frame_num in range(first_frame + 1, last_frame + 1):
            if heights[frame_num] is None or heights[frame_num - 1] is None:
                # Raw detector positions between disconnected reconstructed
                # segments are not a reconstructed flight and must not be
                # reported as its physical speed.
                continue
            current = np.asarray(positions[frame_num][1], dtype=float)
            previous = np.asarray(positions[frame_num - 1][1], dtype=float)
            maximum_step = max(
                maximum_step, float(np.linalg.norm(current - previous))
            )
        maximum_speed_kmh = (
            mini_court.convert_pixels_to_meters(maximum_step)
            * self.fps
            * 3.6
        )
        if maximum_speed_kmh > 260.0:
            raise ValueError(
                "Implausible mini-court trajectory speed: "
                f"{maximum_speed_kmh:.1f} km/h"
            )

        return {
            "events": len(events),
            "bounces": sum(event["type"] == "bounce" for event in events),
            "inferred_bounces": sum(
                event["type"] == "bounce" and event.get("inferred", False)
                for event in events
            ),
            "reconstructed_frames": len(reconstructed_frames),
            "maximum_height_m": round(float(maximum_height), 2),
            "maximum_horizontal_speed_kmh": round(maximum_speed_kmh, 1),
        }

    def _refine_bounce_frame(
        self,
        candidate_frame,
        has_vertical_reversal,
        ball_detections,
    ):
        start = max(1, candidate_frame - 4)
        end = min(len(ball_detections) - 2, candidate_frame + 4)
        centers = {}
        for frame_num in range(start - 1, end + 2):
            bbox = ball_detections[frame_num].get(1)
            if bbox is not None:
                centers[frame_num] = self._bbox_center(bbox)

        valid_frames = [
            frame_num for frame_num in range(start, end + 1)
            if frame_num - 1 in centers
            and frame_num in centers
            and frame_num + 1 in centers
        ]
        if not valid_frames:
            return candidate_frame

        if has_vertical_reversal:
            # At the bottom of a visible down/up arc, image y is maximal.
            return max(valid_frames, key=lambda frame_num: centers[frame_num][1])

        # Perspective can hide the y reversal.  The impact still creates the
        # strongest local velocity discontinuity.
        return max(
            valid_frames,
            key=lambda frame_num: np.linalg.norm(
                (centers[frame_num + 1] - centers[frame_num])
                - (centers[frame_num] - centers[frame_num - 1])
            ),
        )

    @staticmethod
    def _on_receiver_side(position, receiver_position, net_y, margin=6.0):
        receiver_side = np.sign(receiver_position[1] - net_y)
        if receiver_side == 0:
            return False
        return (position[1] - net_y) * receiver_side > margin

    @staticmethod
    def _clamp_to_court(position, mini_court, padding=3.0):
        court_bottom = mini_court.drawing_key_points[5]
        return np.asarray(
            (
                np.clip(
                    position[0],
                    mini_court.court_start_x + padding,
                    mini_court.court_end_x - padding,
                ),
                np.clip(
                    position[1],
                    mini_court.court_start_y + padding,
                    court_bottom - padding,
                ),
            ),
            dtype=float,
        )

    def _infer_missing_gap_hits(
        self,
        hits,
        ball_detections,
        player_detections,
        player_mini_positions,
        ball_mini_positions,
    ):
        """Recover a contact hidden at the edge of a long detector gap.

        Consecutive contacts by the same player in singles usually mean the
        opponent's return was lost with the ball. Only add a contact when a
        long empty run starts or ends with the ball at that opponent.
        """
        if len(hits) < 2:
            return hits

        minimum_gap = max(5, round(0.30 * self.fps))
        inferred_hits = []
        for first_hit, second_hit in zip(hits, hits[1:]):
            if (
                first_hit.get("player_id") != second_hit.get("player_id")
                or first_hit.get("player_id") not in (1, 2)
            ):
                continue
            missing_player_id = 1 if first_hit["player_id"] == 2 else 2
            candidate_frames = []
            frame_num = first_hit["frame"] + 1
            while frame_num < second_hit["frame"]:
                if ball_detections[frame_num].get(1) is not None:
                    frame_num += 1
                    continue
                gap_start = frame_num
                while (
                    frame_num < second_hit["frame"]
                    and ball_detections[frame_num].get(1) is None
                ):
                    frame_num += 1
                gap_end = frame_num - 1
                if gap_end - gap_start + 1 < minimum_gap:
                    continue
                for boundary in (gap_start - 1, gap_end + 1):
                    if not first_hit["frame"] < boundary < second_hit["frame"]:
                        continue
                    ball_bbox = ball_detections[boundary].get(1)
                    player_bbox = player_detections[boundary].get(
                        missing_player_id
                    )
                    if ball_bbox is None or player_bbox is None:
                        continue
                    ball_center = self._bbox_center(ball_bbox)
                    player_height = max(1.0, player_bbox[3] - player_bbox[1])
                    distance_ratio = (
                        self._point_to_bbox_distance(ball_center, player_bbox)
                        / player_height
                    )
                    if distance_ratio <= 0.30:
                        candidate_frames.append((distance_ratio, boundary))

            if not candidate_frames:
                continue
            _, inferred_frame = min(candidate_frames)
            inferred_hit = self._make_hit_event(
                inferred_frame,
                ball_detections,
                player_detections,
                player_mini_positions,
                ball_mini_positions,
            )
            if (
                inferred_hit is None
                or inferred_hit.get("player_id") != missing_player_id
            ):
                continue
            inferred_hit["inferred"] = True
            inferred_hit["event_source"] = "detector_gap"
            inferred_hits.append(inferred_hit)

        return sorted(
            [*hits, *inferred_hits],
            key=lambda event: event["frame"],
        )

    def _make_hit_event(
        self,
        frame_num,
        ball_detections,
        player_detections,
        player_mini_positions,
        ball_mini_positions,
    ):
        ball_bbox = ball_detections[frame_num].get(1)
        players = player_detections[frame_num]
        if ball_bbox is None:
            return None

        ball_center = self._bbox_center(ball_bbox)
        player_id = None
        if players:
            player_id = min(
                players,
                key=lambda identifier: self._point_to_bbox_distance(
                    ball_center, players[identifier]
                ),
            )

        position = None
        height = 1.0
        if player_id is not None:
            position = player_mini_positions[frame_num].get(player_id)
            x1, y1, x2, y2 = players[player_id]
            player_height_pixels = max(1.0, y2 - y1)
            height = np.clip(
                (y2 - ball_center[1]) / player_height_pixels * 1.8,
                0.35,
                2.7,
            )
        if position is None:
            position = ball_mini_positions[frame_num].get(1)
        if position is None:
            return None

        return {
            "frame": frame_num,
            "type": "hit",
            "position": tuple(position),
            "height": float(height),
            "player_id": player_id,
        }

    def _fit_flight_segment(
        self,
        output_positions,
        ball_heights,
        start_event,
        end_event,
    ):
        start_frame = start_event["frame"]
        end_frame = end_event["frame"]
        duration = (end_frame - start_frame) / self.fps
        start_position = np.asarray(start_event["position"], dtype=float)
        end_position = np.asarray(end_event["position"], dtype=float)
        start_height = start_event["height"]
        end_height = end_event["height"]
        vertical_velocity = (
            end_height
            - start_height
            + 0.5 * self.gravity * duration * duration
        ) / duration

        final_output_frame = min(end_frame, len(output_positions) - 1)
        for frame_num in range(start_frame, final_output_frame + 1):
            elapsed = (frame_num - start_frame) / self.fps
            ratio = elapsed / duration
            position = start_position + ratio * (end_position - start_position)
            height = (
                start_height
                + vertical_velocity * elapsed
                - 0.5 * self.gravity * elapsed * elapsed
            )
            output_positions[frame_num] = {
                1: (float(position[0]), float(position[1]))
            }
            ball_heights[frame_num] = max(0.0, float(height))

    def _fit_open_ended_tail(
        self,
        output_positions,
        ball_heights,
        events,
        player_mini_positions,
        mini_court,
    ):
        """Continue a final partial flight to an event beyond the clip."""
        if not events:
            return

        if events[-1]["type"] == "bounce":
            last_bounce = events[-1]
            last_output_frame = len(output_positions) - 1
            if last_bounce["frame"] >= last_output_frame:
                return

            historical_contact_delays = []
            for event_index, event in enumerate(events[:-1]):
                next_event = events[event_index + 1]
                if event["type"] == "bounce" and next_event["type"] == "hit":
                    historical_contact_delays.append(
                        next_event["frame"] - event["frame"]
                    )
            if historical_contact_delays:
                frames_until_contact = round(
                    float(np.median(historical_contact_delays))
                )
            else:
                frames_until_contact = round(0.5 * self.fps)
            frames_until_contact = int(
                np.clip(
                    frames_until_contact,
                    round(0.25 * self.fps),
                    round(0.9 * self.fps),
                )
            )
            receiver_id = last_bounce.get("receiver_id")
            receiver_position = player_mini_positions[
                last_bounce["frame"]
            ].get(receiver_id, last_bounce["receiver_position"])
            virtual_contact = {
                "frame": last_bounce["frame"] + frames_until_contact,
                "type": "hit",
                "position": tuple(receiver_position),
                "height": 1.0,
                "inferred": True,
                "virtual": True,
                "player_id": receiver_id,
            }
            self._fit_flight_segment(
                output_positions,
                ball_heights,
                last_bounce,
                virtual_contact,
            )
            return

        if events[-1]["type"] != "hit":
            return

        last_hit = events[-1]
        last_output_frame = len(output_positions) - 1
        if last_hit["frame"] >= last_output_frame:
            return

        hitter_id = last_hit.get("player_id")
        available_players = player_mini_positions[last_hit["frame"]]
        receiver_ids = [
            player_id for player_id in available_players if player_id != hitter_id
        ]
        if not receiver_ids:
            return
        receiver_id = receiver_ids[0]
        receiver_position = available_players[receiver_id]

        historical_bounce_delays = []
        for event_index, event in enumerate(events[:-1]):
            next_event = events[event_index + 1]
            if event["type"] == "hit" and next_event["type"] == "bounce":
                historical_bounce_delays.append(
                    next_event["frame"] - event["frame"]
                )
        if historical_bounce_delays:
            frames_until_landing = round(float(np.median(historical_bounce_delays)))
        else:
            frames_until_landing = round(1.5 * self.fps)
        frames_until_landing = int(
            np.clip(
                frames_until_landing,
                round(0.9 * self.fps),
                round(2.2 * self.fps),
            )
        )

        start_position = np.asarray(last_hit["position"], dtype=float)
        landing_position = start_position + 0.80 * (
            np.asarray(receiver_position, dtype=float) - start_position
        )
        landing_position = self._clamp_to_court(landing_position, mini_court)
        virtual_landing = {
            "frame": last_hit["frame"] + frames_until_landing,
            "type": "bounce",
            "position": tuple(landing_position),
            "height": 0.0,
            "inferred": True,
            "virtual": True,
            "receiver_id": receiver_id,
            "receiver_position": tuple(receiver_position),
        }
        self._fit_flight_segment(
            output_positions,
            ball_heights,
            last_hit,
            virtual_landing,
        )

    @staticmethod
    def _bbox_center(bbox):
        return np.asarray(
            ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
            dtype=float,
        )

    def _distance_to_players(self, point, player_boxes):
        if not player_boxes:
            return float("inf")
        return min(
            self._point_to_bbox_distance(point, bbox)
            for bbox in player_boxes.values()
        )

    def _nearest_player_id(self, point, player_boxes):
        if not player_boxes:
            return None
        return min(
            player_boxes,
            key=lambda identifier: self._point_to_bbox_distance(
                point, player_boxes[identifier]
            ),
        )

    @staticmethod
    def _point_to_bbox_distance(point, bbox):
        x, y = point
        x1, y1, x2, y2 = bbox
        dx = max(x1 - x, 0.0, x - x2)
        dy = max(y1 - y, 0.0, y - y2)
        return float(np.hypot(dx, dy))

    @staticmethod
    def _inside_court(position, mini_court, padding=8.0):
        x, y = position
        court_bottom = mini_court.drawing_key_points[5]
        return (
            mini_court.court_start_x - padding
            <= x
            <= mini_court.court_end_x + padding
            and mini_court.court_start_y - padding <= y <= court_bottom + padding
        )

    @staticmethod
    def get_predicted_high_arc_frames(
        ball_heights,
        events,
        minimum_height=3.5,
    ):
        inferred_flights = [
            (event["frame"], next_event["frame"])
            for event, next_event in zip(events, events[1:])
            if event["type"] == "hit"
            and next_event["type"] == "bounce"
            and next_event.get("inferred", False)
        ]
        return {
            frame_num
            for start, end in inferred_flights
            for frame_num in range(start, end + 1)
            if ball_heights[frame_num] is not None
            and ball_heights[frame_num] >= minimum_height
        }

    @classmethod
    def draw_debug_overlay(cls, frames, ball_positions, ball_heights, events):
        event_by_frame = {event["frame"]: event for event in events}
        predicted_high_arc_frames = cls.get_predicted_high_arc_frames(
            ball_heights,
            events,
        )
        for frame_num, frame in enumerate(frames):
            height = ball_heights[frame_num]
            position = ball_positions[frame_num].get(1)
            if height is not None and position is not None:
                cv2.putText(
                    frame,
                    f"h={height:.1f}m",
                    (int(position[0]) + 8, int(position[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 180, 255),
                    1,
                    cv2.LINE_AA,
                )
                is_predicted_high_arc = (
                    frame_num in predicted_high_arc_frames
                )
                if is_predicted_high_arc:
                    cv2.putText(
                        frame,
                        "HIGH ARC - PREDICTED",
                        (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 180, 255),
                        2,
                        cv2.LINE_AA,
                    )

            event = event_by_frame.get(frame_num)
            if event is not None:
                color = (0, 0, 255) if event["type"] == "hit" else (255, 0, 255)
                label = event["type"].upper()
                if event.get("inferred", False):
                    label += " (EST)"
                cv2.putText(
                    frame,
                    label,
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        return frames

    def draw_bounce_markers(
        self,
        frames,
        events,
        court_keypoints_per_frame,
        mini_court,
    ):
        """Mark every landing until the receiver returns the ball.

        Markers are numbered per incoming shot. This leaves B1 and B2 visible
        together for wheelchair play, and labels a bounce beyond the permitted
        count as the bounce-limit event.
        """
        bounces = self._bounce_display_records(events, len(frames))

        for frame_num, frame in enumerate(frames):
            active_bounces = [
                record
                for record in bounces
                if record["frame"] <= frame_num < record["display_until"]
            ]
            if not active_bounces:
                continue

            try:
                image_to_mini = mini_court.get_court_homography(
                    court_keypoints_per_frame[frame_num]
                )
                mini_to_image = np.linalg.inv(image_to_mini)
            except (ValueError, np.linalg.LinAlgError):
                mini_to_image = None

            for event in active_bounces:
                alpha = 0.86
                scale = min(
                    frame.shape[1] / 1280.0,
                    frame.shape[0] / 720.0,
                )
                radius = max(5, round(9 * scale))
                mini_position = event["position"]

                marker_positions = [mini_position]
                if mini_to_image is not None:
                    actual_position = mini_court.project_point_to_mini_court(
                        mini_position,
                        mini_to_image,
                    )
                    marker_positions.append(actual_position)

                overlay = frame.copy()
                bounce_number = event["bounce_number"]
                colors = {
                    1: (0, 140, 255),
                    2: (255, 0, 255),
                }
                marker_color = colors.get(bounce_number, (0, 0, 255))
                label = f"B{bounce_number}"
                if bounce_number > self.allowed_bounces:
                    label += " LIMIT"
                if event.get("inferred", False):
                    label += " EST"
                for x, y in marker_positions:
                    center = (int(round(x)), int(round(y)))
                    cv2.circle(overlay, center, radius + 3, (255, 255, 255), 2)
                    cv2.circle(overlay, center, radius, marker_color, -1)
                    cv2.putText(
                        overlay,
                        label,
                        (center[0] + radius + 4, center[1] - radius - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.38, 0.50 * scale),
                        (255, 255, 255),
                        max(1, round(2 * scale)),
                        cv2.LINE_AA,
                    )
                cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        return frames

    def _bounce_display_records(self, events, frame_count):
        """Attach per-flight numbers and the next-hit display boundary."""
        ordered = sorted(events, key=lambda event: event["frame"])
        records = []
        bounce_number = 0
        for index, event in enumerate(ordered):
            if event.get("type") == "hit":
                bounce_number = 0
                continue
            if event.get("type") != "bounce":
                continue
            bounce_number = int(event.get("bounce_number", bounce_number + 1))
            next_hit_frame = next(
                (
                    later["frame"]
                    for later in ordered[index + 1 :]
                    if later.get("type") == "hit"
                ),
                frame_count,
            )
            records.append(
                {
                    **event,
                    "bounce_number": bounce_number,
                    "display_until": next_hit_frame,
                }
            )
        return records
