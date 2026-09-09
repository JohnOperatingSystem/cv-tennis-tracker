from dataclasses import asdict, dataclass, field

import numpy as np

import constants


@dataclass
class PointResult:
    winner_id: int | None
    loser_id: int | None
    reason: str
    confidence: float
    terminal_frame: int | None
    details: dict = field(default_factory=dict)

    @property
    def decided(self):
        return self.winner_id is not None

    def to_dict(self):
        value = asdict(self)
        value["winner_side"] = {1: "far", 2: "near"}.get(self.winner_id)
        value["loser_side"] = {1: "far", 2: "near"}.get(self.loser_id)
        return value


class PointAnalyzer:
    """Convert tracking evidence into a conservative singles point result."""

    def __init__(
        self,
        fps,
        singles=True,
        line_uncertainty_meters=0.08,
        clip_ends_with_point=True,
        allowed_bounces=1,
    ):
        if fps <= 0:
            raise ValueError("Video FPS must be greater than zero")
        self.fps = float(fps)
        self.singles = bool(singles)
        self.line_uncertainty_meters = float(line_uncertainty_meters)
        self.clip_ends_with_point = bool(clip_ends_with_point)
        self.allowed_bounces = int(allowed_bounces)
        if self.allowed_bounces < 1:
            raise ValueError("At least one bounce must be allowed")

    def analyze(
        self,
        ball_events,
        motion_candidates,
        raw_ball_positions,
        player_positions,
        mini_court,
        ball_heights=None,
        observed_ball_frames=None,
        temporal_outcome=None,
    ):
        geometry_result = self._analyze_geometry(
            ball_events,
            motion_candidates,
            raw_ball_positions,
            player_positions,
            mini_court,
            ball_heights=ball_heights,
            observed_ball_frames=observed_ball_frames,
        )
        return self._fuse_temporal_outcome(geometry_result, temporal_outcome)

    def _analyze_geometry(
        self,
        ball_events,
        motion_candidates,
        raw_ball_positions,
        player_positions,
        mini_court,
        ball_heights=None,
        observed_ball_frames=None,
    ):
        hits = sorted(
            (
                event
                for event in ball_events
                if event.get("type") == "hit"
                and not event.get("virtual", False)
            ),
            key=lambda event: event["frame"],
        )
        if not hits:
            return self._unknown("No racket contact was detected")

        last_hit = hits[-1]
        hitter_id = last_hit.get("player_id")
        if hitter_id not in (1, 2):
            return self._unknown(
                "The final hitter could not be assigned to a player",
                terminal_frame=last_hit.get("frame"),
            )
        opponent_id = 1 if hitter_id == 2 else 2
        observed_frames = set(observed_ball_frames or ())
        heights = (
            ball_heights
            if ball_heights is not None
            else [None] * len(raw_ball_positions)
        )

        net_event = self._detect_terminal_net_contact(
            last_hit,
            raw_ball_positions,
            heights,
            motion_candidates,
            mini_court,
            observed_frames,
        )
        impacts = self._terminal_bounce_candidates(
            last_hit,
            hits,
            ball_events,
            motion_candidates,
            raw_ball_positions,
            mini_court,
            observed_frames,
        )

        first_bounce = impacts[0] if impacts else None
        if net_event is not None and (
            first_bounce is None
            or net_event["frame"] <= first_bounce["frame"]
        ):
            return PointResult(
                winner_id=opponent_id,
                loser_id=hitter_id,
                reason="net",
                confidence=net_event["confidence"],
                terminal_frame=net_event["frame"],
                details={"net_event": net_event},
            )

        if first_bounce is None:
            return self._unknown(
                "No observed terminal net contact or landing was found",
                terminal_frame=last_hit["frame"],
                details={"last_hitter_id": hitter_id},
            )

        is_serve = len(hits) == 1
        if is_serve:
            landing_state = self._classify_service_landing(
                first_bounce["position"],
                last_hit["position"],
                mini_court,
            )
        else:
            landing_state = self._classify_court_landing(
                first_bounce["position"], mini_court
            )

        if landing_state == "uncertain":
            return self._unknown(
                "The terminal landing is too close to a line to call reliably",
                terminal_frame=first_bounce["frame"],
                details={"bounce": first_bounce, "serve": is_serve},
            )
        if landing_state == "out":
            return PointResult(
                winner_id=opponent_id,
                loser_id=hitter_id,
                reason="out",
                confidence=first_bounce["confidence"],
                terminal_frame=first_bounce["frame"],
                details={
                    "bounce": first_bounce,
                    "serve_fault": is_serve,
                },
            )

        # Standard tennis permits one bounce; wheelchair tennis permits two.
        # The first landing must be in bounds in both formats. In wheelchair
        # tennis, the second bounce may be inside or outside the court, and a
        # third bounce ends the point.
        limit_bounce = (
            impacts[self.allowed_bounces]
            if len(impacts) > self.allowed_bounces
            else None
        )
        frames_after_bounce = len(raw_ball_positions) - 1 - first_bounce["frame"]
        enough_terminal_video = frames_after_bounce >= max(
            2, round(0.20 * self.fps)
        )
        if limit_bounce is not None:
            confidence = min(
                0.98,
                max(first_bounce["confidence"], limit_bounce["confidence"])
                + 0.08,
            )
            terminal_frame = limit_bounce["frame"]
            terminal_inferred = False
        elif (
            self.clip_ends_with_point
            and enough_terminal_video
            and len(impacts) >= self.allowed_bounces
        ):
            confidence = min(0.88, first_bounce["confidence"])
            terminal_frame = impacts[self.allowed_bounces - 1]["frame"]
            terminal_inferred = True
        else:
            return self._unknown(
                "The ball has not exceeded the permitted bounce count",
                terminal_frame=first_bounce["frame"],
                details={
                    "bounces": impacts[: self.allowed_bounces],
                    "detected_bounce_count": len(impacts),
                    "allowed_bounces": self.allowed_bounces,
                },
            )

        return PointResult(
            winner_id=hitter_id,
            loser_id=opponent_id,
            reason="winner",
            confidence=confidence,
            terminal_frame=terminal_frame,
            details={
                "bounce": first_bounce,
                "bounces": impacts[: self.allowed_bounces + 1],
                "detected_bounce_count": len(impacts),
                "allowed_bounces": self.allowed_bounces,
                "bounce_limit_exceeded": limit_bounce is not None,
                "terminal_inferred_from_clip_end": terminal_inferred,
                "serve_winner": is_serve,
            },
        )

    @staticmethod
    def _fuse_temporal_outcome(geometry_result, temporal_outcome):
        """Use a temporal model conservatively as fallback/corroboration."""
        if not temporal_outcome:
            return geometry_result
        side = str(temporal_outcome.get("winner_side", "")).lower()
        reason = str(temporal_outcome.get("reason", "")).lower()
        try:
            model_confidence = float(temporal_outcome.get("confidence", 0.0))
        except (TypeError, ValueError):
            return geometry_result
        winner_id = {"far": 1, "near": 2}.get(side)
        if (
            winner_id is None
            or reason not in ("net", "out", "winner")
            or not np.isfinite(model_confidence)
        ):
            return geometry_result

        temporal_record = dict(temporal_outcome)
        temporal_record["mapped_winner_id"] = winner_id
        if not geometry_result.decided:
            if model_confidence < 0.75:
                details = dict(geometry_result.details)
                details["temporal_outcome"] = temporal_record
                details["message"] = (
                    details.get("message", "Geometry was inconclusive")
                    + "; temporal confidence was below 0.75"
                )
                geometry_result.details = details
                return geometry_result
            return PointResult(
                winner_id=winner_id,
                loser_id=1 if winner_id == 2 else 2,
                reason=reason,
                confidence=round(min(0.82, model_confidence), 3),
                terminal_frame=geometry_result.terminal_frame,
                details={
                    **geometry_result.details,
                    "temporal_outcome": temporal_record,
                    "decision_source": "tenniset_temporal_fallback",
                },
            )

        details = dict(geometry_result.details)
        details["temporal_outcome"] = temporal_record
        if (
            geometry_result.winner_id == winner_id
            and geometry_result.reason == reason
        ):
            geometry_result.confidence = round(
                min(
                    0.99,
                    geometry_result.confidence
                    + 0.08 * max(0.0, model_confidence - 0.5),
                ),
                3,
            )
            details["temporal_agreement"] = True
        else:
            # Weak commentary labels and a small dataset should not override a
            # complete geometric call. Preserve the conflict for evaluation.
            details["temporal_agreement"] = False
        geometry_result.details = details
        return geometry_result

    def _terminal_bounce_candidates(
        self,
        last_hit,
        hits,
        ball_events,
        motion_candidates,
        raw_ball_positions,
        mini_court,
        observed_frames,
    ):
        margin = max(2, round(0.10 * self.fps))
        hit_frames = [hit["frame"] for hit in hits]
        net_y = self._net_y(mini_court)
        net_exclusion = mini_court.convert_meters_to_pixels(0.65)
        impacts = []

        for event in ball_events:
            if (
                event.get("type") != "bounce"
                or event.get("inferred", False)
                or event["frame"] <= last_hit["frame"] + margin
            ):
                continue
            position = self._finite_position(event.get("position"))
            if position is None:
                continue
            quality = self._observation_quality(event["frame"], observed_frames)
            impacts.append(
                self._impact_record(
                    event["frame"], position, quality, source="trajectory_event"
                )
            )

        for candidate in motion_candidates:
            frame_num = int(candidate["frame"])
            if frame_num <= last_hit["frame"] + margin:
                continue
            if any(abs(frame_num - hit_frame) <= margin for hit_frame in hit_frames):
                continue
            if not candidate.get("vertical_reversal", False):
                continue
            if candidate.get("player_distance_ratio", 0.0) < 0.12:
                continue
            position = self._position_at(raw_ball_positions, frame_num)
            if position is None:
                continue
            if abs(position[1] - net_y) <= net_exclusion:
                continue
            quality = self._observation_quality(frame_num, observed_frames)
            impacts.append(
                self._impact_record(
                    frame_num, position, quality, source="raw_motion"
                )
            )

        impacts.sort(key=lambda impact: impact["frame"])
        deduplicated = []
        dedupe_frames = max(2, round(0.12 * self.fps))
        for impact in impacts:
            if deduplicated and impact["frame"] - deduplicated[-1]["frame"] <= dedupe_frames:
                if impact["confidence"] > deduplicated[-1]["confidence"]:
                    deduplicated[-1] = impact
                continue
            deduplicated.append(impact)
        return deduplicated

    def _detect_terminal_net_contact(
        self,
        last_hit,
        raw_ball_positions,
        ball_heights,
        motion_candidates,
        mini_court,
        observed_frames,
    ):
        net_y = self._net_y(mini_court)
        net_zone = mini_court.convert_meters_to_pixels(0.75)
        hit_position = self._finite_position(last_hit.get("position"))
        if hit_position is None:
            return None
        hitter_side = np.sign(hit_position[1] - net_y)
        if hitter_side == 0:
            return None

        samples = []
        analysis_end = min(
            len(raw_ball_positions),
            last_hit["frame"] + max(6, round(1.5 * self.fps)) + 1,
        )
        for frame_num in range(last_hit["frame"] + 1, analysis_end):
            if observed_frames and frame_num not in observed_frames:
                continue
            position = self._position_at(raw_ball_positions, frame_num)
            if position is not None:
                samples.append((frame_num, position))
        if len(samples) < 2:
            return None

        nearest_frame, nearest_position = min(
            samples, key=lambda sample: abs(sample[1][1] - net_y)
        )
        nearest_distance = abs(nearest_position[1] - net_y)
        crossed_to_receiver = self._has_sustained_net_crossing(
            samples,
            hitter_side,
            net_y,
            net_zone,
        )
        if crossed_to_receiver or nearest_distance > net_zone:
            return None

        candidate_near_net = any(
            abs(candidate["frame"] - nearest_frame) <= max(2, round(0.12 * self.fps))
            and candidate.get("acceleration", 0.0) >= 1.5
            for candidate in motion_candidates
        )
        terminal_near_net = abs(samples[-1][1][1] - net_y) <= 1.25 * net_zone
        reversed_away = bool(self._reversed_away_from_net(
            samples, hitter_side, net_y
        ))
        if not (candidate_near_net or terminal_near_net or reversed_away):
            return None

        height = (
            ball_heights[nearest_frame]
            if nearest_frame < len(ball_heights)
            else None
        )
        low_enough = height is None or height <= constants.NET_HEIGHT_POSTS + 0.25
        evidence_count = sum(
            (candidate_near_net, terminal_near_net, reversed_away, low_enough)
        )
        confidence = float(min(0.94, 0.55 + 0.09 * evidence_count))
        if height is not None and not low_enough:
            # Final-flight height is often reconstructed rather than observed,
            # so it may lower confidence but must not veto raw stop/reversal
            # evidence at the net plane.
            confidence -= 0.10
        return {
            "frame": nearest_frame,
            "position": self._serializable_position(nearest_position),
            "height_m": None if height is None else round(float(height), 3),
            "confidence": round(confidence, 3),
            "motion_discontinuity": candidate_near_net,
            "reversed": reversed_away,
        }

    def _has_sustained_net_crossing(
        self,
        samples,
        hitter_side,
        net_y,
        net_zone,
    ):
        """Require a real receiver-side run, not one false detection."""
        minimum_frames = max(2, round(0.08 * self.fps))
        run = 0
        previous_frame = None
        for frame_num, position in samples:
            receiver_side = (position[1] - net_y) * hitter_side < -net_zone
            consecutive = (
                previous_frame is not None
                and frame_num - previous_frame <= 2
            )
            if receiver_side:
                run = run + 1 if consecutive else 1
                if run >= minimum_frames:
                    return True
            else:
                run = 0
            previous_frame = frame_num
        return False

    @staticmethod
    def _reversed_away_from_net(samples, hitter_side, net_y):
        if len(samples) < 4:
            return False
        y_values = np.asarray([position[1] for _, position in samples], dtype=float)
        nearest_index = int(np.argmin(np.abs(y_values - net_y)))
        velocity = np.diff(y_values)
        if nearest_index == 0 or nearest_index >= len(velocity):
            return False
        before = float(np.median(velocity[:nearest_index]))
        after = float(np.median(velocity[nearest_index:]))
        return before * hitter_side < 0 and after * hitter_side > 0

    def _classify_court_landing(self, position, mini_court):
        if self.singles:
            left = float(mini_court.drawing_key_points[8])
            right = float(mini_court.drawing_key_points[12])
        else:
            left = float(mini_court.court_start_x)
            right = float(mini_court.court_end_x)
        top = float(mini_court.court_start_y)
        bottom = float(mini_court.drawing_key_points[5])
        return self._classify_rectangle(position, left, right, top, bottom, mini_court)

    def _classify_service_landing(self, position, server_position, mini_court):
        left = float(mini_court.drawing_key_points[8])
        right = float(mini_court.drawing_key_points[12])
        center_x = (left + right) / 2.0
        net_y = self._net_y(mini_court)
        server_is_near = server_position[1] > net_y
        server_is_right = server_position[0] > center_x

        target_left = server_is_right
        box_left, box_right = (
            (left, center_x) if target_left else (center_x, right)
        )
        if server_is_near:
            box_top = float(mini_court.drawing_key_points[17])
            box_bottom = net_y
        else:
            box_top = net_y
            box_bottom = float(mini_court.drawing_key_points[21])
        return self._classify_rectangle(
            position,
            box_left,
            box_right,
            min(box_top, box_bottom),
            max(box_top, box_bottom),
            mini_court,
        )

    def _classify_rectangle(self, position, left, right, top, bottom, mini_court):
        x, y = position
        ball_radius = mini_court.convert_meters_to_pixels(constants.BALL_RADIUS)
        uncertainty = mini_court.convert_meters_to_pixels(
            self.line_uncertainty_meters
        )

        inside_with_ball = (
            left - ball_radius <= x <= right + ball_radius
            and top - ball_radius <= y <= bottom + ball_radius
        )
        if inside_with_ball:
            return "in"
        clearly_out = (
            x < left - ball_radius - uncertainty
            or x > right + ball_radius + uncertainty
            or y < top - ball_radius - uncertainty
            or y > bottom + ball_radius + uncertainty
        )
        return "out" if clearly_out else "uncertain"

    @staticmethod
    def _position_at(positions, frame_num):
        if not 0 <= frame_num < len(positions):
            return None
        return PointAnalyzer._finite_position(positions[frame_num].get(1))

    @staticmethod
    def _finite_position(position):
        if position is None:
            return None
        point = np.asarray(position, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            return None
        return point

    @staticmethod
    def _observation_quality(frame_num, observed_frames):
        if not observed_frames:
            return ("observed", 0.82)
        if any(abs(frame_num - observed) <= 1 for observed in observed_frames):
            return ("observed", 0.90)
        return ("interpolated", 0.58)

    @staticmethod
    def _impact_record(frame_num, position, quality, source):
        quality_name, confidence = quality
        return {
            "frame": int(frame_num),
            "position": PointAnalyzer._serializable_position(position),
            "quality": quality_name,
            "confidence": round(float(confidence), 3),
            "source": source,
        }

    @staticmethod
    def _serializable_position(position):
        return [round(float(position[0]), 3), round(float(position[1]), 3)]

    @staticmethod
    def _net_y(mini_court):
        return float(
            (
                mini_court.drawing_key_points[1]
                + mini_court.drawing_key_points[5]
            )
            / 2.0
        )

    @staticmethod
    def _unknown(message, terminal_frame=None, details=None):
        result_details = dict(details or {})
        result_details["message"] = message
        return PointResult(
            winner_id=None,
            loser_id=None,
            reason="unknown",
            confidence=0.0,
            terminal_frame=terminal_frame,
            details=result_details,
        )
