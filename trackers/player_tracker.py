from ultralytics import YOLO
import cv2
import itertools
import math
import numpy as np
from pathlib import Path
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_foot_position

class PlayerTracker:
    VALID_MODES = {"auto", "standard", "wheelchair"}
    STANDARD_CONFIDENCE = 0.10
    WHEELCHAIR_CONFIDENCE = 0.08
    WHEELCHAIR_IMAGE_SIZE = 960

    def __init__(
        self,
        model_path,
        wheelchair_model_path=None,
        mode="auto",
    ):
        mode = str(mode).lower()
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Player mode must be one of {sorted(self.VALID_MODES)}"
            )

        self.mode = mode
        self.active_mode = None
        self.standard_model_path = Path(model_path)
        self.wheelchair_model_path = (
            None
            if wheelchair_model_path is None
            else Path(wheelchair_model_path)
        )

        self.standard_model = (
            None
            if mode == "wheelchair"
            else YOLO(str(self.standard_model_path))
        )
        wheelchair_exists = (
            self.wheelchair_model_path is not None
            and self.wheelchair_model_path.is_file()
        )
        if mode == "wheelchair" and not wheelchair_exists:
            raise FileNotFoundError(
                "Wheelchair mode requires a valid wheelchair model: "
                f"{self.wheelchair_model_path}"
            )
        self.wheelchair_model = (
            YOLO(str(self.wheelchair_model_path))
            if wheelchair_exists
            else None
        )
        # Retain the old public attribute for small external scripts that use
        # PlayerTracker.model directly.
        self.model = self.standard_model or self.wheelchair_model

    def cache_signature(self):
        """Return all detector inputs that can change cached predictions."""
        return {
            "requested_mode": self.mode,
            "standard_model": self._file_signature(self.standard_model_path),
            "wheelchair_model": self._file_signature(
                self.wheelchair_model_path
            ),
            "standard_confidence": self.STANDARD_CONFIDENCE,
            "wheelchair_confidence": self.WHEELCHAIR_CONFIDENCE,
            "wheelchair_image_size": self.WHEELCHAIR_IMAGE_SIZE,
        }

    @staticmethod
    def _file_signature(path):
        if path is None or not path.is_file():
            return None
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "modified_ns": stat.st_mtime_ns,
        }

    def choose_and_filter_players(self, court_keypoints, player_detections):
        """Return the two competitors with stable logical IDs 1 and 2.

        Ultralytics IDs are implementation details and can be any integers.  A
        downstream statistics or rules component must therefore never assume
        that the two competitors happened to receive raw IDs 1 and 2.  Logical
        player 1 is initialized on the far/top half of the image and logical
        player 2 on the near/bottom half.  Subsequent frames are assigned using
        tracker-ID affinity, position continuity, and court-side consistency.
        """
        if not player_detections:
            return []

        keypoints_per_frame = self._normalize_keypoint_frames(
            court_keypoints,
            len(player_detections),
        )
        seed_frame = next(
            (
                frame_num
                for frame_num, detections in enumerate(player_detections)
                if len(detections) >= 2
            ),
            None,
        )
        if seed_frame is None:
            raise ValueError("At least two player detections are required")

        chosen_players = self.choose_players(
            keypoints_per_frame[seed_frame],
            player_detections[seed_frame],
        )
        chosen_players.sort(
            key=lambda track_id: get_foot_position(
                player_detections[seed_frame][track_id]
            )[1]
        )
        preferred_raw_id = {1: chosen_players[0], 2: chosen_players[1]}
        last_bbox = {
            logical_id: player_detections[seed_frame][raw_id]
            for logical_id, raw_id in preferred_raw_id.items()
        }

        filtered_player_detections = []
        for frame_num, player_dict in enumerate(player_detections):
            if frame_num < seed_frame:
                filtered_player_detections.append({})
                continue

            assignments = self._assign_logical_players(
                player_dict,
                keypoints_per_frame[frame_num],
                preferred_raw_id,
                last_bbox,
            )
            logical_detections = {}
            for logical_id, raw_id in assignments.items():
                bbox = player_dict[raw_id]
                logical_detections[logical_id] = bbox
                preferred_raw_id[logical_id] = raw_id
                last_bbox[logical_id] = bbox
            filtered_player_detections.append(logical_detections)
        return filtered_player_detections

    @staticmethod
    def _normalize_keypoint_frames(court_keypoints, frame_count):
        keypoints = np.asarray(court_keypoints, dtype=float)
        if keypoints.ndim == 1:
            return [keypoints] * frame_count
        if keypoints.ndim != 2 or len(keypoints) != frame_count:
            raise ValueError(
                "Court keypoints must be one vector or one vector per frame"
            )
        return list(keypoints)

    def _assign_logical_players(
        self,
        player_dict,
        court_keypoints,
        preferred_raw_id,
        last_bbox,
    ):
        if not player_dict:
            return {}

        ranked_raw_ids = sorted(
            player_dict,
            key=lambda raw_id: min(
                self._court_role_cost(1, player_dict[raw_id], court_keypoints),
                self._court_role_cost(2, player_dict[raw_id], court_keypoints),
            ),
        )
        candidate_ids = set(ranked_raw_ids[:6])
        candidate_ids.update(
            raw_id
            for raw_id in preferred_raw_id.values()
            if raw_id in player_dict
        )
        candidate_ids = list(candidate_ids)

        logical_ids = [1, 2]
        if len(candidate_ids) == 1:
            raw_id = candidate_ids[0]
            logical_id = min(
                logical_ids,
                key=lambda identifier: self._assignment_cost(
                    identifier,
                    raw_id,
                    player_dict[raw_id],
                    court_keypoints,
                    preferred_raw_id,
                    last_bbox,
                ),
            )
            if self._court_role_cost(
                logical_id,
                player_dict[raw_id],
                court_keypoints,
            ) > 0.35:
                return {}
            return {logical_id: raw_id}

        best_cost = float("inf")
        best_assignment = {}
        for raw_pair in itertools.permutations(candidate_ids, 2):
            cost = sum(
                self._assignment_cost(
                    logical_id,
                    raw_id,
                    player_dict[raw_id],
                    court_keypoints,
                    preferred_raw_id,
                    last_bbox,
                )
                for logical_id, raw_id in zip(logical_ids, raw_pair)
            )
            if cost < best_cost:
                best_cost = cost
                best_assignment = dict(zip(logical_ids, raw_pair))
        return {
            logical_id: raw_id
            for logical_id, raw_id in best_assignment.items()
            if self._court_role_cost(
                logical_id,
                player_dict[raw_id],
                court_keypoints,
            )
            <= 0.35
        }

    def _assignment_cost(
        self,
        logical_id,
        raw_id,
        bbox,
        court_keypoints,
        preferred_raw_id,
        last_bbox,
    ):
        foot = np.asarray(get_foot_position(bbox), dtype=float)
        previous_bbox = last_bbox.get(logical_id)
        if previous_bbox is None:
            continuity_cost = 0.0
        else:
            previous_foot = np.asarray(
                get_foot_position(previous_bbox), dtype=float
            )
            reference_height = max(1.0, previous_bbox[3] - previous_bbox[1])
            continuity_cost = 0.10 * min(2.0, float(
                np.linalg.norm(foot - previous_foot) / reference_height
            ))

        court_position = self._normalized_court_position(
            foot,
            court_keypoints,
        )
        if court_position is None:
            side_cost = 0.0
            court_position_cost = 0.0
        else:
            court_x, court_y = court_position
            on_expected_side = (
                court_y <= 0.5 if logical_id == 1 else court_y >= 0.5
            )
            side_cost = 0.0 if on_expected_side else 4.0
            lateral_outside = max(0.0, -0.05 - court_x, court_x - 1.05)
            court_position_cost = (
                15.0 * lateral_outside
                + 0.15 * abs(court_x - 0.5)
            )

        # Raw tracker IDs are only a tie-breaker. Small/distant players often
        # lose their ID to a stationary line judge for several frames.
        raw_id_bonus = -0.02 if preferred_raw_id.get(logical_id) == raw_id else 0.0
        return (
            continuity_cost
            + side_cost
            + court_position_cost
            + raw_id_bonus
        )

    @classmethod
    def _court_role_cost(cls, logical_id, bbox, court_keypoints):
        position = cls._normalized_court_position(
            np.asarray(get_foot_position(bbox), dtype=float),
            court_keypoints,
        )
        if position is None:
            return cls._distance_to_court_keypoints(bbox, court_keypoints)
        court_x, court_y = position
        target_y = 0.0 if logical_id == 1 else 1.0
        lateral_outside = max(0.0, -0.05 - court_x, court_x - 1.05)
        wrong_half = (
            max(0.0, court_y - 0.5)
            if logical_id == 1
            else max(0.0, 0.5 - court_y)
        )
        return (
            15.0 * lateral_outside
            + 5.0 * wrong_half
            + 0.15 * abs(court_y - target_y)
            + 0.20 * abs(court_x - 0.5)
        )

    @staticmethod
    def _normalized_court_position(point, court_keypoints):
        points = np.asarray(court_keypoints, dtype=np.float32).reshape(-1, 2)
        if len(points) < 4 or not np.isfinite(points[:4]).all():
            return None
        source = points[:4]
        polygon = source[[0, 1, 3, 2]]
        if abs(cv2.contourArea(polygon)) < 1.0:
            return None
        destination = np.float32(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        )
        try:
            homography = cv2.getPerspectiveTransform(source, destination)
            projected = cv2.perspectiveTransform(
                np.float32(point).reshape(1, 1, 2),
                homography,
            )[0, 0]
        except cv2.error:
            return None
        if not np.isfinite(projected).all():
            return None
        return float(projected[0]), float(projected[1])

    @staticmethod
    def _distance_to_court_keypoints(bbox, court_keypoints):
        player_center = get_center_of_bbox(bbox)
        points = np.asarray(court_keypoints, dtype=float).reshape(-1, 2)
        if not len(points):
            return float("inf")
        return float(np.min(np.linalg.norm(points - player_center, axis=1)))

    def choose_players(self, court_keypoints, player_dict):
        if len(player_dict) < 2:
            raise ValueError("At least two player detections are required")

        best_pair = None
        best_cost = float("inf")
        for far_id, near_id in itertools.permutations(player_dict, 2):
            cost = self._court_role_cost(
                1, player_dict[far_id], court_keypoints
            ) + self._court_role_cost(
                2, player_dict[near_id], court_keypoints
            )
            if cost < best_cost:
                best_cost = cost
                best_pair = [far_id, near_id]
        return best_pair

    def detect_frames(self, frames):
        if not frames:
            return []
        selected_mode = self._select_detection_mode(frames)
        self.active_mode = selected_mode
        if selected_mode == "wheelchair":
            model = self.wheelchair_model
            confidence = self.WHEELCHAIR_CONFIDENCE
            image_size = self.WHEELCHAIR_IMAGE_SIZE
        else:
            model = self.standard_model
            confidence = self.STANDARD_CONFIDENCE
            image_size = None

        print(f"Using {selected_mode} player detector")
        results = self._predict_results(
            model,
            frames,
            confidence=confidence,
            image_size=image_size,
        )
        return [self._result_to_player_dict(result) for result in results]

    def _select_detection_mode(self, frames):
        if self.mode != "auto":
            return self.mode
        if self.wheelchair_model is None:
            return "standard"

        probe_count = min(24, len(frames))
        probe_indices = np.unique(
            np.linspace(0, len(frames) - 1, probe_count, dtype=int)
        )
        probe_frames = [frames[index] for index in probe_indices]
        wheelchair_results = self._predict_results(
            self.wheelchair_model,
            probe_frames,
            confidence=self.WHEELCHAIR_CONFIDENCE,
            image_size=self.WHEELCHAIR_IMAGE_SIZE,
        )
        standard_results = self._predict_results(
            self.standard_model,
            probe_frames,
            confidence=self.STANDARD_CONFIDENCE,
            image_size=None,
        )
        if self._looks_like_wheelchair_video(
            wheelchair_results,
            standard_results,
        ):
            return "wheelchair"
        return "standard"

    @staticmethod
    def _predict_results(
        model,
        frames,
        confidence,
        image_size,
    ):
        if model is None:
            raise RuntimeError("The selected player detector is not loaded")
        results = []
        # Ultralytics attempts to stack an entire in-memory list even with
        # stream=True. Small batches keep long point videos within GPU memory.
        for start in range(0, len(frames), 8):
            predict_options = {
                "classes": [0],
                "conf": confidence,
                "iou": 0.50,
                "max_det": 12 if image_size is not None else 50,
                "verbose": False,
            }
            if image_size is not None:
                predict_options["imgsz"] = image_size
            batch_results = model.predict(
                frames[start : start + 8],
                **predict_options,
            )
            results.extend(batch_results)
        return results

    @classmethod
    def _looks_like_wheelchair_video(
        cls,
        wheelchair_results,
        standard_results,
    ):
        """Conservatively distinguish chair boxes from standing people.

        The custom network can respond to a standing tennis player. A real
        athlete-and-chair annotation is normally wider and extends below the
        generic YOLO person box, or has no generic-person match because the
        chair/player silhouette is too different. Requiring this evidence for
        both competitors on several probe frames prevents auto mode from
        switching on ordinary tennis footage.
        """
        if not wheelchair_results or (
            len(wheelchair_results) != len(standard_results)
        ):
            return False

        frames_with_evidence = 0
        frames_with_two_players = 0
        for wheelchair_result, standard_result in zip(
            wheelchair_results,
            standard_results,
        ):
            wheelchair_boxes = cls._box_records(wheelchair_result)
            standard_boxes = cls._box_records(standard_result)
            evidence_count = sum(
                cls._has_wheelchair_geometry(candidate, standard_boxes)
                for candidate in wheelchair_boxes
            )
            if evidence_count:
                frames_with_evidence += 1
            if evidence_count >= 2:
                frames_with_two_players += 1

        probe_count = len(wheelchair_results)
        required_evidence_frames = min(
            probe_count,
            max(2, math.ceil(0.20 * probe_count)),
        )
        required_two_player_frames = min(
            probe_count,
            max(1, math.ceil(0.06 * probe_count)),
        )
        return (
            frames_with_evidence >= required_evidence_frames
            and frames_with_two_players >= required_two_player_frames
        )

    @staticmethod
    def _box_records(result):
        if result is None or result.boxes is None:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        return [
            {"bbox": np.asarray(bbox, dtype=float), "confidence": float(conf)}
            for bbox, conf in zip(boxes, confidences)
        ]

    @classmethod
    def _has_wheelchair_geometry(cls, candidate, standard_boxes):
        bbox = candidate["bbox"]
        confidence = candidate["confidence"]
        width = max(1.0, float(bbox[2] - bbox[0]))
        height = max(1.0, float(bbox[3] - bbox[1]))
        aspect_ratio = width / height

        matches = [
            (cls._bbox_iou(bbox, record["bbox"]), record["bbox"])
            for record in standard_boxes
        ]
        best_iou, person_bbox = max(
            matches,
            key=lambda match: match[0],
            default=(0.0, None),
        )
        if best_iou < 0.20 or person_bbox is None:
            return confidence >= 0.15 and 0.55 <= aspect_ratio <= 1.80

        person_width = max(1.0, float(person_bbox[2] - person_bbox[0]))
        person_height = max(1.0, float(person_bbox[3] - person_bbox[1]))
        area_ratio = (width * height) / (person_width * person_height)
        width_ratio = width / person_width
        bottom_extension = float(bbox[3] - person_bbox[3]) / person_height
        return (
            confidence >= 0.12
            and aspect_ratio >= 0.50
            and area_ratio >= 1.35
            and (width_ratio >= 1.25 or bottom_extension >= 0.10)
        )

    @staticmethod
    def _bbox_iou(first, second):
        left = max(float(first[0]), float(second[0]))
        top = max(float(first[1]), float(second[1]))
        right = min(float(first[2]), float(second[2]))
        bottom = min(float(first[3]), float(second[3]))
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, float(first[2] - first[0])) * max(
            0.0, float(first[3] - first[1])
        )
        second_area = max(0.0, float(second[2] - second[0])) * max(
            0.0, float(second[3] - second[1])
        )
        return intersection / max(
            1e-9,
            first_area + second_area - intersection,
        )

    def detect_frame(self, frame):
        selected_mode = self._select_detection_mode([frame])
        self.active_mode = selected_mode
        if selected_mode == "wheelchair":
            model = self.wheelchair_model
            confidence = self.WHEELCHAIR_CONFIDENCE
            image_size = self.WHEELCHAIR_IMAGE_SIZE
        else:
            model = self.standard_model
            confidence = self.STANDARD_CONFIDENCE
            image_size = None
        result = self._predict_results(
            model,
            [frame],
            confidence,
            image_size,
        )[0]
        return self._result_to_player_dict(result)

    @staticmethod
    def _result_to_player_dict(result):
        # IDs are deliberately frame-local. The court-aware assignment above
        # owns identity; relying on YOLO's raw tracker IDs caused a distant
        # player to disappear or inherit a line judge's identity.
        return {
            detection_index: box.xyxy.tolist()[0]
            for detection_index, box in enumerate(result.boxes)
        }

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # draw bounding boxes on the frame
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                side = "Far" if track_id == 1 else "Near"
                cv2.putText(frame, f"{side} player (P{track_id})", (int(bbox[0]), int(bbox[1])- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames
