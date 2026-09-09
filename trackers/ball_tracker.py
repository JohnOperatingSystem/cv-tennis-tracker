from ultralytics import YOLO
import cv2
import numpy as np

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # Tennis balls are small, so keep a permissive detector threshold and
        # reject false positives using motion and temporal consistency below.
        self.detection_confidence = 0.05
        self.minimum_track_confidence = 0.10
        self.reacquire_confidence = 0.18
        self.nms_iou = 0.30
        self.motion_threshold = 0.025
        # The valid detections in the supplied 1080p/30 FPS rally move at
        # under 65 px/frame. Keep generous headroom while rejecting isolated
        # jumps to unrelated moving regions.
        self.max_jump_per_frame = 120.0
        self.max_tracking_gap = 6
        self.interpolation_max_gap = 12
        self.display_interpolation_max_gap = 40

    def configure_for_video(self, frame_height, fps):
        """Scale pixel and frame thresholds to the active video's geometry."""
        if frame_height <= 0 or fps <= 0:
            raise ValueError("Video height and FPS must be greater than zero")
        resolution_scale = frame_height / 720.0
        frame_rate_scale = 25.0 / fps
        self.max_jump_per_frame = 120.0 * resolution_scale * frame_rate_scale
        self.max_tracking_gap = max(2, round(0.24 * fps))
        # Keep event analysis conservative: long synthetic spans must not
        # create artificial acceleration peaks. Display interpolation can span
        # high-lob occlusions without feeding those boxes back into events.
        self.interpolation_max_gap = max(3, round(0.48 * fps))
        self.display_interpolation_max_gap = max(3, round(1.60 * fps))
        
    def interpolate_ball_positions(
        self,
        ball_positions,
        max_gap=12,
        maximum_average_step=120.0,
    ):
        """Interpolate bounded gaps whose endpoints imply plausible motion.

        Long gaps and gaps at the beginning/end remain empty. This prevents a
        false detection from being propagated across a large part of a rally.
        """
        interpolated = [dict(position) for position in ball_positions]
        valid_indices = [
            index for index, position in enumerate(ball_positions)
            if position.get(1)
        ]

        for start_index, end_index in zip(valid_indices, valid_indices[1:]):
            gap = end_index - start_index - 1
            if gap <= 0 or gap > max_gap:
                continue

            start_box = np.asarray(ball_positions[start_index][1], dtype=float)
            end_box = np.asarray(ball_positions[end_index][1], dtype=float)
            start_center = np.asarray(
                ((start_box[0] + start_box[2]) / 2,
                 (start_box[1] + start_box[3]) / 2),
                dtype=float,
            )
            end_center = np.asarray(
                ((end_box[0] + end_box[2]) / 2,
                 (end_box[1] + end_box[3]) / 2),
                dtype=float,
            )
            average_step = (
                np.linalg.norm(end_center - start_center)
                / (end_index - start_index)
            )
            if average_step > maximum_average_step:
                continue

            for offset in range(1, gap + 1):
                ratio = offset / (gap + 1)
                box = start_box + ratio * (end_box - start_box)
                interpolated[start_index + offset] = {1: box.tolist()}

        return interpolated

    def get_ball_shot_frames(self, ball_positions):
        mid_y = np.full(len(ball_positions), np.nan, dtype=float)
        for frame_num, position in enumerate(ball_positions):
            bbox = position.get(1)
            if bbox is not None:
                mid_y[frame_num] = (bbox[1] + bbox[3]) / 2.0
        rolling_mean = np.full(len(mid_y), np.nan, dtype=float)
        for frame_num in range(len(mid_y)):
            window = mid_y[max(0, frame_num - 4) : frame_num + 1]
            if np.isfinite(window).any():
                rolling_mean[frame_num] = np.nanmean(window)
        delta_y = np.diff(rolling_mean, prepend=np.nan)
        ball_hit = np.zeros(len(ball_positions), dtype=bool)
        minimum_change_frames_for_hit = 25
        for i in range(1, len(ball_positions)-int(minimum_change_frames_for_hit*1.2)):
            negative_position_change = delta_y[i] > 0 and delta_y[i + 1] < 0
            positive_position_change = delta_y[i] < 0 and delta_y[i + 1] > 0
            
            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = delta_y[i] > 0 and delta_y[change_frame] < 0
                    positive_position_change_following_frame = delta_y[i] < 0 and delta_y[change_frame] > 0
                    
                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1
                
                if change_count > minimum_change_frames_for_hit-1:
                    ball_hit[i] = True
                    
        return np.flatnonzero(ball_hit).tolist()

    def align_predicted_boxes_to_trajectory(
        self,
        ball_detections,
        ball_mini_court_positions,
        ball_heights,
        player_detections,
        player_mini_court_positions,
        court_keypoints_per_frame,
        mini_court,
        predicted_frame_numbers,
        player_height_meters=1.85,
    ):
        """Re-center inferred-flight boxes using the reconstructed 3-D path.

        The court homography supplies the image location directly beneath the
        ball. Its height is converted to an image-space vertical offset using
        the detected players as per-frame depth/scale references. This is most
        important near an inferred landing, where linear image interpolation
        can otherwise put the box several dozen pixels away from the bounce.
        """
        aligned = [dict(detection) for detection in ball_detections]
        predicted_frame_numbers = set(predicted_frame_numbers)

        for frame_num in predicted_frame_numbers:
            if not 0 <= frame_num < len(aligned):
                continue
            bbox = aligned[frame_num].get(1)
            ball_position = ball_mini_court_positions[frame_num].get(1)
            height = ball_heights[frame_num]
            if bbox is None or ball_position is None or height is None:
                continue

            try:
                image_to_mini = mini_court.get_court_homography(
                    court_keypoints_per_frame[frame_num]
                )
                mini_to_image = np.linalg.inv(image_to_mini)
                ground_point = cv2.perspectiveTransform(
                    np.float32(ball_position).reshape(1, 1, 2),
                    mini_to_image,
                )[0, 0]
            except (ValueError, np.linalg.LinAlgError, cv2.error):
                continue

            pixels_per_meter = self._estimate_vertical_pixels_per_meter(
                ball_position,
                player_detections[frame_num],
                player_mini_court_positions[frame_num],
                player_height_meters,
            )
            center_x = float(ground_point[0])
            center_y = float(ground_point[1] - height * pixels_per_meter)

            x1, y1, x2, y2 = np.asarray(bbox, dtype=float)
            width = max(2.0, x2 - x1)
            box_height = max(2.0, y2 - y1)
            aligned[frame_num] = {1: [
                center_x - width / 2,
                center_y - box_height / 2,
                center_x + width / 2,
                center_y + box_height / 2,
            ]}

        return aligned

    def project_trajectory_to_image(
        self,
        ball_mini_court_positions,
        ball_heights,
        player_detections,
        player_mini_court_positions,
        court_keypoints_per_frame,
        mini_court,
        player_height_meters=1.85,
    ):
        """Project the reconstructed 3-D ball centre into every video frame."""
        frame_count = len(ball_mini_court_positions)
        if not (
            len(ball_heights)
            == len(player_detections)
            == len(player_mini_court_positions)
            == len(court_keypoints_per_frame)
            == frame_count
        ):
            raise ValueError("All projection inputs must contain the same frames")

        projected_centers = [None] * frame_count
        for frame_num in range(frame_count):
            ball_position = ball_mini_court_positions[frame_num].get(1)
            height = ball_heights[frame_num]
            if ball_position is None or height is None:
                continue
            try:
                image_to_mini = mini_court.get_court_homography(
                    court_keypoints_per_frame[frame_num]
                )
                mini_to_image = np.linalg.inv(image_to_mini)
                ground_point = cv2.perspectiveTransform(
                    np.float32(ball_position).reshape(1, 1, 2),
                    mini_to_image,
                )[0, 0]
            except (ValueError, np.linalg.LinAlgError, cv2.error):
                continue

            pixels_per_meter = self._estimate_vertical_pixels_per_meter(
                ball_position,
                player_detections[frame_num],
                player_mini_court_positions[frame_num],
                player_height_meters,
            )
            projected_centers[frame_num] = (
                float(ground_point[0]),
                float(ground_point[1] - height * pixels_per_meter),
            )
        return projected_centers

    @staticmethod
    def estimate_bounded_gap_image_centers(
        observed_ball_detections,
        projected_image_centers,
        maximum_gap_frames,
        sample_frames=5,
    ):
        """Estimate only the image path needed to classify a returning gap.

        This quadratic fit is deliberately not promoted to a ball detection,
        court trajectory, height, or speed measurement. It lets the visibility
        classifier recognize an above-frame lob when the 3-D event model has
        no arc, while a real observation on both sides keeps the estimate
        bounded.
        """
        frame_count = len(observed_ball_detections)
        if len(projected_image_centers) != frame_count:
            raise ValueError("A projected centre is required for every frame")
        if maximum_gap_frames < 1:
            raise ValueError("Maximum gap must be at least one frame")
        if sample_frames < 2:
            raise ValueError("At least two boundary samples are required")

        centers = list(projected_image_centers)

        def observed_center(frame_num):
            bbox = observed_ball_detections[frame_num].get(1)
            if bbox is None:
                return None
            return np.asarray(
                (
                    (bbox[0] + bbox[2]) / 2.0,
                    (bbox[1] + bbox[3]) / 2.0,
                ),
                dtype=float,
            )

        observed = [bool(frame.get(1)) for frame in observed_ball_detections]
        frame_num = 0
        while frame_num < frame_count:
            if observed[frame_num]:
                frame_num += 1
                continue
            gap_start = frame_num
            while frame_num < frame_count and not observed[frame_num]:
                frame_num += 1
            gap_end = frame_num - 1
            gap_length = gap_end - gap_start + 1
            previous_frame = gap_start - 1
            next_frame = frame_num
            if (
                previous_frame < 0
                or next_frame >= frame_count
                or gap_length > maximum_gap_frames
            ):
                continue

            left_samples = []
            for index in range(
                previous_frame,
                max(-1, previous_frame - sample_frames),
                -1,
            ):
                center = observed_center(index)
                if center is not None:
                    left_samples.append((index, center))
            right_samples = []
            for index in range(
                next_frame,
                min(frame_count, next_frame + sample_frames),
            ):
                center = observed_center(index)
                if center is not None:
                    right_samples.append((index, center))
            if len(left_samples) < 2 or len(right_samples) < 2:
                continue

            samples = list(reversed(left_samples)) + right_samples
            sample_indices = np.asarray(
                [sample[0] for sample in samples], dtype=float
            )
            sample_x = np.asarray(
                [sample[1][0] for sample in samples], dtype=float
            )
            sample_y = np.asarray(
                [sample[1][1] for sample in samples], dtype=float
            )
            center_frame = 0.5 * (previous_frame + next_frame)
            local_indices = sample_indices - center_frame
            x_coefficients = np.polyfit(local_indices, sample_x, 1)
            y_coefficients = np.polyfit(local_indices, sample_y, 2)

            for index in range(gap_start, gap_end + 1):
                if centers[index] is not None:
                    continue
                local_index = index - center_frame
                centers[index] = (
                    float(np.polyval(x_coefficients, local_index)),
                    float(np.polyval(y_coefficients, local_index)),
                )

        return centers

    @staticmethod
    def classify_visibility(
        observed_ball_detections,
        interpolated_ball_detections,
        projected_image_centers,
        frame_shape,
    ):
        """Classify detector gaps without confusing high lobs with lost balls.

        An ``out_of_frame`` interval must be bracketed by real detections near
        the top of the image. The reconstructed path must rise above the top
        edge and have its apex between those two observations. This offline,
        bidirectional check prevents a one-sided detector miss from creating a
        speculative off-screen trajectory.
        """
        frame_count = len(observed_ball_detections)
        if not (
            len(interpolated_ball_detections)
            == len(projected_image_centers)
            == frame_count
        ):
            raise ValueError("All visibility inputs must contain the same frames")
        if len(frame_shape) < 2:
            raise ValueError("Frame shape must contain height and width")
        frame_height, frame_width = frame_shape[:2]
        if frame_height <= 0 or frame_width <= 0:
            raise ValueError("Frame dimensions must be greater than zero")

        observed = [bool(frame.get(1)) for frame in observed_ball_detections]
        states = ["unknown"] * frame_count
        for frame_num in range(frame_count):
            if observed[frame_num]:
                states[frame_num] = "observed"
            elif interpolated_ball_detections[frame_num].get(1):
                states[frame_num] = "interpolated"

        def detection_center(frame_num):
            bbox = observed_ball_detections[frame_num].get(1)
            if bbox is None:
                return None
            return (
                float((bbox[0] + bbox[2]) / 2.0),
                float((bbox[1] + bbox[3]) / 2.0),
            )

        frame_num = 0
        top_gate = 0.40 * frame_height
        minimum_apex_change = 0.06 * frame_height
        horizontal_margin = 0.15 * frame_width
        while frame_num < frame_count:
            if observed[frame_num]:
                frame_num += 1
                continue
            gap_start = frame_num
            while frame_num < frame_count and not observed[frame_num]:
                frame_num += 1
            gap_end = frame_num - 1
            previous_frame = gap_start - 1
            next_frame = frame_num
            if previous_frame < 0 or next_frame >= frame_count:
                continue

            exit_center = detection_center(previous_frame)
            return_center = detection_center(next_frame)
            if exit_center is None or return_center is None:
                continue

            valid_projected = [
                (index, projected_image_centers[index])
                for index in range(gap_start, gap_end + 1)
                if projected_image_centers[index] is not None
            ]
            if not valid_projected:
                continue

            # A bounded in-frame miss is an occlusion, not an extrapolated
            # visible box. Short detector interpolation retains its stronger
            # state unless the physical path proves that it is above frame.
            for index, center in valid_projected:
                x, y = center
                if 0.0 <= x < frame_width and 0.0 <= y < frame_height:
                    if states[index] == "unknown":
                        states[index] = "occluded"

            path = [(previous_frame, exit_center)]
            path.extend(valid_projected)
            path.append((next_frame, return_center))
            apex_index, apex_center = min(path, key=lambda item: item[1][1])
            apex_x, apex_y = apex_center
            credible_high_lob = (
                exit_center[1] <= top_gate
                and return_center[1] <= top_gate
                and gap_start <= apex_index <= gap_end
                and apex_y < 0.0
                and exit_center[1] - apex_y >= minimum_apex_change
                and return_center[1] - apex_y >= minimum_apex_change
                and -horizontal_margin
                <= apex_x
                <= frame_width + horizontal_margin
            )
            if not credible_high_lob:
                continue

            for index, center in valid_projected:
                x, y = center
                if (
                    y < 0.0
                    and -horizontal_margin
                    <= x
                    <= frame_width + horizontal_margin
                ):
                    states[index] = "out_of_frame"

        return states

    @staticmethod
    def _estimate_vertical_pixels_per_meter(
        ball_position,
        player_detections,
        player_mini_court_positions,
        player_height_meters,
    ):
        """Interpolate image scale by court depth from visible players."""
        scale_samples = []
        for player_id, player_position in player_mini_court_positions.items():
            bbox = player_detections.get(player_id)
            if bbox is None:
                continue
            player_pixel_height = max(1.0, float(bbox[3] - bbox[1]))
            scale_samples.append((
                float(player_position[1]),
                player_pixel_height / player_height_meters,
            ))

        if not scale_samples:
            return 45.0
        if len(scale_samples) == 1:
            return scale_samples[0][1]

        scale_samples.sort(key=lambda sample: sample[0])
        depth_values = np.asarray(
            [sample[0] for sample in scale_samples],
            dtype=float,
        )
        pixel_scales = np.asarray(
            [sample[1] for sample in scale_samples],
            dtype=float,
        )
        return float(np.interp(
            float(ball_position[1]),
            depth_values,
            pixel_scales,
        ))
    
    def detect_frames(self, frames, court_keypoints_per_frame=None):
        if not frames:
            return []
        if (
            court_keypoints_per_frame is not None
            and len(court_keypoints_per_frame) != len(frames)
        ):
            raise ValueError("Court keypoints must correspond to every video frame")

        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        candidates_by_frame = [
            self._detect_candidates(
                frame,
                None
                if court_keypoints_per_frame is None
                else court_keypoints_per_frame[frame_num],
            )
            for frame_num, frame in enumerate(frames)
        ]
        ball_detections = self._select_temporal_trajectory(
            candidates_by_frame,
            gray_frames,
        )
        return ball_detections

    def detect_frame(self, frame):
        """Single-frame fallback; video tracking should use detect_frames."""
        candidates = self._detect_candidates(frame)
        if not candidates:
            return {}

        best_candidate = max(candidates, key=lambda candidate: candidate['confidence'])
        return {1: best_candidate['bbox']}

    def _detect_candidates(self, frame, court_keypoints=None):
        results = self.model.predict(
            frame,
            conf=self.detection_confidence,
            iou=self.nms_iou,
            verbose=False,
        )[0]

        candidates = []
        for box in results.boxes:
            bbox = box.xyxy.tolist()[0]
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = bbox
            candidate = {
                'bbox': bbox,
                'confidence': confidence,
                'center': np.asarray(
                    [(x1 + x2) / 2, (y1 + y2) / 2],
                    dtype=float,
                ),
            }
            if court_keypoints is not None and not self._near_playing_area(
                candidate['center'],
                court_keypoints,
                frame.shape,
            ):
                continue
            candidates.append(candidate)

        return candidates

    @staticmethod
    def _near_playing_area(center, court_keypoints, frame_shape):
        points = np.asarray(court_keypoints, dtype=np.float32).reshape(-1, 2)
        if len(points) < 4 or not np.isfinite(points[:4]).all():
            return True
        polygon = points[[0, 1, 3, 2]]
        court_width = max(
            np.linalg.norm(points[1] - points[0]),
            np.linalg.norm(points[3] - points[2]),
        )
        # Keep legal wide/long landings and high balls, while excluding crowd,
        # scoreboard, and courtside false positives far from the playing area.
        margin = max(35.0, 0.10 * court_width)
        signed_distance = cv2.pointPolygonTest(
            polygon,
            (float(center[0]), float(center[1])),
            True,
        )
        if signed_distance >= -margin:
            return True
        height, width = frame_shape[:2]
        return (
            -margin <= center[0] <= width + margin
            and -margin <= center[1] <= height + margin
            and signed_distance >= -1.75 * margin
        )

    def _select_temporal_trajectory(self, candidates_by_frame, gray_frames):
        detections = []
        last_center = None
        last_frame_index = None
        velocity = None

        for frame_index, candidates in enumerate(candidates_by_frame):
            moving_candidates = []
            for candidate in candidates:
                if candidate['confidence'] < self.minimum_track_confidence:
                    continue
                motion = self._measure_candidate_motion(
                    candidate['bbox'],
                    frame_index,
                    gray_frames,
                )
                if motion < self.motion_threshold:
                    continue

                candidate = dict(candidate)
                candidate['motion'] = motion
                # Detector confidence is substantially more discriminative on
                # this model than raw patch motion (players and score graphics
                # can have much stronger motion than a five-pixel ball).
                candidate['base_score'] = candidate['confidence'] * (
                    0.65 + 0.35 * motion
                )
                moving_candidates.append(candidate)

            selected = None
            if moving_candidates:
                gap = (
                    frame_index - last_frame_index
                    if last_frame_index is not None
                    else None
                )

                if last_center is None or gap > self.max_tracking_gap:
                    reacquired = max(
                        moving_candidates,
                        key=lambda candidate: candidate['base_score'],
                    )
                    if reacquired['confidence'] >= self.reacquire_confidence:
                        selected = reacquired
                        velocity = None
                else:
                    predicted_center = last_center.copy()
                    if velocity is not None:
                        predicted_center += velocity * gap

                    plausible_candidates = []
                    for candidate in moving_candidates:
                        distance = float(np.linalg.norm(
                            candidate['center'] - predicted_center
                        ))
                        if distance > self.max_jump_per_frame * gap:
                            continue

                        continuity = np.exp(
                            -0.5 * (distance / (90.0 * gap)) ** 2
                        )
                        score = candidate['base_score'] * (
                            0.25 + 0.75 * continuity
                        )
                        plausible_candidates.append((score, candidate))

                    if plausible_candidates:
                        selected = max(
                            plausible_candidates,
                            key=lambda item: item[0],
                        )[1]

            if selected is None:
                detections.append({})
                continue

            if last_center is not None and last_frame_index is not None:
                gap = frame_index - last_frame_index
                observed_velocity = (selected['center'] - last_center) / gap
                if velocity is None:
                    velocity = observed_velocity
                else:
                    velocity = 0.5 * velocity + 0.5 * observed_velocity

            last_center = selected['center']
            last_frame_index = frame_index
            detections.append({1: selected['bbox']})

        return detections

    def _measure_candidate_motion(self, bbox, frame_index, gray_frames):
        current = gray_frames[frame_index]
        previous = gray_frames[max(0, frame_index - 1)]
        following = gray_frames[min(len(gray_frames) - 1, frame_index + 1)]

        difference = np.maximum(
            cv2.absdiff(current, previous),
            cv2.absdiff(current, following),
        )

        x1, y1, x2, y2 = bbox
        padding = 5
        left = max(0, int(x1) - padding)
        top = max(0, int(y1) - padding)
        right = min(difference.shape[1], int(x2) + padding)
        bottom = min(difference.shape[0], int(y2) + padding)
        patch = difference[top:bottom, left:right]

        if patch.size == 0:
            return 0.0

        return float(np.mean(patch > 20))
    
    def draw_bboxes(
        self,
        video_frames,
        ball_detections,
        observed_frame_numbers=None,
        short_interpolated_frame_numbers=None,
        predicted_high_arc_frame_numbers=None,
        visibility_states=None,
        projected_image_centers=None,
    ):
        """Draw detector, interpolation, and out-of-frame states distinctly.

        A normal bounding box is only truthful while the ball is in the image.
        During a reconstructed high lob the ball can be above the camera's
        field of view, so drawing a box on the court would falsely imply that
        the pixels inside it contain the ball. Those frames receive a top-edge
        direction marker instead.
        """
        observed_frame_numbers = (
            None
            if observed_frame_numbers is None
            else set(observed_frame_numbers)
        )
        short_interpolated_frame_numbers = set(
            short_interpolated_frame_numbers or ()
        )
        predicted_high_arc_frame_numbers = set(
            predicted_high_arc_frame_numbers or ()
        )
        if visibility_states is not None and len(visibility_states) != len(
            video_frames
        ):
            raise ValueError("Visibility state is required for every frame")
        if projected_image_centers is not None and len(
            projected_image_centers
        ) != len(video_frames):
            raise ValueError("Projected ball centre is required for every frame")
        output_video_frames = []
        for frame_num, (frame, ball_dict) in enumerate(
            zip(video_frames, ball_detections)
        ):
            visibility = (
                None if visibility_states is None else visibility_states[frame_num]
            )
            is_above_frame = (
                visibility == "out_of_frame"
                or frame_num in predicted_high_arc_frame_numbers
            )
            if is_above_frame:
                projected_center = (
                    None
                    if projected_image_centers is None
                    else projected_image_centers[frame_num]
                )
                bbox = ball_dict.get(1)
                if projected_center is not None:
                    center_x = int(np.clip(
                        projected_center[0], 24, frame.shape[1] - 24
                    ))
                elif bbox is not None:
                    center_x = int(np.clip(
                        (bbox[0] + bbox[2]) / 2,
                        24,
                        frame.shape[1] - 24,
                    ))
                else:
                    center_x = frame.shape[1] // 2
                marker_color = (0, 180, 255)
                cv2.arrowedLine(
                    frame,
                    (center_x, 28),
                    (center_x, 3),
                    marker_color,
                    2,
                    cv2.LINE_AA,
                    tipLength=0.35,
                )
                label = "BALL ABOVE FRAME (PREDICTED)"
                text_size = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    2,
                )[0]
                label_x = int(np.clip(
                    center_x - text_size[0] / 2,
                    4,
                    frame.shape[1] - text_size[0] - 4,
                ))
                cv2.putText(
                    frame,
                    label,
                    (label_x, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    marker_color,
                    2,
                    cv2.LINE_AA,
                )
                output_video_frames.append(frame)
                continue

            if visibility in ("occluded", "unknown"):
                output_video_frames.append(frame)
                continue

            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox

                is_observed = (
                    observed_frame_numbers is None
                    or frame_num in observed_frame_numbers
                )
                if is_observed:
                    color = (0, 255, 0)
                    label = f"Ball detected: {track_id}"
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        2,
                    )
                else:
                    is_short_gap = (
                        frame_num in short_interpolated_frame_numbers
                    )
                    color = (255, 255, 0) if is_short_gap else (0, 180, 255)
                    label = (
                        "Ball interpolated"
                        if is_short_gap
                        else "Ball predicted"
                    )
                    self._draw_dashed_rectangle(frame, bbox, color)

                label_y = max(18, int(y1) - 8)
                cv2.putText(
                    frame,
                    label,
                    (max(2, int(x1)), label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            output_video_frames.append(frame)

        return output_video_frames

    @staticmethod
    def _draw_dashed_rectangle(frame, bbox, color, dash_length=5):
        """Draw a dashed box so a prediction cannot look detector-verified."""
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)

        def draw_dashed_line(start, end):
            start = np.asarray(start, dtype=float)
            end = np.asarray(end, dtype=float)
            length = float(np.linalg.norm(end - start))
            if length == 0:
                return
            direction = (end - start) / length
            for distance in np.arange(0, length, dash_length * 2):
                segment_end = min(distance + dash_length, length)
                point_a = start + direction * distance
                point_b = start + direction * segment_end
                cv2.line(
                    frame,
                    tuple(np.rint(point_a).astype(int)),
                    tuple(np.rint(point_b).astype(int)),
                    color,
                    2,
                    cv2.LINE_AA,
                )

        draw_dashed_line((x1, y1), (x2, y1))
        draw_dashed_line((x2, y1), (x2, y2))
        draw_dashed_line((x2, y2), (x1, y2))
        draw_dashed_line((x1, y2), (x1, y1))
