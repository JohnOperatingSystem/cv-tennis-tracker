import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import torchvision.models as models

class CourtLineDetector:
    def __init__(self, model_path, device=None):
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.model = models.resnet50(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)  # Assuming binary classification: court line or not
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def predict(self, image):

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = img_rgb.shape[:2]

        keypoints[::2] *= original_w/224.0
        keypoints[1::2] *= original_h/224.0

        return keypoints

    def predict_frames(self, video_frames):
        return [self.predict(frame) for frame in video_frames]

    @staticmethod
    def is_plausible_full_court(keypoints, frame_shape):
        """Reject keypoint hallucinations on close-ups and crowd shots."""
        points = np.asarray(keypoints, dtype=float).reshape(-1, 2)
        if len(points) < 4 or not np.isfinite(points).all():
            return False

        height, width = frame_shape[:2]
        if height <= 0 or width <= 0:
            return False
        inside = (
            (points[:, 0] >= -0.05 * width)
            & (points[:, 0] <= 1.05 * width)
            & (points[:, 1] >= -0.05 * height)
            & (points[:, 1] <= 1.05 * height)
        )
        corners = points[:4]
        polygon = np.float32(corners[[0, 1, 3, 2]])
        area_ratio = abs(float(cv2.contourArea(polygon))) / (width * height)
        far_y = float(np.mean(corners[:2, 1]))
        near_y = float(np.mean(corners[2:, 1]))
        depth_ratio = (near_y - far_y) / height
        far_width = float(np.linalg.norm(corners[1] - corners[0]))
        near_width = float(np.linalg.norm(corners[3] - corners[2]))
        center_x = float(np.mean(corners[:, 0]))
        return bool(
            np.mean(inside) >= 0.85
            and 0.12 <= area_ratio <= 0.55
            and depth_ratio >= 0.32
            and far_width <= 0.46 * width
            and near_width >= 0.52 * width
            and near_width / max(1.0, far_width) >= 1.75
            and abs(center_x - 0.5 * width) <= 0.18 * width
            and far_y <= 0.43 * height
            and near_y >= 0.70 * height
        )

    @staticmethod
    def stabilize_keypoints(
        keypoints_per_frame,
        assume_fixed_camera=True,
        window_size=9,
    ):
        """Suppress detection jitter without discarding real camera motion.

        Fixed-camera clips use one robust median calibration. Moving-camera
        clips use a centered rolling median, leaving a distinct court geometry
        for every frame so its homography follows pans, tilts, and zooms.
        """
        points = np.asarray(keypoints_per_frame, dtype=float)
        if points.ndim != 2 or not len(points):
            raise ValueError("Court keypoints must be a non-empty 2-D array")

        if assume_fixed_camera:
            stable = np.nanmedian(points, axis=0)
            return [stable.copy() for _ in range(len(points))]

        window_size = max(1, int(window_size))
        if window_size % 2 == 0:
            window_size += 1
        radius = window_size // 2
        stabilized = []
        for frame_num in range(len(points)):
            start = max(0, frame_num - radius)
            end = min(len(points), frame_num + radius + 1)
            stabilized.append(np.nanmedian(points[start:end], axis=0))
        return stabilized

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image

    def draw_keypoints_on_video(self, video_frames, keypoints_per_frame):
        if len(video_frames) != len(keypoints_per_frame):
            raise ValueError(
                "Each video frame must have corresponding court keypoints"
            )

        output_video_frames = []
        for frame, keypoints in zip(video_frames, keypoints_per_frame):
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames
