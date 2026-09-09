"""Modern PyTorch datasets, model, and inference for TenniSet video labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.models.video import R3D_18_Weights, r3d_18

from .annotations import EVENT_CLASSES, REASON_CLASSES, TenniSetAnnotations


SIDE_TO_INDEX = {"far": 0, "near": 1}
INDEX_TO_SIDE = {value: key for key, value in SIDE_TO_INDEX.items()}


def _video_path(video_dir, video_id):
    video_dir = Path(video_dir)
    for suffix in (".mp4", ".avi", ".mkv", ".mov"):
        candidate = video_dir / f"{video_id}{suffix}"
        if candidate.is_file():
            return candidate
    return video_dir / f"{video_id}.mp4"


def read_clip(video_path, frame_indices, image_size=112):
    """Read selected frames as a normalized ``C,T,H,W`` float tensor."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(
            f"Missing TenniSet video {video_path}. Download the video archive "
            "or point --video-dir at V006.mp4 ... V010.mp4."
        )
    indices = [max(0, int(index)) for index in frame_indices]
    if not indices:
        raise ValueError("At least one frame index is required")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    frames = []
    previous_index = None
    try:
        for frame_index in indices:
            if previous_index is None or frame_index != previous_index + 1:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise OSError(
                    f"Could not read frame {frame_index} from {video_path}"
                )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            scale = (image_size + 16) / min(height, width)
            resized = cv2.resize(
                frame,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
            y0 = max(0, (resized.shape[0] - image_size) // 2)
            x0 = max(0, (resized.shape[1] - image_size) // 2)
            frames.append(resized[y0 : y0 + image_size, x0 : x0 + image_size])
            previous_index = frame_index
    finally:
        capture.release()

    values = np.stack(frames).astype(np.float32) / 255.0
    values = (values - np.asarray((0.43216, 0.394666, 0.37645))) / np.asarray(
        (0.22803, 0.22145, 0.216989)
    )
    return torch.from_numpy(values.astype(np.float32)).permute(3, 0, 1, 2)


class EventWindowDataset(Dataset):
    """Balanced serve/hit/background windows from strong TenniSet labels."""

    def __init__(
        self,
        annotations: TenniSetAnnotations,
        video_dir,
        video_ids=None,
        clip_frames=16,
        image_size=112,
        background_ratio=1.0,
        training=False,
    ):
        self.video_dir = Path(video_dir)
        self.clip_frames = int(clip_frames)
        self.image_size = int(image_size)
        self.training = bool(training)
        if self.clip_frames < 4:
            raise ValueError("clip_frames must be at least 4")
        allowed = set(video_ids or annotations.video_ids)
        events = annotations.event_rows(allowed)
        self.samples = [
            {
                "video_id": event.video_id,
                "center": event.center_frame,
                "label": EVENT_CLASSES.index(event.event_class),
            }
            for event in events
        ]

        exclusion = {}
        for event in events:
            exclusion.setdefault(event.video_id, []).append(
                (event.start_frame, event.end_frame)
            )
        background = []
        margin = max(2, self.clip_frames // 3)
        stride = max(8, self.clip_frames)
        for point in annotations.point_rows(allowed):
            for center in range(point.start_frame + margin, point.end_frame, stride):
                overlaps = any(
                    start - margin <= center <= end + margin
                    for start, end in exclusion.get(point.video_id, ())
                )
                if not overlaps:
                    background.append(
                        {"video_id": point.video_id, "center": center, "label": 0}
                    )
        maximum_background = round(len(self.samples) * float(background_ratio))
        if maximum_background > 0 and len(background) > maximum_background:
            selected = np.linspace(
                0,
                len(background) - 1,
                maximum_background,
                dtype=int,
            )
            background = [background[index] for index in selected]
        self.samples.extend(background)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        center = sample["center"]
        if self.training:
            center += int(torch.randint(-2, 3, ()).item())
        first = center - self.clip_frames // 2
        indices = range(first, first + self.clip_frames)
        clip = read_clip(
            _video_path(self.video_dir, sample["video_id"]),
            indices,
            self.image_size,
        )
        return clip, torch.tensor(sample["label"], dtype=torch.long)


class PointOutcomeDataset(Dataset):
    """Sparse full-point clips with winner-side and weak reason targets."""

    def __init__(
        self,
        annotations: TenniSetAnnotations,
        video_dir,
        video_ids=None,
        clip_frames=32,
        image_size=112,
        minimum_reason_confidence=0.90,
    ):
        self.video_dir = Path(video_dir)
        self.clip_frames = int(clip_frames)
        self.image_size = int(image_size)
        allowed = set(video_ids or annotations.video_ids)
        self.samples = [
            point
            for point in annotations.point_rows(allowed)
            if point.winner_side in SIDE_TO_INDEX
            and point.coarse_reason in REASON_CLASSES
            and point.reason_confidence >= minimum_reason_confidence
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        point = self.samples[index]
        frame_indices = np.linspace(
            point.start_frame,
            point.end_frame,
            self.clip_frames,
        ).round().astype(int)
        clip = read_clip(
            _video_path(self.video_dir, point.video_id),
            frame_indices,
            self.image_size,
        )
        targets = {
            "winner": torch.tensor(
                SIDE_TO_INDEX[point.winner_side], dtype=torch.long
            ),
            "reason": torch.tensor(
                REASON_CLASSES.index(point.coarse_reason), dtype=torch.long
            ),
        }
        return clip, targets


class TennisTemporalNet(nn.Module):
    """Shared 3D video backbone with event and point-outcome heads."""

    def __init__(self, pretrained=False):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        self.backbone = r3d_18(weights=weights)
        feature_count = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.event_head = nn.Linear(feature_count, len(EVENT_CLASSES))
        self.winner_head = nn.Linear(feature_count, len(SIDE_TO_INDEX))
        self.reason_head = nn.Linear(feature_count, len(REASON_CLASSES))

    def forward(self, video):
        features = self.backbone(video)
        return {
            "event": self.event_head(features),
            "winner": self.winner_head(features),
            "reason": self.reason_head(features),
        }


@dataclass(frozen=True)
class ModelConfiguration:
    task: str
    clip_frames: int
    image_size: int


class TemporalPredictor:
    """Load a training checkpoint and emit tracker-side prediction records."""

    def __init__(self, checkpoint_path, device=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.configuration = ModelConfiguration(
            task=str(checkpoint.get("task", "event")),
            clip_frames=int(checkpoint.get("clip_frames", 16)),
            image_size=int(checkpoint.get("image_size", 112)),
        )
        self.model = TennisTemporalNet(pretrained=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict_events(
        self,
        frames,
        stride=2,
        batch_size=8,
        minimum_confidence=0.55,
    ):
        if self.configuration.task != "event":
            raise ValueError(
                "predict_events requires a checkpoint trained with --task event"
            )
        if not frames:
            return []
        centers = list(range(0, len(frames), max(1, int(stride))))
        predictions = []
        for offset in range(0, len(centers), batch_size):
            batch_centers = centers[offset : offset + batch_size]
            clips = [
                self._clip_from_memory(frames, center)
                for center in batch_centers
            ]
            output = self.model(torch.stack(clips).to(self.device))["event"]
            probabilities = output.softmax(dim=1).cpu()
            confidences, labels = probabilities.max(dim=1)
            for center, label_index, confidence in zip(
                batch_centers, labels.tolist(), confidences.tolist()
            ):
                label = EVENT_CLASSES[label_index]
                if label == "OTH" or confidence < minimum_confidence:
                    continue
                predictions.append(
                    {
                        "frame": center,
                        "label": label,
                        "confidence": round(float(confidence), 5),
                        "kind": "serve" if label.startswith("S") else "hit",
                        "player_side": "near" if label[1] == "N" else "far",
                    }
                )
        return predictions

    @torch.inference_mode()
    def predict_outcome(self, frames):
        if self.configuration.task != "outcome":
            raise ValueError(
                "predict_outcome requires a checkpoint trained with --task outcome"
            )
        if not frames:
            return None
        count = self.configuration.clip_frames
        indices = np.linspace(0, len(frames) - 1, count).round().astype(int)
        clip = self._frames_to_tensor([frames[index] for index in indices])
        output = self.model(clip.unsqueeze(0).to(self.device))
        winner_probabilities = output["winner"].softmax(dim=1)[0].cpu()
        reason_probabilities = output["reason"].softmax(dim=1)[0].cpu()
        winner_confidence, winner_index = winner_probabilities.max(dim=0)
        reason_confidence, reason_index = reason_probabilities.max(dim=0)
        confidence = float(torch.sqrt(winner_confidence * reason_confidence))
        return {
            "winner_side": INDEX_TO_SIDE[int(winner_index)],
            "reason": REASON_CLASSES[int(reason_index)],
            "confidence": round(confidence, 5),
            "winner_confidence": round(float(winner_confidence), 5),
            "reason_confidence": round(float(reason_confidence), 5),
            "source": "tenniset_temporal_model",
        }

    def _clip_from_memory(self, frames, center):
        count = self.configuration.clip_frames
        first = center - count // 2
        indices = [min(len(frames) - 1, max(0, first + i)) for i in range(count)]
        return self._frames_to_tensor([frames[index] for index in indices])

    def _frames_to_tensor(self, frames):
        prepared = []
        size = self.configuration.image_size
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            scale = (size + 16) / min(height, width)
            resized = cv2.resize(
                rgb,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
            y0 = max(0, (resized.shape[0] - size) // 2)
            x0 = max(0, (resized.shape[1] - size) // 2)
            prepared.append(resized[y0 : y0 + size, x0 : x0 + size])
        values = np.stack(prepared).astype(np.float32) / 255.0
        values = (values - np.asarray((0.43216, 0.394666, 0.37645))) / np.asarray(
            (0.22803, 0.22145, 0.216989)
        )
        return torch.from_numpy(values.astype(np.float32)).permute(3, 0, 1, 2)
