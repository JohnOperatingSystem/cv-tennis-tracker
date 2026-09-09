"""Exact-frame tennis bounce spotting with the pretrained E2E-Spot model.

The checkpoint distributed by the E2E-Spot authors was trained on six tennis
events: near/far serve, swing, and bounce.  This adapter keeps the published
RegNetY-200MF + gated temporal shift + bidirectional GRU architecture, while
providing direct inference on the OpenCV frames already loaded by this app.
"""

from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


TENNIS_EVENT_CLASSES = (
    "background",
    "far_court_bounce",
    "far_court_swing",
    "far_court_serve",
    "near_court_bounce",
    "near_court_swing",
    "near_court_serve",
)
BOUNCE_CLASS_INDICES = (1, 4)


class _GatedShiftModule(nn.Module):
    """Device-safe implementation of the GSM block used by E2E-Spot."""

    def __init__(self, channels, segment_count):
        super().__init__()
        self.channels = int(channels)
        self.segment_count = int(segment_count)
        self.conv3D = nn.Conv3d(
            self.channels,
            2,
            (3, 3, 3),
            stride=1,
            padding=(1, 1, 1),
            groups=2,
        )
        self.bn = nn.BatchNorm3d(self.channels)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    @staticmethod
    def _shift_left(values):
        return torch.cat((values[:, :, 1:], torch.zeros_like(values[:, :, :1])), 2)

    @staticmethod
    def _shift_right(values):
        return torch.cat((torch.zeros_like(values[:, :, :1]), values[:, :, :-1]), 2)

    def forward(self, values):
        batch_size = values.size(0) // self.segment_count
        channels, height, width = values.shape[1:]
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} GSM channels, got {channels}")
        sequence = values.view(
            batch_size, self.segment_count, channels, height, width
        ).permute(0, 2, 1, 3, 4).contiguous()
        gate = self.tanh(self.conv3D(self.relu(self.bn(sequence))))
        first, second = sequence.chunk(2, dim=1)
        first_gate = gate[:, 0:1]
        second_gate = gate[:, 1:2]
        first_shift = first_gate * first
        second_shift = second_gate * second
        first = self._shift_left(first_shift) + first - first_shift
        second = self._shift_right(second_shift) + second - second_shift

        # Interleave the two channel groups exactly as in the published GSM.
        quarter = channels // 4
        first = first.view(
            batch_size, 2, quarter, self.segment_count, height, width
        ).permute(0, 2, 1, 3, 4, 5)
        second = second.view(
            batch_size, 2, quarter, self.segment_count, height, width
        ).permute(0, 2, 1, 3, 4, 5)
        merged = torch.cat(
            (
                first.contiguous().view(
                    batch_size, channels // 2, self.segment_count, height, width
                ),
                second.contiguous().view(
                    batch_size, channels // 2, self.segment_count, height, width
                ),
            ),
            dim=1,
        )
        return merged.permute(0, 2, 1, 3, 4).contiguous().view_as(values)


class _GatedShift(nn.Module):
    def __init__(self, layer, segment_count, division=4):
        super().__init__()
        convolution = getattr(layer, "conv", None)
        if convolution is None:
            raise TypeError(f"Unsupported RegNet convolution block: {type(layer)!r}")
        input_channels = int(convolution.in_channels)
        shifted_channels = math.ceil((input_channels // division) / 4) * 4
        shifted_channels = max(4, shifted_channels)
        self.gsm = _GatedShiftModule(shifted_channels, segment_count)
        self.net = layer
        self.shifted_channels = shifted_channels

    def forward(self, values):
        shifted = values.clone()
        shifted[:, : self.shifted_channels] = self.gsm(
            values[:, : self.shifted_channels]
        )
        return self.net(shifted)


def _add_gated_temporal_shift(regnet, clip_frames):
    for stage in (regnet.s1, regnet.s2, regnet.s3, regnet.s4):
        for block in stage.children():
            block.conv1 = _GatedShift(block.conv1, clip_frames)


class TennisBounceSpotterNet(nn.Module):
    """Published E2E-Spot tennis architecture compatible with its checkpoint."""

    def __init__(self, clip_frames=100):
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise ImportError(
                "Bounce spotting requires timm. Install it with "
                "`python -m pip install timm`."
            ) from error

        self.clip_frames = int(clip_frames)
        self._features = timm.create_model("regnety_002", pretrained=False)
        feature_count = int(self._features.head.fc.in_features)
        self._features.head.fc = nn.Identity()
        _add_gated_temporal_shift(self._features, self.clip_frames)
        self._pred_fine = nn.Module()
        self._pred_fine._gru = nn.GRU(
            feature_count,
            feature_count,
            batch_first=True,
            bidirectional=True,
        )
        self._pred_fine._dropout = nn.Dropout()
        self._pred_fine._fc_out = nn.Module()
        self._pred_fine._fc_out._fc_out = nn.Linear(
            2 * feature_count,
            len(TENNIS_EVENT_CLASSES),
        )

    def forward(self, clips):
        batch_size, true_length, channels, height, width = clips.shape
        if true_length > self.clip_frames:
            raise ValueError(
                f"Bounce model supports at most {self.clip_frames} frames per clip"
            )
        if true_length < self.clip_frames:
            clips = F.pad(clips, (0,) * 7 + (self.clip_frames - true_length,))
        features = self._features(
            clips.reshape(-1, channels, height, width)
        ).reshape(batch_size, self.clip_frames, -1)
        features = features[:, :true_length]
        sequence, _ = self._pred_fine._gru(features)
        sequence = self._pred_fine._dropout(sequence)
        return self._pred_fine._fc_out._fc_out(sequence)


class TennisBounceSpotter:
    """Run exact-frame bounce inference and return non-max-suppressed events."""

    def __init__(
        self,
        checkpoint_path,
        device=None,
        confidence_threshold=0.35,
        clip_frames=100,
        overlap_frames=50,
        image_size=224,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing tennis bounce checkpoint: {self.checkpoint_path}"
            )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.confidence_threshold = float(confidence_threshold)
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.clip_frames = int(clip_frames)
        self.overlap_frames = int(overlap_frames)
        self.image_size = int(image_size)
        if not 0 <= self.overlap_frames < self.clip_frames:
            raise ValueError("overlap_frames must be between 0 and clip_frames")
        self.model = TennisBounceSpotterNet(self.clip_frames).to(self.device)
        state = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def _prepare_frames(self, video_frames):
        prepared = []
        mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
        std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
        for frame in video_frames:
            height, width = frame.shape[:2]
            if height > self.image_size:
                resized_width = round(width * self.image_size / height)
                frame = cv2.resize(
                    frame,
                    (resized_width, self.image_size),
                    interpolation=cv2.INTER_AREA,
                )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            if height < self.image_size or width < self.image_size:
                pad_y = max(0, self.image_size - height)
                pad_x = max(0, self.image_size - width)
                rgb = cv2.copyMakeBorder(
                    rgb,
                    pad_y // 2,
                    pad_y - pad_y // 2,
                    pad_x // 2,
                    pad_x - pad_x // 2,
                    cv2.BORDER_CONSTANT,
                )
                height, width = rgb.shape[:2]
            y0 = (height - self.image_size) // 2
            x0 = (width - self.image_size) // 2
            crop = rgb[y0 : y0 + self.image_size, x0 : x0 + self.image_size]
            values = crop.astype(np.float32) / 255.0
            values = (values - mean) / std
            prepared.append(torch.from_numpy(values).permute(2, 0, 1))
        return prepared

    @staticmethod
    def _local_maxima(scores, radius):
        selected = []
        for frame_num, score in enumerate(scores):
            start = max(0, frame_num - radius)
            end = min(len(scores), frame_num + radius + 1)
            if score >= np.max(scores[start:end]):
                # Keep one frame when adjacent values tie.
                tied = np.flatnonzero(scores[start:end] == score) + start
                if frame_num == int(tied[len(tied) // 2]):
                    selected.append(frame_num)
        return selected

    def predict(self, video_frames, fps):
        if not video_frames:
            return []
        prepared = self._prepare_frames(video_frames)
        frame_count = len(prepared)
        scores = np.zeros((frame_count, len(TENNIS_EVENT_CLASSES)), np.float32)
        support = np.zeros(frame_count, np.float32)
        step = self.clip_frames - self.overlap_frames
        starts = list(range(-5, max(0, frame_count - self.overlap_frames), step))
        if not starts:
            starts = [-5]

        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode():
            for start in starts:
                clip = torch.zeros(
                    (self.clip_frames, 3, self.image_size, self.image_size),
                    dtype=torch.float32,
                )
                valid_start = max(0, start)
                valid_end = min(frame_count, start + self.clip_frames)
                if valid_start < valid_end:
                    destination = valid_start - start
                    clip[destination : destination + valid_end - valid_start] = (
                        torch.stack(prepared[valid_start:valid_end])
                    )
                with amp_context:
                    logits = self.model(clip.unsqueeze(0).to(self.device))
                    probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                source_start = valid_start - start
                source_end = source_start + valid_end - valid_start
                scores[valid_start:valid_end] += probabilities[source_start:source_end]
                support[valid_start:valid_end] += 1

        support[support == 0] = 1
        scores /= support[:, None]
        bounce_scores = scores[:, BOUNCE_CLASS_INDICES].max(axis=1)
        nms_radius = max(2, round(0.20 * float(fps)))
        events = []
        for frame_num in self._local_maxima(bounce_scores, nms_radius):
            confidence = float(bounce_scores[frame_num])
            if confidence < self.confidence_threshold:
                continue
            class_index = BOUNCE_CLASS_INDICES[
                int(np.argmax(scores[frame_num, BOUNCE_CLASS_INDICES]))
            ]
            events.append(
                {
                    "frame": frame_num,
                    "label": TENNIS_EVENT_CLASSES[class_index],
                    "confidence": confidence,
                    "source": "e2e_spot",
                }
            )
        return events
