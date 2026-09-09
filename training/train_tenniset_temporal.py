"""Train event timing or point outcome heads on the TenniSet videos."""

import argparse
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tenniset import TenniSetAnnotations  # noqa: E402
from tenniset.temporal import (  # noqa: E402
    EventWindowDataset,
    PointOutcomeDataset,
    TennisTemporalNet,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("event", "outcome"), default="event")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_DIR / "external" / "tenniset" / "annotations",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=PROJECT_DIR / "external" / "tenniset" / "videos",
    )
    parser.add_argument("--overrides", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "models" / "tenniset_temporal.pt",
    )
    parser.add_argument("--train-videos", nargs="+", default=["V006", "V007", "V008", "V009"])
    parser.add_argument("--val-videos", nargs="+", default=["V010"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--clip-frames", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def make_dataset(args, annotations, video_ids, training):
    clip_frames = args.clip_frames or (16 if args.task == "event" else 32)
    common = {
        "annotations": annotations,
        "video_dir": args.video_dir,
        "video_ids": video_ids,
        "clip_frames": clip_frames,
        "image_size": args.image_size,
    }
    if args.task == "event":
        return EventWindowDataset(**common, training=training)
    return PointOutcomeDataset(**common)


def calculate_loss(task, outputs, targets, criterion):
    if task == "event":
        loss = criterion(outputs["event"], targets)
        predictions = outputs["event"].argmax(dim=1)
        correct = int((predictions == targets).sum())
        return loss, correct, len(targets)

    winner_targets = targets["winner"]
    reason_targets = targets["reason"]
    winner_loss = criterion(outputs["winner"], winner_targets)
    reason_loss = criterion(outputs["reason"], reason_targets)
    winner_correct = outputs["winner"].argmax(dim=1) == winner_targets
    reason_correct = outputs["reason"].argmax(dim=1) == reason_targets
    jointly_correct = int((winner_correct & reason_correct).sum())
    return winner_loss + reason_loss, jointly_correct, len(winner_targets)


def run_epoch(model, loader, task, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    sample_count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for videos, targets in loader:
            videos = videos.to(device)
            if isinstance(targets, dict):
                targets = {name: value.to(device) for name, value in targets.items()}
            else:
                targets = targets.to(device)
            outputs = model(videos)
            loss, batch_correct, batch_count = calculate_loss(
                task, outputs, targets, criterion
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss) * batch_count
            correct += batch_correct
            sample_count += batch_count
    return total_loss / max(1, sample_count), correct / max(1, sample_count)


def main():
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    annotations = TenniSetAnnotations(args.annotations, args.overrides)
    train_dataset = make_dataset(args, annotations, args.train_videos, True)
    val_dataset = make_dataset(args, annotations, args.val_videos, False)
    if not train_dataset or not val_dataset:
        raise ValueError("The selected train/validation split produced no samples")
    for video_id in set(args.train_videos + args.val_videos):
        if not any((args.video_dir / f"{video_id}{suffix}").is_file() for suffix in (".mp4", ".avi", ".mkv", ".mov")):
            raise FileNotFoundError(
                f"Missing {video_id} in {args.video_dir}. Download the 11.1 GB "
                "TenniSet video archive before training."
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = TennisTemporalNet(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    clip_frames = args.clip_frames or (16 if args.task == "event" else 32)

    print(
        f"Training {args.task} model on {device}: "
        f"{len(train_dataset)} train / {len(val_dataset)} validation samples"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, args.task, device, criterion, optimizer
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, args.task, device, criterion
        )
        print(
            f"epoch {epoch:02d} train loss={train_loss:.4f} "
            f"accuracy={train_accuracy:.3f} val loss={val_loss:.4f} "
            f"accuracy={val_accuracy:.3f}"
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "task": args.task,
                    "clip_frames": clip_frames,
                    "image_size": args.image_size,
                    "train_videos": args.train_videos,
                    "val_videos": args.val_videos,
                    "validation_accuracy": val_accuracy,
                },
                args.output,
            )
    print(f"Best validation accuracy: {best_accuracy:.3f}")
    print(f"Checkpoint: {args.output}")


if __name__ == "__main__":
    main()
