# Tennis CV Tracker

A computer-vision pipeline for **standard and wheelchair tennis**. The project detects the **court, players, and ball**, reconstructs rally events, estimates **player and ball speeds**, identifies **bounces**, and determines the **point winner / ending reason**.

## Demo

### Sample Output

![Tracker demo](assets/tracker_demo.png)

### Animated Preview

![Tracker demo GIF](assets/tracker_demo.gif)

The overlay shows:
- court keypoints and a mini-court view
- player tracking with logical IDs (`P1` / `P2`)
- ball detection and trajectory cues
- live speed statistics for both players and the ball

## Features

- **Court detection** using a learned keypoint model
- **Player detection and tracking** for both standard and wheelchair tennis
- **Ball detection and trajectory reconstruction**
- **Scene-cut filtering** to ignore pre-roll and post-point broadcast cuts
- **Speed estimation** for players and shots
- **Bounce detection** with an optional learned E2E-Spot model
- **Point analysis** to infer the winner and point-ending reason
- **Detection caching** for faster reruns

## How It Works

The pipeline in `main.py` works roughly as follows:

1. Load the input tennis video.
2. Detect the full-court scene and skip broadcast pre-roll.
3. Detect court keypoints.
4. Detect players and assign stable player IDs.
5. Detect and track the ball.
6. Reconstruct the rally and bounce sequence.
7. Estimate player speed and shot speed.
8. Determine the point result and render the annotated output video.

## Project Structure

```text
cv-tennis-tracker/
├── constants/
├── court_line_detector/
├── mini_court/
├── tenniset/
├── trackers/
├── training/
├── utils/
├── main.py
├── requirements-bounce.txt
└── README.md
```

## Environment Setup (Conda)

I recommend using a dedicated **Conda** environment.

### 1. Create and activate the environment

```bash
conda create -n tennis python=3.10 -y
conda activate tennis
```

### 2. Install core dependencies

Install the common scientific/computer-vision packages with Conda:

```bash
conda install -c conda-forge numpy pandas opencv pillow -y
```

Then install the deep-learning packages and YOLO dependency:

```bash
pip install torch torchvision ultralytics
```

### 3. Optional: install the bounce-model dependency

If you want the exact-frame learned bounce detector, install:

```bash
pip install -r requirements-bounce.txt
```

That file currently installs:

- `timm`

### 4. Optional: save this as an environment file

You can also create an `environment.yml` like this:

```yaml
name: tennis
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.10
  - numpy
  - pandas
  - opencv
  - pillow
  - pip
  - pytorch
  - torchvision
  - pip:
      - ultralytics
      - timm
```

Then create it with:

```bash
conda env create -f environment.yml
conda activate tennis
```

## Required Model Files

Model weights are **not included** in the repository and should be placed manually.

Expected paths:

```text
models/keypoints_model.pth          # court keypoint model
models/best.pt                      # tennis ball detector
yolov8x.pt                          # generic player detector
models/wheelchair_best.pt           # optional wheelchair player detector
models/tennis_bounce_e2espot.pt     # optional learned bounce detector
```

## Running the Tracker

Place your input video in `input_videos/`, then run:

```bash
python main.py input_videos/input_video.mp4
```

To prevent the rendered video from opening automatically:

```bash
python main.py input_videos/input_video.mp4 --no-open
```

The program writes the annotated output video, statistics, point result, and cached detections to `output_videos/`.

## Optional Bounce Model

To download the bounce checkpoint helper assets used by the repo:

```bash
python training/download_bounce_model.py
```

Useful options:

```bash
python main.py input_videos/input_video.mp4 --no-bounce-model
python main.py input_videos/input_video.mp4 --bounce-model models/tennis_bounce_e2espot.pt
python main.py input_videos/input_video.mp4 --bounce-confidence 0.35
```

## Optional TenniSet Temporal Models

TenniSet-based temporal models can be used for extra event and outcome signals.

Prepare and train from the repository root:

```bash
python training/prepare_tenniset.py
python training/train_tenniset_temporal.py --task event --pretrained --output models/tenniset_event.pt
python training/train_tenniset_temporal.py --task outcome --pretrained --output models/tenniset_outcome.pt
```

Run with the trained models:

```bash
python main.py input_videos/input_video.mp4 \
  --event-model models/tenniset_event.pt \
  --outcome-model models/tenniset_outcome.pt
```

## Output

The tracker can produce:

- an annotated rally video
- player speed statistics
- shot speed statistics
- bounce predictions
- cached detections for faster reruns
- point winner / point-ending reason

## Notes

- This project is designed for **broadcast-style tennis footage** with a visible full-court view.
- The code supports both **standard** and **wheelchair** tennis.
- Some features depend on optional model files that are intentionally excluded from Git.

## Author

**John Chen**
