# Tennis CV Tracker

A computer-vision pipeline for standard and wheelchair tennis. It detects the
court, players, and ball; reconstructs the rally; estimates player and ball
speeds; identifies bounces; and determines the point winner and reason.

## Run the tracker

Activate the project environment, place the video in `input_videos`, and
run:

```powershell
conda activate tennis
python main.py input_videos/input_video.mp4
```

The annotated video, point result, statistics, and detection caches are written
to `output_videos`. Add `--no-open` to prevent the rendered video from opening
automatically.

Model weights are intentionally excluded from Git. The application expects the
ball and court models under `models`, the generic player model at `yolov8x.pt`,
and optionally `models/wheelchair_best.pt` for wheelchair detection.

## Bounce detection

The tracker combines E2E-Spot frame-level predictions with the tracked ball
path, court homography, player proximity, and rally order. The model predicts
bounce timing; trajectory geometry supplies and validates the landing position.

The checkpoint is expected at `models/tennis_bounce_e2espot.pt`. Install its
additional dependency and download the checkpoint with:

```powershell
python -m pip install -r requirements-bounce.txt
python training/download_bounce_model.py
```

Useful options include `--no-bounce-model`, `--bounce-model PATH`, and
`--bounce-confidence 0.35`. Predictions are cached under `output_videos`.

For wheelchair footage, fine-tune on exact impact frames from several matches.
Include at least 300 checked bounces, at least 100 second bounces, and difficult
swing/serve negatives. Split training and validation by match.

- Training implementation: <https://github.com/jhong93/spot>
- Published checkpoints: <https://github.com/jhong93/e2e-spot-models>

## Optional TenniSet models

TenniSet can provide temporal signals for serves, hits, winner side, and the
point-ending reason. Court and ball geometry remain authoritative for position,
speed, and complete geometric calls.

Download the annotations from <https://github.com/HaydenFaulkner/Tennis> into
`external/tenniset/annotations`. Place `V006.mp4` through `V010.mp4` in
`external/tenniset/videos`, then prepare and train from the repository root:

```powershell
python training/prepare_tenniset.py
python training/train_tenniset_temporal.py --task event --pretrained --output models/tenniset_event.pt
python training/train_tenniset_temporal.py --task outcome --pretrained --output models/tenniset_outcome.pt
```

Run the trained models with:

```powershell
python main.py input_videos/input_video.mp4 `
  --event-model models/tenniset_event.pt `
  --outcome-model models/tenniset_outcome.pt
```

The paths may instead be set through `TENNIS_EVENT_MODEL` and
`TENNIS_OUTCOME_MODEL`.

## Third-party notices

### E2E-Spot tennis model

`models/tennis_bounce_e2espot.pt` is the published
`tennis_rny002gsm_gru_rgb/checkpoint_040.pt` checkpoint from
<https://github.com/jhong93/e2e-spot-models>. The compatible architecture in
`tenniset/bounce_spotter.py` is adapted from <https://github.com/jhong93/spot>.

Copyright 2022 James Hong, Haotian Zhang, Matthew Fisher, Michael Gharbi,
Kayvon Fatahalian

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
