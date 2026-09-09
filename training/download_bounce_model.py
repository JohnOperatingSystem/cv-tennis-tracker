"""Download and verify the published E2E-Spot tennis RGB checkpoint."""

import argparse
import hashlib
from pathlib import Path
import tempfile
from urllib.request import urlopen


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_URL = (
    "https://github.com/jhong93/e2e-spot-models/raw/refs/heads/main/"
    "tennis_rny002gsm_gru_rgb/checkpoint_040.pt"
)
EXPECTED_SHA256 = (
    "5949e3e869f20480f51d89c68514b3eda4399779d9d2ca20c9b78344b11c8e2c"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "models" / "tennis_bounce_e2espot.pt",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    output = args.output.resolve()
    if output.is_file() and not args.force:
        if file_sha256(output) == EXPECTED_SHA256:
            print(f"Already installed and verified: {output}")
            return
        raise ValueError(
            f"A different file exists at {output}; pass --force to replace it"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        suffix=".download",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        with urlopen(MODEL_URL, timeout=60) as response:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                temporary.write(block)
    try:
        actual_hash = file_sha256(temporary_path)
        if actual_hash != EXPECTED_SHA256:
            raise ValueError(
                "Downloaded checkpoint failed SHA-256 verification: "
                f"{actual_hash}"
            )
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Installed verified bounce model: {output}")


if __name__ == "__main__":
    main()
