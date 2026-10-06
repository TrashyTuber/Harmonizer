"""Assemble the Hugging Face Space upload in space_build/.

Copies the Space config (space/), the modules app.py imports, and the
trained artifacts into one folder, then checks the app imports from there.

Usage:
    uv run python scripts/build_space.py
    uv run hf upload <username>/<space-name> space_build . --repo-type space
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "space_build"

MODULES = ["app.py", "checkpoint.py", "decode.py", "model.py", "render.py", "tokenizer.py"]
ARTIFACTS = ["lstm.pt", "tokenizer.json", "transition_matrix.pt"]


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "artifacts").mkdir(parents=True)

    for f in (ROOT / "space").iterdir():
        shutil.copy(f, OUT / f.name)
    for name in MODULES:
        shutil.copy(ROOT / name, OUT / name)
    for name in ARTIFACTS:
        shutil.copy(ROOT / "artifacts" / name, OUT / "artifacts" / name)

    check = subprocess.run(
        [sys.executable, "-c", "import app; print(app.from_text('C4 E4 G4 C5:2', app.SMOOTHED, 110)[3])"],
        cwd=OUT, capture_output=True, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if check.returncode != 0:
        raise SystemExit(f"app failed to run from {OUT}:\n{check.stderr}")

    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            print(f"{path.relative_to(OUT)}  ({path.stat().st_size / 1024:.0f} KB)")
    print(f"\nSmoke test: {check.stdout.strip()}")


if __name__ == "__main__":
    main()
