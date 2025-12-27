# src/03_env_snapshot.py
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata

OUT_DIR = os.path.join(os.path.dirname(__file__), "..")
FREEZE_PATH = os.path.join(OUT_DIR, "requirements.freeze.txt")
SUMMARY_PATH = os.path.join(OUT_DIR, "requirements.summary.json")


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
        return p.returncode, out.strip()
    except Exception as e:
        return 999, f"[exception running {cmd}] {e}"


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except Exception:
        return None


def main() -> None:
    print("=== Interpreter ===")
    print("python:", sys.version.replace("\n", " "))
    print("exe:", sys.executable)
    print("venv:", os.environ.get("VIRTUAL_ENV", "(not set)"))
    print()

    print("=== System ===")
    print("platform:", platform.platform())
    print("machine:", platform.machine())
    print("processor:", platform.processor())
    print()

    # Key packages you likely care about for this project
    pkgs = [
        "opencv-python",
        "opencv-contrib-python",
        "numpy",
        "mediapipe",
        "ultralytics",
        "torch",
        "torchvision",
        "torchaudio",
        "cvzone",
        "protobuf",
    ]

    versions = {p: _version(p) for p in pkgs if _version(p) is not None}

    print("=== Key package versions ===")
    if versions:
        for k in sorted(versions):
            print(f"{k}: {versions[k]}")
    else:
        print("(none found)")
    print()

    print("=== MediaPipe solutions packaging check ===")
    mp_info = {}
    try:
        import mediapipe as mp  # type: ignore

        mp_info["mediapipe_file"] = getattr(mp, "__file__", None)
        mp_info["mediapipe_version_runtime"] = getattr(mp, "__version__", None)
        mp_info["has_mp_solutions_attr"] = hasattr(mp, "solutions")

        # classic import path
        rc, out = _run([sys.executable, "-c", "import mediapipe.solutions as s; print('OK', s.__file__)"])
        mp_info["import_mediapipe_solutions_rc"] = rc
        mp_info["import_mediapipe_solutions_out"] = out

        # python.solutions import path (often the reliable one on Windows wheels)
        rc2, out2 = _run([sys.executable, "-c", "from mediapipe.python.solutions import pose; print('OK', pose.__file__)"])
        mp_info["import_mediapipe_python_solutions_pose_rc"] = rc2
        mp_info["import_mediapipe_python_solutions_pose_out"] = out2

    except Exception as e:
        mp_info["error"] = str(e)

    for k, v in mp_info.items():
        print(f"{k}: {v}")
    print()

    print("=== Torch acceleration ===")
    torch_info = {}
    try:
        import torch  # type: ignore

        torch_info["torch_version"] = torch.__version__
        torch_info["cuda_available"] = bool(torch.cuda.is_available())
        torch_info["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch_info["cuda_available"]:
            torch_info["cuda_device_count"] = torch.cuda.device_count()
            torch_info["cuda_device_name_0"] = torch.cuda.get_device_name(0)
    except Exception as e:
        torch_info["error"] = str(e)

    for k, v in torch_info.items():
        print(f"{k}: {v}")
    print()

    print("=== pip check (dependency sanity) ===")
    rc, out = _run([sys.executable, "-m", "pip", "check"])
    print(out if out else "OK (no broken requirements found)")
    print()

    print("=== Writing requirements.freeze.txt (pip freeze) ===")
    rc, out = _run([sys.executable, "-m", "pip", "freeze"])
    if rc == 0 and out:
        with open(FREEZE_PATH, "w", encoding="utf-8") as f:
            f.write(out + "\n")
        print(f"Wrote: {os.path.abspath(FREEZE_PATH)}")
    else:
        print("Failed to freeze:", out[:4000])
    print()

    print("=== Writing requirements.summary.json ===")
    summary = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "versions": versions,
        "mediapipe_packaging": mp_info,
        "torch": torch_info,
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote: {os.path.abspath(SUMMARY_PATH)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
