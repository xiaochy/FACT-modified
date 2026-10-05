"""Convert raw RoboTwin collection episodes (HDF5, e.g. RoboTwin/data/<task>/<config>/data/episodeN.hdf5)
into staging dirs that scripts/build_failure_dataset.py turns into a LeRobot v2.1 task dir.

    python -m scripts.raw_to_staging \
        --raw_dir ../RoboTwin/data/place_bread_skillet/demo_clean_depth \
        --out_dir ./datasets/own_staging/place_bread_skillet
    python -m scripts.build_failure_dataset --include_success \
        --staging_dir ./datasets/own_staging/place_bread_skillet \
        --template_task_dir ./datasets/RoboTwin/Clean/place_bread_skillet \
        --out_task_dir ./datasets/RoboTwin_own/Clean/place_bread_skillet

Conventions (checked against the released robotwin-v2 data):
  state[t]  = joint_action/vector[t]        (14-d: L arm 6, L grip, R arm 6, R grip)
  action[t] = state[t+1], last action = last state
  views     = head_camera -> cam_high, left_camera -> cam_left_wrist, right_camera -> cam_right_wrist
  frames are decoded with cv2.imdecode (RoboTwin stores true-RGB JPEGs, no channel swap needed), 15 fps.
The instruction of each episode is drawn (seeded by the episode index) from instructions/episodeN.json["seen"].
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import h5py
import imageio_ffmpeg
import numpy as np

CAMS = (("head_camera", "cam_high"), ("left_camera", "cam_left_wrist"), ("right_camera", "cam_right_wrist"))
FPS = 15


def _write_mp4(frames: np.ndarray, path: Path) -> None:
    h, w = frames.shape[1:3]
    proc = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{w}x{h}", "-framerate", str(FPS), "-i", "-",
            "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "18", str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    proc.stdin.write(np.ascontiguousarray(frames).tobytes())
    proc.stdin.close()
    proc.wait()


def convert_episode(args: tuple[str, int, str, int | None]) -> str:
    raw_dir, idx, out_dir, seed = args
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out = out_dir / f"ep{idx:04d}"
    if (out / "meta.json").is_file():
        return f"skip {out.name}"
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(raw_dir / "data" / f"episode{idx}.hdf5", "r") as f:
        vec = f["joint_action/vector"][()].astype(np.float32)
        for cam, name in CAMS:
            blobs = f["observation"][cam]["rgb"][()]
            frames = np.stack([cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in blobs])
            assert len(frames) == len(vec), f"episode{idx}: {cam} has {len(frames)} frames, state has {len(vec)}"
            _write_mp4(frames, out / f"{name}.mp4")
    state = vec
    action = np.concatenate([vec[1:], vec[-1:]], axis=0)
    np.savez(out / "data.npz", state=state, action=action)
    ins = json.loads((raw_dir / "instructions" / f"episode{idx}.json").read_text())["seen"]
    instruction = ins[int(np.random.RandomState(idx).randint(len(ins)))]
    (out / "meta.json").write_text(
        json.dumps(
            {
                "task_name": raw_dir.parent.name, "seed": seed, "episode_id": idx, "instruction": instruction,
                "success": True, "num_steps": int(len(vec)), "fps": FPS, "tag": "raw", "finalize_reason": "raw",
            },
            ensure_ascii=False, indent=2,
        )
    )
    return f"done {out.name} ({len(vec)} frames)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_episodes", type=int, default=0)
    a = ap.parse_args()
    n = len(list((a.raw_dir / "data").glob("episode*.hdf5")))
    if a.max_episodes:
        n = min(n, a.max_episodes)
    seeds = None
    seed_file = a.raw_dir / "seed.txt"
    if seed_file.is_file():
        seeds = [int(s) for s in seed_file.read_text().split()]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(a.raw_dir), i, str(a.out_dir), None if seeds is None or i >= len(seeds) else seeds[i]) for i in range(n)]
    with ProcessPoolExecutor(a.workers) as ex:
        for k, msg in enumerate(ex.map(convert_episode, jobs), 1):
            if k % 10 == 0 or k == n:
                print(f"[{k}/{n}] {msg}", flush=True)


if __name__ == "__main__":
    main()
