"""Convert staging rollouts from evaluation/robotwin/collect_failure_interface.py into a LeRobot v2.1 task dir.

    python -m scripts.build_failure_dataset \
        --staging_dir ./datasets/collect_staging/beat_block_hammer \
        --template_task_dir ./datasets/RoboTwin/Clean/beat_block_hammer \
        --out_task_dir ./datasets/RoboTwin/Fail/beat_block_hammer

Every failed rollout becomes one episode and gets a row in ``meta/failure_rollouts.jsonl``
(``failure_active_from_frame=0`` by default, i.e. the whole episode counts as a failure).
Re-running is idempotent: staging dirs already imported (tracked in ``meta/imported.json``) are skipped.
T5 embeddings and VAE latents are NOT computed here; run scripts.compute_t5_embedding and
scripts.compute_vae_latents on the output dir afterwards.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging_dir", required=True, type=Path)
    ap.add_argument("--template_task_dir", required=True, type=Path, help="expert task dir to copy info.json from")
    ap.add_argument("--out_task_dir", required=True, type=Path)
    ap.add_argument("--failure_active_from_frame", type=int, default=0)
    ap.add_argument("--include_success", action="store_true", help="also import successful rollouts as expert-like episodes (not recommended)")
    args = ap.parse_args()

    out = args.out_task_dir
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    info = json.loads((args.template_task_dir / "meta" / "info.json").read_text())
    fps = int(info["fps"])

    episodes = _read_jsonl(meta / "episodes.jsonl")
    tasks = _read_jsonl(meta / "tasks.jsonl")
    failures = _read_jsonl(meta / "failure_rollouts.jsonl")
    imported_path = meta / "imported.json"
    imported: list[str] = json.loads(imported_path.read_text()) if imported_path.is_file() else []
    task_idx = {t["task"]: int(t["task_index"]) for t in tasks}
    next_ep = len(episodes)
    next_frame = sum(int(e["length"]) for e in episodes)

    added = skipped = 0
    for ep_dir in sorted(p for p in args.staging_dir.iterdir() if p.is_dir()):
        if ep_dir.name in imported:
            continue
        m = json.loads((ep_dir / "meta.json").read_text())
        if m["success"] and not args.include_success:
            skipped += 1
            continue
        d = np.load(ep_dir / "data.npz")
        state, action = d["state"].astype(np.float32), d["action"].astype(np.float32)
        length = state.shape[0]
        instr = m["instruction"]
        if instr not in task_idx:
            task_idx[instr] = len(task_idx)
            tasks.append({"task_index": task_idx[instr], "task": instr})
        ep = next_ep
        df = {
            "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(state.reshape(-1)), 14),
            "action": pa.FixedSizeListArray.from_arrays(pa.array(action.reshape(-1)), 14),
            "timestamp": pa.array((np.arange(length) / fps).astype(np.float32)),
            "frame_index": pa.array(np.arange(length, dtype=np.int64)),
            "episode_index": pa.array(np.full(length, ep, dtype=np.int64)),
            "index": pa.array(np.arange(next_frame, next_frame + length, dtype=np.int64)),
            "task_index": pa.array(np.full(length, task_idx[instr], dtype=np.int64)),
        }
        data_path = out / info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(df), data_path)
        for cam in CAMS:
            key = f"observation.images.{cam}"
            vpath = out / info["video_path"].format(episode_chunk=ep // info["chunks_size"], video_key=key, episode_index=ep)
            vpath.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ep_dir / f"{cam}.mp4", vpath)
        episodes.append(
            {
                "episode_index": ep,
                "tasks": [instr],
                "length": length,
                "t5_embedding_path": f"t5_embedding/episode_{ep:06d}.pt",
            }
        )
        if not m["success"]:
            failures.append(
                {
                    "episode_index": ep,
                    "failure_episode": True,
                    "failure_active_from_frame": int(args.failure_active_from_frame),
                    "source": ep_dir.name,
                    "seed": m.get("seed"),
                }
            )
        imported.append(ep_dir.name)
        next_ep += 1
        next_frame += length
        added += 1

    info = dict(info)
    info["total_episodes"] = len(episodes)
    info["total_frames"] = next_frame
    info["total_tasks"] = len(tasks)
    info["total_videos"] = len(episodes) * len(CAMS)
    info["total_chunks"] = max(1, -(-len(episodes) // info["chunks_size"]))
    info["splits"] = {"train": f"0:{len(episodes)}"}
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    _write_jsonl(meta / "episodes.jsonl", episodes)
    _write_jsonl(meta / "tasks.jsonl", tasks)
    _write_jsonl(meta / "failure_rollouts.jsonl", failures)
    imported_path.write_text(json.dumps(imported))
    print(f"imported {added} episodes (skipped {skipped} successes); {len(episodes)} total, {next_frame} frames, {len(failures)} failure rows -> {out}")


if __name__ == "__main__":
    main()
