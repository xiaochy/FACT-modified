"""RoboTwin policy adapter that runs the FACT client and records rollouts for failure co-training.

Drop-in replacement for ``evaluation.robotwin.model2robotwin_interface`` (pass it as POLICY_NAME).
Every env step it stores the three RGB views, the 14-d joint state and the executed action; when an
episode ends it writes a staging directory (3 mp4 + npz + json). Only failed episodes are kept unless
FACT_COLLECT_SAVE_SUCCESS=1. ``scripts/build_failure_dataset.py`` converts the staging dirs into a
LeRobot v2.1 task dir with ``meta/failure_rollouts.jsonl``.

Env vars:
  FACT_COLLECT_DIR           staging root (default ./datasets/collect_staging)
  FACT_COLLECT_SAVE_SUCCESS  1 to also keep successful rollouts (default 0)
  FACT_COLLECT_TAG           string put in the staging dir name (e.g. n1 / bon4)
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.robotwin import model2robotwin_interface as _base

_CAMS = (
    ("head_camera", "cam_high"),
    ("left_camera", "cam_left_wrist"),
    ("right_camera", "cam_right_wrist"),
)
FPS = 15


class _Recorder:
    def __init__(self) -> None:
        self.root = Path(os.environ.get("FACT_COLLECT_DIR", "./datasets/collect_staging")).expanduser().resolve()
        self.save_success = os.environ.get("FACT_COLLECT_SAVE_SUCCESS", "0").strip().lower() in ("1", "true", "yes")
        self.tag = os.environ.get("FACT_COLLECT_TAG", "run")
        self.reset()

    def reset(self) -> None:
        self.episode_id: int | None = None
        self.task_name = "unknown_task"
        self.seed: int | None = None
        self.instruction = ""
        self.success = False
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.frames: dict[str, list[np.ndarray]] = {name: [] for _, name in _CAMS}

    def start(self, task_env, instruction: str) -> None:
        self.episode_id = int(getattr(task_env, "test_num", 0))
        self.task_name = str(getattr(task_env, "task_name", "unknown_task"))
        self.instruction = str(instruction)
        for attr in ("current_seed", "episode_seed", "seed", "now_seed"):
            if hasattr(task_env, attr):
                self.seed = int(getattr(task_env, attr))
                break

    def add(self, images, state: np.ndarray, action: np.ndarray) -> None:
        for (_, name), image in zip(_CAMS, images):
            self.frames[name].append(np.ascontiguousarray(np.asarray(image, dtype=np.uint8)))
        self.states.append(np.asarray(state, dtype=np.float32).reshape(-1).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(-1).copy())

    def finalize(self, reason: str) -> None:
        if self.episode_id is None or not self.states:
            self.reset()
            return
        keep = self.save_success or not self.success
        if keep:
            self._write(reason)
        self.reset()

    def _write(self, reason: str) -> None:
        out = self.root / self.task_name / f"{self.tag}_seed{self.seed}_ep{self.episode_id:04d}"
        out.mkdir(parents=True, exist_ok=True)
        for name, frames in self.frames.items():
            h, w = frames[0].shape[:2]
            proc = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                    "-video_size", f"{w}x{h}", "-framerate", str(FPS), "-i", "-",
                    "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "18", str(out / f"{name}.mp4"),
                ],
                stdin=subprocess.PIPE,
            )
            assert proc.stdin is not None
            for frame in frames:
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            proc.wait()
        np.savez(out / "data.npz", state=np.stack(self.states), action=np.stack(self.actions))
        (out / "meta.json").write_text(
            json.dumps(
                {
                    "task_name": self.task_name,
                    "seed": self.seed,
                    "episode_id": self.episode_id,
                    "instruction": self.instruction,
                    "success": bool(self.success),
                    "num_steps": len(self.states),
                    "fps": FPS,
                    "tag": self.tag,
                    "finalize_reason": reason,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"[FACT collect] wrote {'success' if self.success else 'FAILURE'} rollout -> {out} ({len(self.states)} steps)")


def get_model(usr_args: dict[str, Any]):
    # Per-step RGB is required for recording, so the low-frequency render shortcut must be off.
    usr_args = dict(usr_args)
    usr_args["low_frequency_rgb"] = False
    usr_args["skip_action_render_sync"] = False
    model = _base.get_model(usr_args)
    model._fact_recorder = _Recorder()
    atexit.register(model._fact_recorder.finalize, "atexit")
    return model


def reset_model(model) -> None:
    rec = getattr(model, "_fact_recorder", None)
    if rec is not None:
        rec.finalize("reset")
    _base.reset_model(model)


def eval(TASK_ENV, model, observation):  # noqa: A001
    rec: _Recorder = model._fact_recorder
    if rec.episode_id is not None and rec.episode_id != int(getattr(TASK_ENV, "test_num", 0)):
        rec.finalize("episode_switch")
    instruction = str(TASK_ENV.get_instruction())
    if rec.episode_id is None:
        rec.start(TASK_ENV, instruction)
    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    images = [observation["observation"][cam]["rgb"] for cam, _ in _CAMS]
    step = int(getattr(TASK_ENV, "take_action_cnt", 0))
    action = model.step({"lang": instruction, "state": state, "image": images}, step=step)
    rec.add(images, state, action)
    TASK_ENV.take_action(action)
    if getattr(TASK_ENV, "eval_success", False):
        rec.success = True
        model.mark_episode(success=True)
