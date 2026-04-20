"""Episode-level 1-step absolute action shift.

Replaces episode['action'] with q_{t+1} (next-step observation copy) and
trims every per-frame array to length T - 1. Called by the orchestrator
between frames_to_episode() and DataCreator.convert_episode().
"""

from typing import Any, Dict

import numpy as np


def apply_one_step_shift(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new episode dict with action = obs[1:] and obs/images trimmed.

    The input episode layout must match the shape emitted by
    ``frames_to_episode`` (``obs`` as np.ndarray, ``images`` as dict of
    per-camera lists, ``task`` as string). Additional numpy arrays on the
    episode are trimmed too if present.
    """
    obs = np.asarray(episode["obs"], dtype=np.float32)
    if obs.shape[0] < 2:
        raise ValueError(
            "apply_one_step_shift requires at least 2 frames; "
            f"got {obs.shape[0]}."
        )

    new_obs = obs[:-1]
    new_action = obs[1:]

    new_images = {
        name: list(frames[:-1]) for name, frames in episode.get("images", {}).items()
    }

    shifted: Dict[str, Any] = {
        "obs": new_obs,
        "images": new_images,
        "task": episode.get("task", "no_task_specified"),
        "action": new_action,
    }

    # Carry over any extra per-frame numpy fields (e.g. wrench) by trimming.
    for key, val in episode.items():
        if key in ("obs", "images", "task", "action"):
            continue
        if isinstance(val, np.ndarray) and val.shape and val.shape[0] == obs.shape[0]:
            shifted[key] = val[:-1]
    return shifted
