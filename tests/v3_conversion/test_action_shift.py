import numpy as np

from v3_conversion.action_shift import apply_one_step_shift


def test_apply_one_step_shift_moves_obs_forward():
    episode = {
        "obs": np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]], dtype=np.float32),
        "images": {"cam_left": [np.zeros((1, 1, 3), dtype=np.uint8)] * 3},
        "task": "Insert cable.",
        "action": np.array([[9.0, 9.0], [9.0, 9.0], [9.0, 9.0]], dtype=np.float32),
    }

    shifted = apply_one_step_shift(episode)

    assert shifted["obs"].shape == (2, 2)
    assert np.allclose(shifted["obs"], [[0.0, 0.1], [0.2, 0.3]])
    assert np.allclose(shifted["action"], [[0.2, 0.3], [0.4, 0.5]])
    assert len(shifted["images"]["cam_left"]) == 2
    assert shifted["task"] == "Insert cable."


def test_apply_one_step_shift_rejects_single_frame():
    episode = {
        "obs": np.zeros((1, 2), dtype=np.float32),
        "images": {"cam_left": [np.zeros((1, 1, 3), dtype=np.uint8)]},
        "task": "x",
        "action": np.zeros((1, 2), dtype=np.float32),
    }
    import pytest

    with pytest.raises(ValueError):
        apply_one_step_shift(episode)


def test_apply_one_step_shift_rewrites_top_level_action_slices():
    episode = {
        "obs": np.array(
            [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]],
            dtype=np.float32,
        ),
        "action_left": np.array([[9.0], [9.0], [9.0]], dtype=np.float32),
        "action_right": np.array(
            [[8.0, 8.0], [8.0, 8.0], [8.0, 8.0]], dtype=np.float32
        ),
        "wrench": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        "images": {"cam_left": [np.zeros((1, 1, 3), dtype=np.uint8)] * 3},
        "task": "Insert cable.",
    }

    shifted = apply_one_step_shift(episode)

    assert shifted["obs"].shape == (2, 3)
    assert np.allclose(shifted["action_left"], [[0.3], [0.6]])
    assert np.allclose(shifted["action_right"], [[0.4, 0.5], [0.7, 0.8]])
    assert np.allclose(shifted["wrench"], [[1.0], [2.0]])
    assert len(shifted["images"]["cam_left"]) == 2
