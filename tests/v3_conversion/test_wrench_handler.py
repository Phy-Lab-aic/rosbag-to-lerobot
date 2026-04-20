from types import SimpleNamespace

import numpy as np
import pytest

from v3_conversion.data_converter import _convert_joint_msg


def test_wrench_stamped_converts_to_6d_vector():
    msg = SimpleNamespace(
        header=SimpleNamespace(),
        wrench=SimpleNamespace(
            force=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            torque=SimpleNamespace(x=4.0, y=5.0, z=6.0),
        ),
    )
    result = _convert_joint_msg(msg, None, "geometry_msgs/msg/WrenchStamped")
    assert result.dtype == np.float32
    assert result.shape == (6,)
    assert np.allclose(result, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_wrench_stamped_validates_joint_order_width():
    msg = SimpleNamespace(
        header=SimpleNamespace(),
        wrench=SimpleNamespace(
            force=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            torque=SimpleNamespace(x=4.0, y=5.0, z=6.0),
        ),
    )
    with pytest.raises(ValueError, match="WrenchStamped produces 6 values"):
        _convert_joint_msg(
            msg,
            ["a", "b", "c", "d", "e"],
            "geometry_msgs/msg/WrenchStamped",
        )
