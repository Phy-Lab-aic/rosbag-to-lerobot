from types import SimpleNamespace

import numpy as np

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
    assert np.allclose(result, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
