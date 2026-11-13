import numpy as np

from openpi.models.model import ModelType
from openpi.policies.libero_policy import LiberoInputs


def test_libero_inputs_preserves_named_dynamic_urma_fields():
    values = np.arange(80, dtype=np.float32).reshape(10, 8)
    mask = np.ones((10, 8), dtype=bool)
    output = LiberoInputs(model_type=ModelType.PI0)(
        {
            "observation/image": np.zeros((8, 8, 3), np.uint8),
            "observation/wrist_image": np.zeros((8, 8, 3), np.uint8),
            "observation/state": np.zeros(10, np.float32),
            "dynamic_urma_values": values,
            "dynamic_urma_mask": mask,
        }
    )
    np.testing.assert_array_equal(output["dynamic_urma_values"], values)
    np.testing.assert_array_equal(output["dynamic_urma_mask"], mask)
