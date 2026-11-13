from types import SimpleNamespace

import numpy as np

from openpi.training import train_metrics


def test_grouped_means_separates_task_and_embodiment():
    grouped = train_metrics.grouped_means(
        [1.0, 3.0, 8.0],
        task_ids=[0, 0, 1],
        embodiment_ids=[13, 13, 18],
    )

    assert grouped["open/embodiment_13"] == 2.0
    assert grouped["close/embodiment_18"] == 8.0
    assert grouped["_aggregate"] == 4.0


def test_data_visualization_contains_model_inputs_partitions_and_actions():
    observation = SimpleNamespace(
        images={
            "base_0_rgb": np.zeros((2, 8, 8, 3), np.float32),
            "left_wrist_0_rgb": np.zeros((2, 8, 8, 3), np.float32),
            "right_wrist_0_rgb": np.zeros((2, 8, 8, 3), np.float32),
        },
        state=np.zeros((2, 10), np.float32),
        embodiment_id=np.array([13, 18], np.int32),
        task_id=np.array([0, 1], np.int32),
        tokenized_prompt=np.array([[1, 2, 0], [3, 4, 0]], np.int32),
        tokenized_prompt_mask=np.array([[True, True, False], [True, True, False]]),
        action_loss_mask=np.ones((2, 4, 10), bool),
        robotoken_values=np.zeros((2, 4, 32), np.float32),
        robotoken_types=np.array([[0, 1, 0, 0], [0, 0, 0, 0]], np.int32),
        robotoken_reference_ids=np.full((2, 4, 5), -1, np.int32),
        robotoken_mask=np.array([[True, True, False, False], [True, False, False, False]]),
    )
    actions = np.zeros((2, 4, 10), np.float32)
    predictions = np.ones_like(actions)

    panel = train_metrics.data_visualization_html(
        observation,
        actions,
        predictions,
        count=1,
        physical_state_dim=7,
        state_conditioning_dim=2,
        prompt_conditioning="EmbodimentIdPrompt",
        state_encoding="EmbodimentOneHotState",
        normalization="PerEmbodimentQuantileNorm",
        action_expert_conditioning="LearnedEmbodimentAdaRMS",
        token_describer=lambda ids: ("Task: open, State: 1 2; Action:", ("<bos>", "open")),
    )

    assert "Training sample 0: open, embodiment 13" in panel
    assert panel.count("data:image/jpeg;base64,") == 3
    assert "physical [0:7]" in panel
    assert "embodiment [7:9]" in panel
    assert "Token IDs (2):</b> 1 2" in panel
    assert "LearnedEmbodimentAdaRMS" in panel
    assert "Physical-unit action chunk" in panel
    assert "Static RoboToken prefix sequence" in panel
    assert "Valid tokens:</b> 2/4" in panel
