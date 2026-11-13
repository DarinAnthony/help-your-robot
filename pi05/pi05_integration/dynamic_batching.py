"""Dynamic request batching for the Pi policy-server process."""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np


def stack_unbatched_observations(observations):
    """Stack compatible single-example wire payloads for one accelerator call."""
    required = ("image_primary", "image_wrist", "proprio", "language")
    if any(np.asarray(obs["image_primary"]).ndim != 3 for obs in observations):
        raise ValueError("dynamic batching accepts only unbatched image payloads")
    if any(any(key not in obs for key in required) for obs in observations):
        raise ValueError("dynamic batching received an incomplete observation")

    batched = {
        "image_primary": np.stack([obs["image_primary"] for obs in observations]),
        "image_wrist": np.stack([obs["image_wrist"] for obs in observations]),
        "proprio": np.stack([obs["proprio"] for obs in observations]),
        "language": [obs["language"] for obs in observations],
    }
    for key in (
        "embodiment_id",
        "dataset_component_id",
        "embodiment_descriptor",
        "embodiment_variant",
        "embodiment_action_dim",
        "embodiment_action_mask",
        "embodiment_state_mask",
        "sample_seed",
        "action_prefix",
        "action_prefix_weights",
        "response_state_history",
        "response_state_history_mask",
        "response_action_history",
        "response_action_history_mask",
        "dynamic_urma_values",
        "dynamic_urma_mask",
        "dynamic_urma5_values",
        "dynamic_urma51_values",
        "dynamic_urma51_analytic_mask",
        "base_state",
        "base_state_mask",
        "base_contract",
        "base_dt_s",
        "base_action_prefix",
        "arm_action",
        "action_dt_s",
        "joint_frame_sign",
        "joint_frame_zero",
    ):
        present = [key in obs for obs in observations]
        if any(present) and not all(present):
            raise ValueError(f"dynamic batch has inconsistent optional field {key!r}")
        if not any(present):
            continue
        values = [obs[key] for obs in observations]
        batched[key] = values if np.asarray(values[0]).ndim == 0 else np.stack(values)
    # The contracts a request names (image geometry, gripper value, joint frame; the server checks each against the
    # checkpoint's): one name for the whole batch. Before 2026-11-10 they were dropped here, so a batched server
    # refused every client of a checkpoint trained with one.
    for key in ("image_geometry", "gripper_value", "joint_frame"):
        present = [key in obs for obs in observations]
        if any(present) and not all(present):
            raise ValueError(f"dynamic batch has inconsistent optional field {key!r}")
        if not any(present):
            continue
        names = {obs[key].decode() if isinstance(obs[key], bytes | bytearray) else obs[key] for obs in observations}
        if len(names) != 1:
            raise ValueError(f"dynamic batch mixes {key} contracts {sorted(names)}")
        batched[key] = names.pop()
    return batched


class DynamicInferenceBatcher:
    """Coalesce concurrent simulator queries into deterministic accelerator batches."""

    def __init__(self, server, *, max_batch_size, wait_ms, logger=None):
        self.server = server
        self.max_batch_size = max_batch_size
        self.wait_s = wait_ms / 1_000
        self.queue = asyncio.Queue()
        self.request_count = 0
        self.batch_count = 0
        self.logger = logger or logging.getLogger(__name__)

    async def infer(self, observation):
        if self.max_batch_size == 1:
            return await asyncio.to_thread(self.server.infer, observation)
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((observation, future))
        return await future

    async def run(self):
        while True:
            pending = [await self.queue.get()]
            deadline = asyncio.get_running_loop().time() + self.wait_s
            while len(pending) < self.max_batch_size:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    break
                try:
                    pending.append(await asyncio.wait_for(self.queue.get(), timeout))
                except TimeoutError:
                    break
            observations = [item[0] for item in pending]
            futures = [item[1] for item in pending]
            started = time.perf_counter()
            try:
                if self.max_batch_size == 1:
                    outputs = [await asyncio.to_thread(self.server.infer, observations[0])]
                else:
                    # Keep one compiled accelerator shape. In particular, the first simulator
                    # often arrives seconds before its peers finish MuJoCo/EGL startup; compiling
                    # that singleton and then recompiling the full batch wastes tens of seconds.
                    padded = observations + [observations[-1]] * (
                        self.max_batch_size - len(observations)
                    )
                    result = await asyncio.to_thread(
                        self.server.infer,
                        stack_unbatched_observations(padded),
                    )
                    actions = np.asarray(result["actions"])
                    if len(actions) != self.max_batch_size:
                        raise ValueError(
                            f"batched policy returned {len(actions)} actions for fixed "
                            f"batch size {self.max_batch_size}"
                        )
                    # Split every returned array ("actions", and "base_actions" for a mobile-base
                    # checkpoint) back into its request.
                    outputs = [
                        {key: np.asarray(value)[index] for key, value in result.items()}
                        for index in range(len(pending))
                    ]
                for future, output in zip(futures, outputs, strict=True):
                    if not future.cancelled():
                        future.set_result(output)
            except Exception as error:
                for future in futures:
                    if not future.cancelled():
                        future.set_exception(error)
            finally:
                self.request_count += len(pending)
                self.batch_count += 1
                self.logger.info(
                    "inference batch=%d latency_ms=%.1f requests=%d batches=%d",
                    len(pending),
                    (time.perf_counter() - started) * 1_000,
                    self.request_count,
                    self.batch_count,
                )
