"""Cheap JAX collective smoke for a multi-host job (Cloud TPU slice or multi-pod GPU)."""

from __future__ import annotations

import argparse

from pi05_integration.runtime import describe_runtime, initialize_distributed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-processes", type=int, required=True)
    parser.add_argument("--expected-devices", type=int, required=True)
    parser.add_argument(
        "--platform",
        choices=("tpu", "gpu"),
        default="tpu",
        help="gpu needs MOLMOSPACES_JAX_DISTRIBUTED=1 (see runtime.initialize_distributed).",
    )
    args = parser.parse_args()

    initialize_distributed(args.platform)

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    info = describe_runtime(args.platform)
    if info["process_count"] != args.expected_processes:
        raise RuntimeError(
            f"expected {args.expected_processes} processes, found {info['process_count']}"
        )
    if info["device_count"] != args.expected_devices:
        raise RuntimeError(
            f"expected {args.expected_devices} global devices, found {info['device_count']}"
        )

    local_value = jnp.asarray(jax.process_index() + 1, dtype=jnp.int32)
    gathered = multihost_utils.process_allgather(local_value, tiled=False)
    expected = jnp.arange(1, args.expected_processes + 1, dtype=jnp.int32)
    if not bool(jnp.array_equal(gathered, expected)):
        raise RuntimeError(f"collective mismatch: expected {expected}, found {gathered}")

    # Device-level all-reduce over every global device (crosses the inter-host fabric,
    # e.g. NCCL between GPU pods): each device holds its global index + 1, so the sum is
    # n * (n + 1) / 2.
    devices = np.asarray(jax.devices())
    n_devices = devices.size
    mesh = Mesh(devices, ("devices",))
    per_device = jax.make_array_from_callback(
        (n_devices,),
        NamedSharding(mesh, PartitionSpec("devices")),
        lambda index: np.arange(n_devices, dtype=np.float32)[index] + 1,
    )
    total = jax.jit(jnp.sum, out_shardings=NamedSharding(mesh, PartitionSpec()))(per_device)
    total_value = float(total.addressable_data(0))
    expected_total = n_devices * (n_devices + 1) / 2
    if total_value != expected_total:
        raise RuntimeError(f"device all-reduce mismatch: expected {expected_total}, found {total_value}")

    multihost_utils.sync_global_devices("molmospaces_multihost_smoke_complete")
    print(
        {**info, "process_values": gathered.tolist(), "device_allreduce": total_value},
        flush=True,
    )


if __name__ == "__main__":
    main()
