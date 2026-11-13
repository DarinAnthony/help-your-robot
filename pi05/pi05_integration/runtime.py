"""Runtime provenance and guardrails shared by Pi training and serving entrypoints."""

from __future__ import annotations

import os
import socket
import subprocess


def _patch_orbax_empty_array_dtype(jax) -> bool:
    """Bridge Orbax 0.11.13 to JAX's dtype-aware empty-array constructor."""
    if not hasattr(jax, "make_array_from_single_device_arrays"):
        return False

    import contextvars

    from orbax.checkpoint._src.serialization import serialization, type_handlers

    original_read = serialization.read_and_create_array
    if getattr(original_read, "_molmospaces_dtype_patch", False):
        return False

    restore_dtype = contextvars.ContextVar("orbax_restore_dtype", default=None)
    original_make_array = jax.make_array_from_single_device_arrays

    async def read_and_create_array(*args, dtype, **kwargs):
        tensorstore = args[0] if args else kwargs["t"]
        effective_dtype = dtype
        if effective_dtype is None:
            effective_dtype = tensorstore.dtype.numpy_dtype
        token = restore_dtype.set(effective_dtype)
        try:
            return await original_read(*args, dtype=effective_dtype, **kwargs)
        finally:
            restore_dtype.reset(token)

    def make_array_from_single_device_arrays(shape, sharding, arrays, *, dtype=None):
        # Orbax 0.11.13 knows the restore dtype but drops it in its final JAX call.
        # A ContextVar keeps the value correct across concurrent async restores.
        return original_make_array(
            shape,
            sharding,
            arrays,
            dtype=dtype if dtype is not None else restore_dtype.get(),
        )

    read_and_create_array._molmospaces_dtype_patch = True
    serialization.read_and_create_array = read_and_create_array
    jax.make_array_from_single_device_arrays = make_array_from_single_device_arrays

    original_memory_size = type_handlers.ArrayHandler.memory_size

    def memory_size(handler, values):
        sizes = []
        for value in values:
            if not value.addressable_shards:
                sizes.append((0, 0))
            else:
                sizes.append(original_memory_size(handler, [value])[0])
        return sizes

    type_handlers.ArrayHandler.memory_size = memory_size
    return True


def initialize_distributed(platform: str) -> bool:
    """Initialize JAX's process group before any device access.

    Cloud TPU supplies the coordinator address, process count, and process index, so
    multi-host slices need no launcher-specific arguments here. Calling this on a
    single-host TPU is also valid and keeps the TPU entrypoint uniform.

    A GPU run is multi-process only with MOLMOSPACES_JAX_DISTRIBUTED=1. JAX finds the process
    group itself under Slurm (srun) and Open MPI 4 (mpirun); any other launcher sets
    JAX_COORDINATOR_ADDRESS, MOLMOSPACES_JAX_NUM_PROCESSES and MOLMOSPACES_JAX_PROCESS_ID.
    Which GPUs each process drives follows from the launch layout (``gpu_layout``),
    and every process must drive the same number of the same GPU type.
    MOLMOSPACES_JAX_INIT_TIMEOUT_S (optional) sets how long the processes wait for each other,
    for launchers whose pods can start minutes apart; unset keeps JAX's default.
    """
    if platform != "tpu" and os.environ.get("MOLMOSPACES_JAX_DISTRIBUTED") != "1":
        return False

    import jax

    # Orbax checkpoints written on a single GPU can contain arrays whose multi-process
    # FSDP resharding gives a particular host no addressable shard. JAX needs empty
    # arrays enabled while reconstructing those global arrays across processes.
    jax.config.update("jax_enable_empty_arrays", True)
    _patch_orbax_empty_array_dtype(jax)
    if not jax.distributed.is_initialized():
        kwargs = {}
        if process_count := os.environ.get("MOLMOSPACES_JAX_NUM_PROCESSES"):
            kwargs["num_processes"] = int(process_count)
        if process_id := os.environ.get("MOLMOSPACES_JAX_PROCESS_ID"):
            kwargs["process_id"] = int(process_id)
        if timeout := os.environ.get("MOLMOSPACES_JAX_INIT_TIMEOUT_S"):
            kwargs["initialization_timeout"] = int(timeout)
        if platform != "tpu":
            local_device_ids, isolated = gpu_layout()
            if local_device_ids is not None:
                kwargs["local_device_ids"] = local_device_ids
            if isolated:
                # NCCL's peer-to-peer and shared-memory transports fail ("invalid device ordinal")
                # between processes on one node that cannot see each other's GPUs; its network
                # transport works. Explicit settings win.
                os.environ.setdefault("NCCL_P2P_DISABLE", "1")
                os.environ.setdefault("NCCL_SHM_DISABLE", "1")
                print(
                    "[pi05.distributed] this process is bound to its own GPUs on a shared node, so NCCL "
                    "uses its network transport within the node; let every process see all of the "
                    "node's GPUs for peer-to-peer transfers.",
                    flush=True,
                )
        jax.distributed.initialize(**kwargs)
        if platform != "tpu":
            _check_uniform_gpus(jax)
    return True


def _slurm_tasks_on_node(environ) -> int | None:
    """This node's task count in the job step, from Slurm's compressed list such as ``2(x3),1``."""
    tasks = environ.get("SLURM_STEP_TASKS_PER_NODE") or environ.get("SLURM_TASKS_PER_NODE")
    node = environ.get("SLURM_NODEID")
    if not tasks or node is None:
        return None
    counts = []
    for item in tasks.split(","):
        count, _, repeat = item.partition("(x")
        counts += [int(count)] * (int(repeat.rstrip(")")) if repeat else 1)
    return counts[int(node)] if int(node) < len(counts) else None


def _local_rank_and_size(environ) -> tuple[int, int]:
    """This process's rank among the job's processes on its node, and their number.

    The innermost launcher wins: torchrun-style children of an srun task, or mpirun ranks
    under Slurm, inherit Slurm's variables for the task that started them.
    """
    for rank_key, size in (
        ("LOCAL_RANK", environ.get("LOCAL_WORLD_SIZE")),
        ("OMPI_COMM_WORLD_LOCAL_RANK", environ.get("OMPI_COMM_WORLD_LOCAL_SIZE")),
        ("SLURM_LOCALID", _slurm_tasks_on_node(environ)),
    ):
        if rank_key in environ and size:
            return int(environ[rank_key]), int(size)
    return 0, 1


def _physical_gpu_count() -> int | None:
    """The GPUs this host exposes to the process; nvidia-smi ignores CUDA_VISIBLE_DEVICES."""
    try:
        listing = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return sum(line.startswith("GPU ") for line in listing.splitlines()) or None


def gpu_layout(
    environ=os.environ, visible: int | None = None, node: int | None = None
) -> tuple[list[int] | None, bool]:
    """The visible GPUs this process drives in a multi-process GPU run (None leaves it to JAX), and
    whether it shares its node with processes that cannot see those GPUs.

    Left unset, JAX gives each process only the GPU numbered by its node-local rank: one
    process per node then drives one of its GPUs, and a process bound to its own GPUs is
    pointed at one it may not see. Here a process that sees fewer GPUs than the node gives the
    job (Slurm's SLURM_GPUS_ON_NODE, else what nvidia-smi lists) was bound to them and drives
    all of them; otherwise the node's processes split the GPUs evenly by local rank, and one
    process per node drives every GPU. ``visible`` and ``node`` default to this host's counts;
    JAX_LOCAL_DEVICE_IDS overrides the choice.
    """
    rank, size = _local_rank_and_size(environ)
    if environ.get("JAX_LOCAL_DEVICE_IDS"):
        return None, False
    if node is None:
        node = int(environ["SLURM_GPUS_ON_NODE"]) if environ.get("SLURM_GPUS_ON_NODE") else _physical_gpu_count()
    if visible is None:
        devices = environ.get("CUDA_VISIBLE_DEVICES")
        visible = node if devices is None else len([d for d in devices.split(",") if d.strip()])
    if not visible:
        return None, False
    bound = visible < node if node else visible % size != 0
    if size == 1 or bound:
        return list(range(visible)), bound and size > 1
    if visible % size:
        raise ValueError(
            f"{size} processes on this node cannot split its {visible} GPUs evenly; launch a number "
            "of processes per node that divides the GPU count, or set JAX_LOCAL_DEVICE_IDS."
        )
    share = visible // size
    return list(range(rank * share, (rank + 1) * share)), False


def _check_uniform_gpus(jax) -> None:
    """Fail at startup unless every process drives the same number of the same GPU type.

    XLA shares autotuning results across processes, which fails deep in the first compile
    (``Expect autotune result cache hit``) when GPU types differ, and the input pipeline splits
    each batch evenly across processes. Every process sees every device's owner, so the counts
    are compared without communication; the types are gathered only once the counts agree,
    because an all-gather needs the same number of devices on every process.
    """
    import numpy as np
    from jax.experimental import multihost_utils

    local = jax.local_devices()
    kind = ",".join(sorted({device.device_kind for device in local}))
    print(
        f"[pi05.distributed] process {jax.process_index()}/{jax.process_count()} on "
        f"{socket.gethostname()} drives {len(local)} {kind}",
        flush=True,
    )
    counts = [0] * jax.process_count()
    for device in jax.devices():
        counts[device.process_index] += 1
    if len(set(counts)) == 1:
        gathered = multihost_utils.process_allgather(
            np.frombuffer(kind.encode()[:128].ljust(128), np.uint8)
        )
        kinds = [bytes(row).decode().rstrip() for row in np.asarray(gathered).reshape(-1, 128)]
        if len(set(kinds)) == 1:
            return
    else:
        kinds = ["?"] * len(counts)
    layout = ", ".join(
        f"process {i}: {n}" + (f" {k}" if k != "?" else "") for i, (n, k) in enumerate(zip(counts, kinds))
    )
    raise RuntimeError(
        "Multi-process GPU training needs every process to drive the same number of the same GPU "
        f"type; got {layout}. Request one GPU type and the same GPUs per process, or set "
        "JAX_LOCAL_DEVICE_IDS."
    )


def global_device_get(tree):
    """Materialize a global JAX pytree on every host.

    ``jax.device_get`` only works for arrays whose shards are all addressable by
    the current process. Multi-host visualization batches span remote devices,
    so every process must participate in an all-gather before process zero can
    render the panel. The single-process path retains the ordinary device-get
    behavior used by GPU and v4-8 runs.
    """
    import jax

    if jax.process_count() == 1:
        return jax.device_get(tree)

    from jax.experimental import multihost_utils

    return multihost_utils.process_allgather(tree)


def describe_runtime(expected_platform: str | None = None) -> dict:
    """Return JAX runtime provenance and optionally enforce the accelerator platform."""
    import jax

    devices = jax.devices()
    platforms = sorted({device.platform for device in devices})
    expected = expected_platform or os.environ.get("MOLMOSPACES_EXPECT_ACCELERATOR")
    if expected and expected not in platforms:
        raise RuntimeError(
            f"expected JAX platform {expected!r}, found {platforms}; devices={devices}"
        )
    return {
        "jax_version": jax.__version__,
        "platforms": platforms,
        "device_count": len(devices),
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "devices": [str(device) for device in devices],
        # ``str(device)`` is only ``cuda:0`` on the GPU plugin and therefore cannot
        # distinguish an H100 from an H200/A6000. Closed-loop trajectories can amplify
        # accelerator-specific floating-point differences, so the concrete device kind
        # is part of the evaluation/training provenance contract.
        "device_kinds": [getattr(device, "device_kind", None) for device in devices],
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
    }


def log_runtime(expected_platform: str | None = None) -> dict:
    info = describe_runtime(expected_platform)
    print(f"[pi05.runtime] {info}", flush=True)
    return info
