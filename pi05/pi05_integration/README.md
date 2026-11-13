# pi0.5 integration

This package makes the pinned openpi/pi0.5 backend use MolmoSpaces' shared data and
exact-state evaluation contracts. The backend source lives under `third_party/openpi`;
no external OpenPI or MonoPi checkout or virtualenv is required.

The same source runs on GPU and TPU. OpenPI remains pinned to JAX 0.5.3, with separate
repository-local environments because the CUDA and TPU hardware plugins differ:

```bash
scripts/setup/pi05_env.sh gpu   # .venv/pi05-gpu
scripts/setup/pi05_env.sh tpu   # .venv/pi05-tpu, run this on the TPU VM
```

To pre-download dependencies on a login host, set `PI05_INSTALL_ONLY=1`; a normal invocation on the
accelerator must still be run afterward to verify the JAX backend. GPU CI/smoke can be submitted with
`scripts/cluster/pi05_env_gpu_smoke.sbatch`.

The integration never reads from or writes to external `openpi`, `openpi_gpu`, `monopi`,
or `monopi_gpu` working copies.

The main MolmoSpaces environment remains responsible for simulation and the evaluation client.
The Pi environments are responsible for Pi training and the Pi policy server. This process boundary
is intentional: vendored OpenPI pins MuJoCo 2.3.7, while the simulator uses a newer MuJoCo API.

## Foundational model contracts

Training loss, validation loss, flow sampling, and policy-output semantics are foundational model
contracts. Preserve the behavior of the pinned pretrained implementation unless a demonstrated bug
or inadequacy requires a change. Any alternative must use a new named config and receive explicit
approval before it is used for training or reported evaluation.

For Pi0.5 action flow, Gaussian noise and Euler integration cover all 32 model action coordinates.
Physical robot controls are left-packed and padded clean targets are zero. Two loss contracts are
named precisely: the Stanford **original valid-action loss** supervises only the physical prefix
(`action_loss_coordinates="physical"`), while the **new padded-action loss** supervises all 32
coordinates (`action_loss_coordinates="model"`). Both sample and integrate all 32 coordinates.
At the policy boundary, the server first retains each embodiment's normalized valid prefix and
then unnormalizes only those physical coordinates; invalid width-10 wire slots are constructed as
zeros afterward. MolmoSpaces does not use noncontiguous state/action layouts.

For discrete Pi0.5 state input, corrected configs normalize the fixed-layout tensor, keep the
left-packed coordinates selected by `proprio_mask`, and then tokenize only those physical values.
Legacy configs retain fixed-width state serialization so existing checkpoints remain reproducible.

## Train

```bash
python -m molmo_spaces.training.main list
python -m molmo_spaces.training.main show pi05_lowvar_v2_lora_h32_20k
```

### Recommended recipe for new JOINT production runs

Start an official run from `october28_pop34fk_qanchor_posquant_b1024` (BoAI, 2026-11-06). Its predecessor and
baseline is `october27_pop34fk_qanchor_b1024`, which differs only in using pooled Gaussian action statistics. The
evidence is in `notes/20261027_pop34fk_urma4flat_training.md` (Q3 and "Recommended training recipe").

- **Arm actions:** `QueryAnchoredJoint()` (openpi `DeltaActions`; see [Arm action forms](#arm-action-forms-pi05inputconfigarm_action)).
  On pop34fk this is worth +29 points over per-step deltas.
- **Normalization:** `GroupedChunkPositionQuantileNorm(groups)`. Each action chunk position maps its own q01/q99 to
  −1/+1, and the state does the same. This is openpi's quantile mode with per-position tables; openpi's own
  norm-stats code pools the positions.
- **Everything else:** H32, global batch 1024, 80K updates (warmup 100 to 4e-5, cosine to 2e-5), AdamW with clip 1,
  full fine-tune of `pi05_base`, absolute URMA4-flat embodiment token.
- **What did not help:** batch 2048 (faster per update, not per sample), per-position Gaussian statistics, and H50.
- **Evaluation:** with `Pi05PolicyConfig.arm_action="query_anchored_joint"` and K16.

```bash
python -m molmo_spaces.training.run show --experiment october28_pop34fk_qanchor_posquant_b1024 --runtime pi05_gpu
python -m molmo_spaces.training.run norm-stats --experiment <your copy> --runtime pi05_gpu   # new data: new asset
python -m molmo_spaces.training.run train --experiment <your copy> --runtime pi05_gpu --exp-name <YYYYMMDD>_<key_differences>
```

For new data, register a copy under its own name and change only the data, the cohort and the normalization asset
pins. The per-position asset depends on the dataset and the horizon, so compute and pin it before training.

Production names (BoAI, 2026-11-06):
- **oer11-prod** is this recipe on pop34fk, in `io_v1_stored`.
- **oer12-prod** is `november06_xsrc40_exclv2_qanchor_posquant_urma4flat_io2_b1024`: the recipe on xsrc40 exclusion-v2,
  in `io_v2_canonical`, with the canonical URMA4 bundle.

See [Model I/O presets](#model-io-presets-pi05_integrationmodel_iopy).

After allocating resources, execute the experiment directly with `molmo_spaces.training.run`.
The runtime reads artifact locations and FSDP width from the explicit environment. Training
validates the native OpenPI config against the repository experiment and writes the fully resolved
scientific/runtime record beside the checkpoints.

`--exp-name` is required and is the canonical W&B display name for both fresh and resumed runs.
Use a descriptive, stable name; training rejects missing or whitespace-only names instead of
allowing W&B to generate an unrelated random display name.

For Slurm, resource choices stay in the ordinary `sbatch` command rather than the experiment:

```bash
sbatch --partition=<partition> --gres=gpu:h200:1 --cpus-per-task=16 --mem=224G \
  --job-name=<run> --output=slurm/%x_%j.out \
  --export=ALL,MOLMOSPACES_TRAIN_DATA_DIR=<rlds>,MOLMOSPACES_CHECKPOINT_DIR=<ckpts>,MOLMOSPACES_NORM_ASSETS_DIR=<norm-root>,MOLMOSPACES_NORM_ASSET_ID=<norm-id>,MOLMOSPACES_FSDP_DEVICES=1,WANDB_MODE=online \
  scripts/cluster/run_command.sbatch \
  .venv/pi05-gpu/bin/python -m molmo_spaces.training.run train \
  --experiment pi05_lowvar_v2_lora_h32_20k --runtime pi05_gpu --exp-name <run>
```

For a multi-host Cloud TPU slice, deploy the same repository revision and TPU environment to every
worker, then run the control-plane launcher from a machine with `gcloud` access:

```bash
TPU_NAME=octo-tpu-v4-32 scripts/cluster/pi05_tpu_multihost.sh -- \
  .venv/pi05-tpu/bin/python -u -m pi05_integration.multihost_smoke \
  --expected-processes 4 --expected-devices 16

TPU_NAME=octo-tpu-v4-32 scripts/cluster/pi05_tpu_multihost.sh -- \
  env MOLMOSPACES_EXPECT_ACCELERATOR=tpu WANDB_MODE=online \
  MOLMOSPACES_TRAIN_DATA_DIR=<gcs-rlds> MOLMOSPACES_CHECKPOINT_DIR=<gcs-ckpts> \
  MOLMOSPACES_NORM_ASSETS_DIR=<gcs-norm-root> MOLMOSPACES_NORM_ASSET_ID=<norm-id> \
  MOLMOSPACES_FSDP_DEVICES=4 \
  .venv/pi05-tpu/bin/python -u -m molmo_spaces.training.run train \
  --experiment <experiment> --runtime pi05_tpu --exp-name <run>
```

The launcher does no provisioning or experiment resolution: it quotes and starts the supplied
command on every worker of an existing TPU VM. The TPU entrypoint initializes JAX's
distributed system before querying devices; the registered batch remains global and is divided
equally across processes. Checkpoints must use shared remote storage such as GCS. Orbax save and
restore remain collective, while checkpoint-root mutation, metadata, progress output, dataset
audit, and W&B are process-zero-only. A `v4-32` exposes four processes and 16 global JAX devices;
the default TPU FSDP width of four keeps the FSDP axis within each four-device host and uses the
four hosts as the data-parallel axis.

A multi-process GPU run sets `MOLMOSPACES_JAX_DISTRIBUTED=1` in every process and runs the same
training command in each, for example one `srun` task per node running
`scripts/cluster/pi05_train_gpu.sbatch`. The startup does the following:

- **Process group.** JAX finds the coordinator and the process ranks itself under Slurm (`srun`)
  and Open MPI 4 (`mpirun`). Any other launcher (Open MPI 5, torchrun, Kubernetes, a custom script)
  sets `JAX_COORDINATOR_ADDRESS` (host:port of process 0), `MOLMOSPACES_JAX_NUM_PROCESSES` and
  `MOLMOSPACES_JAX_PROCESS_ID` in every process, plus optionally `MOLMOSPACES_JAX_INIT_TIMEOUT_S`
  (how long processes wait for each other; JAX's default otherwise). `scripts/cluster/xcloud_train/`
  sets these for XManager xcloud H100 pods (one process per pod driving its 8 GPUs, 8 x 8 H100 by
  default).
- **GPUs per process.** These follow the launch layout:
  - one process per node drives every GPU it sees;
  - processes that see all of a node's GPUs split them evenly by node-local rank (`LOCAL_RANK`,
    `OMPI_COMM_WORLD_LOCAL_RANK` or `SLURM_LOCALID`, innermost launcher first);
  - a process that sees fewer GPUs than the node gives the job (`SLURM_GPUS_ON_NODE`, else what
    `nvidia-smi` lists) was bound to them, for example by `--gpus-per-task`, and drives those.

  `JAX_LOCAL_DEVICE_IDS` overrides this. A process count that does not divide the node's GPUs
  fails at startup. NCCL's peer-to-peer and shared-memory transports fail (`invalid device
  ordinal`) between processes on one node that cannot see each other's GPUs. Bound processes
  that share a node therefore default to `NCCL_P2P_DISABLE=1` and `NCCL_SHM_DISABLE=1`, which
  routes their traffic over the network transport. Unbinding keeps the faster transports.
- **Uniform GPUs.** Every process must drive the same number of the same GPU type, or startup
  fails with the per-process layout. Mixed types would otherwise fail in the first compile
  (`Expect autotune result cache hit`), because XLA shares autotuning results across processes.
- **Shared checkpoint directory.** The checkpoint directory must be shared by every process: `gs://`,
  or a network filesystem (NFS, Lustre, GPFS, ...) mounted at the same path on every host. Process 0
  writes a token there that every other process must read back within two minutes, so a host-local
  disk fails at startup. That wait also covers NFS clients, which cache a failed lookup of the new
  root for up to a minute. A gcsfuse mount passes this check but cannot hold Orbax checkpoints,
  because Orbax renames directories; use the `gs://` path instead.
- **Checkpoint step lists.** Every process must list the same checkpoint steps before a restore,
  because Orbax restores and deletes collectively.

Mind the interconnect. SC's IRIS GPU nodes are linked only at 1 GbE (NCCL all-reduce 0.10 GiB/s,
measured 2026-10-26). There, two A40 nodes took about 31.6 s per update at global batch 16, while
one 4x L40S node takes 5 s at batch 128, so on SC scale within one node.
`notes/artifacts/20261026_pi05_gpu_multinode/` has the smoke and the bandwidth benchmark.

Production TPU bundles use `scripts/setup/pi05_tpu_job_setup.sh`. The setup first calls
`scripts/setup/ensure_gsutil_compiled_crcmod.sh`, which idempotently installs the compiled
`crcmod==1.7` extension and verifies that legacy `gsutil` can see it. This is required by older TPU
VM images when a pretrained checkpoint contains composite GCS objects; do not duplicate this
installation in individual request YAMLs. The same setup verifies the pinned `certifi` bundle, and
`molmo_spaces.training.run` configures `CURL_CA_BUNDLE`, `SSL_CERT_FILE`, and
`REQUESTS_CA_BUNDLE` before importing TensorFlow/JAX-backed training code. This keeps long-running
GCS dataset reads independent of a TPU VM image's `/etc/ssl` contents. New durable jobs should add
`python -m molmo_spaces.training.transport_smoke --uri "$MOLMOSPACES_TRAIN_DATA_DIR"` to their
readiness barrier; allocation and JAX readiness alone do not prove that the TensorFlow data path
can authenticate and read GCS.

For a scheduler-managed job that may be retried after a checkpoint exists, the initial
`launch.command` and `launch.resume_command` are separate contracts. The resume command must invoke
the same experiment, runtime, and stable `--exp-name` with exactly one `--resume`; copying the fresh
train command unchanged will fail only after allocating a TPU. A recovery-only request must also
put `--resume` on its initial command and add a read-only exact-step gate to readiness. Validate the
raw YAML/JSON before submission:

```bash
python -m molmo_spaces.training.tpu_request_smoke \
  --request /path/to/request.yaml --expected-checkpoint-index 2999 \
  --require-initial-resume
```

For a fresh request, omit `--expected-checkpoint-index` and `--require-initial-resume`; this validates
the future retry command without imposing a fixed checkpoint on the first launch. For an
exact-step recovery request, use both options as above. The check rejects `--overwrite`,
scientific-identity disagreement between the two commands, a non-GCS checkpoint root, a missing
resume flag, and (when an exact checkpoint index is supplied) readiness that does not run
`molmo_spaces.training.run verify-resume` for the restored training state. Checkpoint directory
`N` stores the state that will execute update `N+1`, so index `2999` must restore state `3000`.
An exact recovery request must set `policy.max_attempts: 1`: after it makes progress, the pinned
readiness step is stale, so an infrastructure loss requires a new request bound to the newest
fully committed checkpoint instead of an automatic retry.


For retryable recovery, use `verify-resume --minimum-resume-step N` instead of
the exact-step option. This requires the latest fully restored state to have at
least N completed updates, while retaining the same namespace, optimizer restore,
input-distribution and deterministic sampler/topology checks. It permits a later
infrastructure retry to restore a newer checkpoint without weakening the floor.
Validate such a request with `tpu_request_smoke --minimum-checkpoint-index N-1
--require-initial-resume`; it may use multiple attempts. Never combine exact and
minimum verification modes. Neither mode takes an optimizer step or writes artifacts.
Build that replacement from the prior immutable request without retyping its scientific or asset
contract:

```bash
python -m molmo_spaces.training.tpu_recovery_request \
  --request /path/to/prior-request.yaml \
  --output /tmp/recovery.yaml \
  --request-id <new-unique-request-id> \
  --checkpoint-index 3999
```

The builder preserves the experiment, runtime, stable run name, checkpoint root, environment,
dataset, normalization, embodiment assets, and resource options. It adds one `--resume` to both
commands, replaces any stale exact-step readiness clause with state `index + 1`, removes persisted
creation metadata, and forces one scheduler attempt. It validates the result before writing it.
For a portfolio alternate, repeat `--resource-id <id>` to retain only exact named resource options
from the source request; omitted resource IDs are rejected rather than guessed.

### Preemption-recovery checkpoints

Two settings, both in completed updates, set the checkpoint cadence: `save_interval` (how often
a checkpoint is written, default 1000) and `keep_every_n_completed_updates` (which ones are kept,
default 4000). Recipes set them as `checkpoint_every_updates` and
`checkpoint_retention_every_updates`; `run.py train --save-interval/--keep-interval` overrides
them for one launch. Saves land in two tiers, both under the zero-based update index `N` (the
state that executes update `N+1`):

| Tier | Path | Cadence and retention | Contents |
|---|---|---|---|
| Milestone | `<checkpoint_dir>/<N>` | every `keep_every_n_completed_updates` updates and the final step; all kept | full params, optimizer state, assets; what evaluation and staging read |
| Resume | `<checkpoint_dir>/resume/<N>` | every other `save_interval` save; only the newest step is kept, and a run that finishes deletes it | see below; no assets |

A short `save_interval` (say 100) therefore bounds the work a crash loses without accumulating
checkpoints. Older configs that set the legacy zero-based `keep_period` make every save a
milestone and let Orbax apply `keep_period`; the resume tier then holds only preemption saves.

With EMA off and a freeze filter (LoRA), a resume step stores only the non-frozen parameters,
the optimizer state and the step. The frozen parameters are written once per run directory to
`<checkpoint_dir>/resume_frozen_params` and merged back on restore. Other configs store the full
state. Saves are asynchronous: training blocks only for the device-to-host copy
(`perf/resume_save_blocking_s`). If an earlier resume save is still writing on any process, the
save is skipped (`perf/resume_saves_skipped`) rather than waited for.

All ranks validate the previous `input_contract.json` before a startup barrier. The primary
then publishes its replacement atomically (same-directory rename on POSIX), so concurrent
readers never observe a truncated contract. Validation failures remain fatal.

A resume restores the newest restorable step across both tiers; a milestone wins at an equal
step, and startup deletes the other resume steps and any interrupted resume write.
`DeterministicReplayV3` replays from the restored step, so a same-topology resume from either tier
continues the exact frame stream.

On SIGTERM the trainer saves a resume checkpoint at the next update boundary, waits for it, and
exits with status 143. Under `jax.distributed` (every TPU run) the coordination service makes all
processes agree on that step. A single GPU process handles SIGTERM and SIGUSR1 itself. On SC,
preemption, `scancel` and the time limit send SIGTERM to every process in the job's cgroup and
SIGKILL 30 s later (`KillWait`); `iris-hi` is not preemptible. A LoRA resume step takes about 47 s
to write to IRIS NFS, so on SC the final save can be cut off. The periodic tier then bounds the
loss, and the next start removes the interrupted write.

An exact recovery gate (`--expected-resume-step`, or `tpu_recovery_request --checkpoint-index`)
must name the newest restorable step across both tiers, so list `<checkpoint_dir>/resume/` as
well as the milestones. A newer resume step makes an exact gate fail closed. Retryable requests
should use `--minimum-resume-step`.

**Opt-in: unified checkpoints** (`TrainConfig.unified_checkpoints`, or `run.py train
--unified-checkpoints` for one launch; off by default, so every registered experiment keeps the two
tiers above). Every save is then the same full checkpoint at `<checkpoint_dir>/<N>`: parameters (EMA
parameters when EMA is on), optimizer state and `assets/` (normalization statistics and manifest,
embodiment descriptions), so any checkpoint can be staged and evaluated as soon as it commits. Saves
differ only in retention: a milestone is kept; an off-grid `save_interval` save or a preemption save
is kept until the next save commits, then deleted by Orbax. A resume save is still skipped while an
earlier save is writing on any process; a milestone or preemption save waits for it. A resume tier
an earlier start wrote is restored from while strictly newer than the run directory and cleared once
the run directory holds a checkpoint at least as new. Startup deletes interrupted step saves: Orbax
temporary directories, and on GCS step directories without `commit_success.txt` that are newer than
every committed step (Orbax's own `cleanup_tmp_directories` would also delete `xla_cache/` and the
resume tier there); an older unmarked step, such as a checkpoint copied from local storage, is kept
with a warning. An off-grid step can be deleted by the next save while a tool copies it: copy it soon
after it commits and rely on `boai-stage-gcs-checkpoint`'s manifest check to reject a partial copy.
An evaluation records the checkpoint's identity before its server loads it for the same reason. An
exact recovery gate names the newest committed step in `<checkpoint_dir>/`. Why it exists (PR #84,
2026-10-28): the resume tier halves the bytes per LoRA resume save but saves no storage at rest and
no training time, and leaves a run's newest state unevaluable until the next milestone; full resume
saves cost about 13 s of a 6,800 s save interval (0.2%) on the October 27 pop34fk v4-32 run.
GPU smoke: `notes/artifacts/20261028_unified_checkpoints/gpu_unified_checkpoint_smoke.sbatch`.

Startup runs its three independent parts at once: building and positioning the input stream (a
background thread), restoring the checkpoint or loading the base weights (the main thread), and
compiling the train step (a second thread). A restart costs about the slowest of them rather than
their sum. The compile targets the abstract train state with its shardings, which initialized and
restored states share exactly, so every start of a run lowers the same program and the loop's first
update reuses it. That thread lowers the train step inside `sharding.set_mesh`, as the loop runs it:
jit reuses a lowered program whatever the mesh state, so a lowering without the mesh would train
without the activation sharding constraints (about 9x slower per update on a v4-32). `set_mesh`
applies to its own thread only, so the main thread's concurrent tracing is unaffected. The compile needs the first batch's shapes, so it starts once the input is
positioned. With more than one process the state is created or restored only after the first batch
exists: building that batch launches device programs (the image conversion in
`Observation.from_dict`), and multi-controller JAX needs every process to launch programs in the same
order. Overlapping them with the restore's cross-process syncs aborted a v4-64 resume (2026-10-29,
"an unexpected peer shows up in the launch group with a different launch id"), so a multi-host
restart overlaps the input with the compile only and pays the restore on top. `Startup overlapped: ...` logs when each part finished; a warning reports a state whose
signature differs, which costs one more compile. A resume skips the dataset mixture audit, which
the run's first start already reported and which never compared starts; whether the dataset files
changed is checked by the input contract's dataset fingerprint (see "RLDS input distribution across
processes").

Compiled executables are cached in `<checkpoint_dir>/xla_cache` (the JAX persistent compilation
cache: process 0 writes, every process reads). A restart with the same code, accelerator type and
topology then loads the train step from the run's own storage instead of compiling it, on any
machine. An explicit `JAX_COMPILATION_CACHE_DIR` overrides the location; for example
`scripts/cluster/pi05_train_gpu.sbatch` keeps its shared IRIS cache. `xla_cache` is not part of a
checkpoint: step discovery ignores it, and deleting it costs only a recompile.

Measured cost and the choice of default interval: `notes/20261018_preemption_resume.md`.

URMA jobs must also materialize their registered robot-description bundle. Set the exact
`MOLMOSPACES_EMBODIMENT_BUNDLE_{SOURCE,DESTINATION,NPZ_SHA256,MANIFEST_SHA256}` contract in the
resource request; the shared TPU setup downloads only the two pinned files and verifies both
hashes. Add `molmo_spaces.training.embodiment_asset_smoke` to readiness so a missing or mismatched
description fails before W&B initialization or optimizer construction. The resource storage plan
must list this input prefix and its materialized region.

The canonical comparison config uses H=32, LoRA, the shared mild image augmentation,
a deterministic 95/5 train/validation split, unchunked action normalization, and a
combined future-timestep/action-dimension loss mask.

### RLDS input distribution across processes

`rlds_input_sampling` in every experiment config decides how the TFDS/RLDS stream is split
across JAX processes and reordered between passes. The default for all new experiments is
`DeterministicReplayV3()` (since 2026-09-06):

1. one deterministic permutation of the TFDS file identities shared by all processes, seeded
   from the sampler seed only;
2. TFDS `InputContext` sharding, so `jax.process_count()` processes read disjoint file cohorts
   whose union is the whole split (TFDS fails closed if a process would receive no file). The
   validation reader is sharded the same way, so a global validation batch is one draw from
   the held-out view rather than `process_count` overlapping local draws;
3. a seeded per-process reshuffle of that cohort on every pass, deterministic in tf.data;
4. the process-local 50K-frame shuffle after chunking and flattening, with image decode after
   the shuffle.

A same-topology resume replays the exact encoded stream and skips
`completed_updates x local_batch` frames, so an interrupted run consumes the same frames as an
uninterrupted one. The run's `input_contract.json` records the contract, process count, local
batch size, sampler seed, and TF/TFDS versions beside the checkpoints. If any of those differ at
resume (for example a Spot migration to a different slice size), V3 fails closed unless the
launch passes `--rlds-input-topology-restart`, which records an explicit distributional restart
in the contract and reseeds from the completed update count.

The contract also fingerprints the dataset files, for every sampling version
(`openpi.training.dataset_fingerprint`): per dataset, a hash of `features.json` and, per split, the
episode and shard counts, a hash of the episodes per shard and a hash of each shard file's name and
size. Every start records it, which costs the TFDS metadata and one size query per shard. A resume
on files that differ from those of the run's last start (a re-converted build, added or removed
episodes, an in-place edit that changes a shard's size, a dataset added to or removed from the mix)
fails before the checkpoint restore and names each difference. The fingerprint is keyed by the path
below the data root and leaves out modification times, so the same files staged elsewhere or copied
to another region's bucket match; it does not hash contents, so an edit that keeps every size is not
detected. To continue on changed data on purpose, for example finetuning or continuing a run on a
new mix, pass `--rlds-input-dataset-restart` with `--resume`: the check is skipped, the differences
are logged and appended to the contract's `dataset_changes`, and a V3 stream restarts at a derived
seed instead of replaying the old position over new files. A lineage that predates fingerprints is
recorded at its next resume, with a warning that this resume could not be checked.

`ProcessShardedFileShuffleV2()` was the default from 2026-09-05 to 2026-09-06. It shards the
training reader identically but reseeds the file permutation from `completed_updates` on every
resume (so a frequently preempted run over-exposes cohort heads) and leaves the validation reader
unsharded. It remains available as an explicit choice for lineages that already recorded it.

`LegacyRldsInputSampling()` is the pre-2026-09-05 behaviour: every process traversed the same
ordered stream and only the downstream frame-buffer seed differed, so a four-host global batch
of 128 was roughly four copies of one 32-frame local batch and a 40K-update run never reached
the last ~30% of the file list (1,640 of 8,000 Camera-D robots were never optimizer-exposed; see
`notes/20260904_embodiment_training_exposure_audit.md`). Experiments registered before the fix
keep it through the explicit pin list in `molmo_spaces/training/legacy_rlds_input_sampling.py`
so their names and checkpoints keep their recorded semantics. Do not add new names to that list;
a corrected rerun of a legacy lineage is a new named experiment.

Before launching on a new dataset or topology, run
`scripts/analysis/audit_rlds_process_sharded_exposure.py` to record per-process episode and
frame counts, and `scripts/analysis/smoke_rlds_process_sharded_reader.py <processes> <passes>
<out.json>` (Pi environment, CPU) to prove cohort membership and per-pass reordering on the
real TFRecords.

Episode selectors, hash/stratified splits, and V3 cohort accounting need every episode's
identity before the first batch. If the build carries
`molmospaces_episode_index.v1.json` (see `README_RLDS.md`, "Episode identity index"), the
loader reads it and validates it against the split's shard fingerprint; otherwise it scans the
split as before, on every process, at every launch and resume. Build the index once with
`scripts/data/format_conversion/build_rlds_episode_index.py`.

**Mixtures of builds that reuse local embodiment IDs.** A population collection converts each family to its own
build, and every family numbers its robots from the same candidate ID (3000000, 3000001, ...), so the same raw
`episode_metadata/embodiment_id` names different robots in different components. `NamespacedHashEmbodimentSplit`
already ranks `(component, ID)` pairs, but per-embodiment normalization and the model see only the ID. The opt-in
`RldsDatasetComponent.embodiment_id_offset` (mirrored by `RLDSDataset.embodiment_id_offset`, default 0 = raw IDs)
adds a per-component constant wherever the loader uses the ID (the emitted `embodiment_id`, split and selector
identities, normalization routing), so components get disjoint model-facing IDs; an evaluation client must send
the same shifted ID. `scripts/data/format_conversion/build_family_mixture_cohort.py` writes the cohort record of such
a mixture (family, offset = family index x stride, per-robot arm joint count, episodes, frames, control periods) from
the builds' records without decoding images, and refuses a family whose raw IDs span the stride or a repeated
model-facing ID. First user: the xsrc40 exclusion-v2 cell (`molmo_spaces/training/nov05_xsrc40_exclv2_training_contract.py`).

### Frame sampling and immutable normalization

Repository Pi recipes use typed frame-processing strategies from
`pi05_integration/data_processing.py`. `NaturalFrameSampling()` is the only supported strategy:
the natural frame distribution of the selected episodes is the training contract. The historical
`PostCloseBalancedFrameSampling(weight=...)` resampled frames around the gripper-close event; it is
refused when a training dataset is built, because rebalancing around a task-specific event bakes a
per-task recipe into the input pipeline and doubles host input work. The five July 2026 configs
that recorded it stay loadable for provenance and evaluation but cannot be trained or resumed.
Change the data, not the sampler.

Validation exists only when the config defines held-out data. A typed episode split (hash, frozen
holdout, anchored, stratified, or contiguous) is the normal way; a leading percent slice such as
`train[:95%]` with no episode split is also accepted because its complement `train[95%:]` is
disjoint. Training on the full `train` split with no episode split and a nonzero `val_interval`
is refused at launch (`validation_source_split`), and with `val_interval=0` no validation loader is
built and nothing is logged under `val/`. Four August 25 lineages predate this guard and validated
on training data; their names are listed in `tests/test_validation_and_frame_sampling_guards.py`.

Episode partitioning is independently typed. Repeated-episode recipes use
`StratifiedEmbodimentEpisodeSplit(validation_fraction=0.05, seed=...)`: it ranks stable RLDS
`episode_metadata.file_path` identities within each global embodiment ID and holds out episodes
from every embodiment (with at least one episode on each side). One-episode-per-embodiment scaling
recipes instead use `AnchoredHashEmbodimentSplit`: it holds out complete embodiment IDs across the
union of all configured RLDS components while pinning named train and validation identities from a
smaller reference population. Both strategies run before frame flattening, chunking, shuffling, or
normalization. Historical configs retain their contiguous TFDS slices so existing checkpoints and
results remain reproducible. Multi-component sampling uses the experiment seed for a reproducible
component-selection sequence.

Normalization has two separately configured parts: the typed scheme in `Pi05InputConfig` (for
example `GlobalQuantileNorm()` or `PerEmbodimentQuantileNorm(ids)`) and an immutable dataset asset.
Build the asset explicitly before training:

```bash
.venv/pi05-gpu/bin/python -m molmo_spaces.training.run norm-stats \
  --experiment <experiment> --runtime pi05_gpu
```

The command scans the finite training split exactly once, computes exact statistics, and writes
`norm_stats.json` plus `normalization_manifest.json` containing dataset identity, split, frame
count, scheme, and SHA-256. New production experiments set
`require_normalization_manifest=True` and pin `normalization_stats_sha256` in the registered
training config. Training fails if the file, identity, scheme, split, or hash differs; launchers do
not recompute or overwrite it.

Per-position chunk statistics keep one float32 global action payload and use basic-slice views
for per-embodiment/group rows. The views preserve the original per-ID row order and masked NaN
bits, so global and grouped reducers receive the same values in the same order. This avoids a
second full H32/H50 chunk payload (approximately84/131GB for65.3M frames). Large-table reduction
still uses the existing one-cell-at-a-time float64 moments and exact quantiles; pooled-action
statistics also share valid frame ranges via prefix row offsets. Above the existing large-table
threshold, pooled reduction gathers each float64 column directly from the original batches, avoiding
a full N-by-D concatenation. Column values, order and all arithmetic match the prior large-table
path exactly. Small-table arithmetic and sampling are unchanged. Plan RAM for states, loader and reduction
temporaries. Byte parity is covered for both whole-table and per-cell reducers, including weighted
mixtures; the [OCI ablation record](../notes/20261028_pop34fk_ablation_oci.md) records deployment evidence.

Each checkpoint copies the verified `norm_stats.json` byte-for-byte and writes
`normalization_bundle.json` beside it. Policy construction always loads this checkpoint-local
processor and validates its hash and scheme; runtime normalization overrides are rejected for
bundled checkpoints.

One explicit shared-statistics contract supports
`GroupedChunkPositionStandardNormQuantileState` and `GroupedChunkPositionQuantileNorm`:
with a required manifest and matching pinned SHA, these may consume the same immutable per-position
asset when groups, mixture weighting and every other fitting field match. Every global/group
action table must contain finite `(H,D)` mean/std/q01/q99 arrays of one shape. The source fitting
manifest and statistics remain unchanged; the checkpoint bundle records its consuming mode and
`statistics_source_normalization` provenance. Other mode mismatches still require a separately
registered asset. Checkpoint loading remains strict about its saved consuming mode. Historical checkpoints without a bundle retain their existing checkpoint-local
compatibility path and parameter shapes.

State and actions share the scheme's mode unless the scheme splits them. Pi0.5 reads the state only as prompt
tokens (256 bins over [-1, 1); `third_party/openpi/src/openpi/models/tokenizer.py`), so under a mean/std scheme every
state value beyond one standard deviation lands in an end token: 22-35% of state values in the OER training sets
measured on 2026-10-25 (`notes/20261025_twin_gen3_check.md`, Q4). `GroupedStandardNormQuantileState(groups)` keeps
`GroupedStandardNorm` for actions (mean/std, clipped at +-5) and maps each state dimension's q01..q99 to -1..+1 (about
2% outside) from the same `norm_stats.json`; trainer and server apply the split through
`NormalizeByEmbodiment(state_quantiles=True)` (other schemes keep one mode for both fields). The normalization
manifest records the scheme, so reusing existing statistics means registering them under a new asset ID with
`molmo_spaces.training.normalization_asset_tools.clone_stats_payload`. On the twin pick-and-place check it matched
mean/std state within noise (16 vs 14/200 held-out, 69 vs 64/160 seen layouts at matched steps); prefer it for new
recipes because it keeps the state readable, not for a measured gain.

Response-history experiments add `CausalResponseHistoryBackboneToken` independently of URMA/AdaRMS.
It consumes exactly nine measured rows `(arm qpos[6], physical aperture metres)` and eight prior
sent-target rows `(absolute arm target[6], semantic gripper target 0=open/1=closed)`, plus temporal
validity masks. No explicit q differences are provided. Valid rows use their own frozen
per-coordinate Gaussian tables `response_{state,action}_history_gaussian`; padded rows are restored
to exact zero after normalization. The full `norm_stats.json` SHA-256 is pinned by the typed input
config and checkpoint normalization bundle. The evaluator records one measured row on every action
step (including render-free chunk playback), forms the payload before the current command, and only
then appends the absolute target actually sent, matching RLDS construction exactly.

### Typed input contract and training visualization

Pi0.5 recipes compose four independent typed strategies from
`pi05_integration/input_config.py`:

- prompt tokens, for example `EmbodimentIdPrompt()` or
  `MorphologyTextPrompt(include_id=True)`;
- state-token augmentation, for example `EmbodimentOneHotState(ids)`;
- reversible data normalization, for example `PerEmbodimentQuantileNorm(ids)`;
- model-side action-expert modulation, for example
  `LearnedEmbodimentAdaRMS(vocab_size=64)`.

The old scalar attributes remain read-only compatibility views for existing loaders, servers,
result readers, and checkpoint config names. They resolve to the same strings, codebooks,
normalization transforms, AdaRMS vocabulary size, model parameter names, and tensor shapes as
before this refactor. In particular, old LoRA/full-FT checkpoints remain restorable with their
registered config names. New runs additionally write `input_contract.json` beside the checkpoints.

### Image geometry (`Pi05InputConfig.image_geometry`)

Default `StoredImageGeometry()`: frames as stored (sim RLDS stores 848×480 renders stretched to a square).
`LetterboxStretched()` (`letterbox_stretched_v1`, opt-in, new config name) un-stretches each sim frame at load time
into the `resize_with_pad` letterbox, the Pi/DROID convention; the eval client must send
`image_geometry="letterbox_stretched_v1"`, and the server refuses a request in another geometry. Contract, box
arithmetic and tests: `scripts/data/format_conversion/README_RLDS.md` ("Image geometry").

### Gripper value (`Pi05InputConfig.gripper_value`)

Default `StoredGripperValue()`: the gripper slots as stored. `GripperClosedness()` (`gripper_closedness_v1`, opt-in, new
config name) trains on closedness, 0 = open, 1 = closed, in state and action. With `endpoints_table` and
`endpoints_table_sha256` (`gripper_endpoints_table.py`; xsrc40:
`artifact/analysis/20261105_gripper_endpoints/xsrc40_gripper_endpoints_table.json`, sha256 `088061a6…d269`) the loader
converts a `physical_aperture_m` build's gripper state at load, per robot by `episode_metadata/robot_artifact_hash`, so
the xsrc40 exclv2 builds need no reconversion; actions are already the command's closedness. The eval client must set
`proprio_gripper_source="gripper_closedness_v1"`, and the server refuses a request in another gripper value. Contract:
`scripts/data/format_conversion/README_RLDS.md` ("Canonical gripper value"); code `pi05_integration/gripper_value.py`;
tests `tests/test_pi05_gripper_value.py`.

### Joint frame (`Pi05InputConfig.joint_frame`)

Default `StoredJointFrame()`: each robot's arm joints in its own model's convention. `CanonicalJointFrame(table,
table_sha256)` (`canonical_mdh_tool_v1`, opt-in, new config name) trains every robot on one geometric dial
(`molmo_spaces/robots/unified/DISTRIBUTION.md` section 7): state φ = θ + s·q, joint-difference actions s·a, with the
last joint's zero on the gripper's closing line. The loader reads each episode's (s, θ) from the sha-pinned table
(`scripts/data/format_conversion/joint_frame_table.py`; xsrc40:
`artifact/analysis/20261105_joint_frame/xsrc40_joint_frame_table.json`, sha256 `a809719f…2927`) by
`episode_metadata/robot_artifact_hash`, so no rebuild is needed; normalization statistics must be computed with the
config. Supported with `per_step_joint_delta` and `query_anchored_joint`; DROID components, dynamic URMA and response
histories are refused. The eval client sets `joint_frame="canonical_mdh_tool_v1"` and sends its robot's (s, θ),
measured from its catalog artifact; the server maps proprio and RTC prefix in and the actions back, and refuses a client
in another frame. Code `pi05_integration/joint_frame.py`; tests `tests/test_pi05_joint_frame.py`.

### Model I/O presets (`pi05_integration/model_io.py`)

A preset names one consistent set of the three load-time representation fields above. A production config moves
between representations as a unit:

| Preset | `image_geometry` | `gripper_value` | `joint_frame` | URMA4 bundle | Used by |
|---|---|---|---|---|---|
| `io_v1_stored` | `stretch` (default) | `stored` (default) | `stored` (default) | stored | every checkpoint through oer11-prod |
| `io_v2_canonical` | `letterbox_stretched_v1` | `gripper_closedness_v1` | `canonical_mdh_tool_v1` | canonical (`--joint-frame-table`) | oer12-prod |

- `io_v2_canonical(...)` takes the dataset's gripper endpoints table and joint-frame table with their sha256.
  `ModelIO.apply(input_config)` sets the three fields.
- `identify(input_config)` names a config's preset, or returns None for a mix: the R2 exclv2 cell has letterbox and
  closedness but the stored joint frame.
- `ModelIO.policy_fields()` gives the `Pi05PolicyConfig` fields an evaluation sets.
- A URMA4-absolute bundle must be in the preset's joint frame. The extractor's `joint_frame` field is checked against
  the bundle's manifest, and `PiExperimentConfig` refuses a mismatch.
- Not built: the server does not yet announce its preset to the client, so evaluation configs set the fields
  themselves.

The production recipe on the xsrc40 exclusion-v2 set is `nov06_xsrc40_exclv2_posquant_contract`, with two cells:
- `november06_xsrc40_exclv2_qanchor_posquant_urma4flat_io1_b1024`, the bridge: oer11-prod's recipe and I/O on the
  exclv2 data;
- `..._io2_b1024`, oer12-prod: the bridge in `io_v2_canonical` with the canonical bundle.

The campaign note is `notes/20261106_oer12_prod_training.md`.

### Arm action forms (`Pi05InputConfig.arm_action`)

What the H32 action chunk predicted at query step `t` means for the arm (`pi05_integration/arm_action.py`).
The gripper is always the absolute command; `pose9` is `[x, y, z, R[:, 0], R[:, 1]]` in the robot base-link frame.

| form | arm target `k` of the chunk | data |
|---|---|---|
| `PerStepJointDelta()` (default) | `q_cmd(t+k) - q_meas(t+k)`: the RLDS `steps/action` itself | any joint build |
| `QueryAnchoredJoint()` | `q_cmd(t+k) - q_meas(t)` (openpi `DeltaActions`) | any joint build (`action + proprio`) |
| `QueryAnchoredBaseTcp()` | `dp = p_cmd(t+k) - p_meas(t)` along base axes, `dR = R_cmd(t+k) R_meas(t)^T` as rotation6D | joint build with `--arm-reference-features` |
| `QueryAnchoredJointAndBaseTcp()` | both in one 20-slot chunk: `[joint (n), TCP (9), gripper]`; state `[q (n), aperture, TCP pose9]` | joint build with `--arm-reference-features`, left-packed layout |

The default form leaves every existing recipe, dataset and checkpoint byte-identical. For the
query-anchored forms the loader builds the chunk label from the absolute targets, the static-frame
filter compares against the form's rest value (identity rotation for TCP labels), and
`compute_norm_stats.py` pools the action statistics over in-episode chunk positions (state stays one
row per frame); a query-anchored recipe therefore needs its own normalization asset. DROID data refuses
non-default forms; mobile-base recipes accept `QueryAnchoredJoint()` (the 10-wide arm block keeps its width, and the
base chunk is built beside it unchanged) and refuse the TCP forms.

The eval client (`Pi05PolicyConfig.arm_action`, `eval_replay.py --arm_action`) measures the anchor at the
query step and turns every executed step into the native per-step joint delta (`target - measured`),
which then takes the unchanged execution path (source velocity slew, gripper mapping). Receding horizon
only (no RTC or temporal ensembling). The TCP forms solve one IK step per action with the bounded
continuation DLS (`molmo_spaces/kinematics/bounded_continuation_dls.py`) inside the source velocity box,
so they need `source_declared_velocity_v2`; a degenerate rotation or an empty box ends the rollout with a
recorded `policy_action_rejection`. The joint + TCP checkpoint executes one head (`arm_action_head`:
`joint` or `ee`). The server refuses a request whose `arm_action` differs from the checkpoint's.

Registered recipe: `october26_oer9_joint_urma4_queryanchored_fullft` (the october13 urma4 cell with
`QueryAnchoredJoint()`; evaluate with `oer9_eval plan --arm-action query_anchored_joint`). Evidence for
the forms: `notes/20261005_xemb1k_ctr_unif_joint_vs_ee_vs_eestate.md`.

Two opt-in additions for query-anchored labels (Oct-28 action-frequency study,
`molmo_spaces/training/oct28_action_freq_contract.py`); both default off and leave existing recipes unchanged:

- `GroupedChunkPositionStandardNorm(groups, mixture_weighting="frames"|"components")` keeps one mean/std row per
  chunk position k and dimension (`(H, D)` action tables; state stays per dimension), masking past-the-end
  positions like the loss. A query-anchored label grows with k, so the pooled table squeezes early positions and
  stretches late ones: on the twin n40 build the per-position arm std is 0.009-0.025 rad at k=0 and 0.10-0.21 at
  k=15 against 0.06-0.13 pooled, and the pooled table puts 0.26% of k=15 labels beyond +-5 (none per position).
  `"components"` weights a mixture by each component's sampling weight instead of its frame count. The action
  boundary and the server (default valid dim, RTC prefix) apply the table row by row and refuse a horizon
  mismatch. `GroupedChunkPositionQuantileNorm(groups)` is its openpi-quantile-mode twin (q01/q99 -> -1/+1 for every
  `(H, D)` action row and the state; the pop34fk R4 scheme, ported with the xsrc40 mixture infrastructure).
  With a mobile base the arm keeps its per-position tables and the base its one global `(3,)` table per key, read
  from chunk position 0 (each frame's own command once, as in an unchunked scan).
- `Pi05InputConfig.action_time=ActionTokenTimeSeconds()` (mirrored by `Pi0Config.action_time_conditioning`,
  Pi0.5 only) adds `tau_k = (k + 1) * action_dt_s` seconds through a zero-initialized bias-free
  `action_time_proj` to the action tokens after `action_in_proj`. The loader reads `action_dt_s` from
  `episode_metadata/policy_dt_s` (`mlspaces_to_rlds.py --policy-dt-metadata`) and fails without it; serving
  requests carry `action_dt_s` (control period, s), which a time-conditioned server requires and other servers
  ignore. The weight loader must leave `action_time_proj/.*` missing from `pi05_base`.

### Mobile manipulators: holonomic-base slots (`mobile_base_slots_v2`; v1 superseded)

**Goal.** Train and serve one Pi0.5 policy for arms on a holonomic base (base plus arm) beside fixed arms,
without changing anything a fixed arm sees. Upstream OpenPI has no mobile-manipulator policy or transform (its
only mobile reference is `is_mobile` robot metadata in the ALOHA LeRobot conversion example); its model action
is a fixed-width padded vector (`action_dim=32` for Pi0.5), so the base is three more action dimensions and no
model change is needed.

**Base semantics (two named contracts, same slots).**

| contract | RLDS conversion | base action | base state |
|---|---|---|---|
| `mobile_base_slots_v2` (`MobileBaseSlotsV2`, current) | `mobile_base_feature_v2` | commanded body-frame velocity [v_forward m/s, v_left m/s, yaw_rate rad/s], an absolute command | measured body-frame velocity, same order |
| `mobile_base_slots_v1` (`MobileBaseSlotsV1`, superseded 2026-10-25) | `mobile_base_feature_v1` | base command minus measured pose in the current base frame [forward m, left m, yaw rad] | base pose in the episode's first base frame (yaw wraps, so it jumps past 180 degrees of turning) |

v2 is BoAI's decision: body-frame velocities carry no heading or pose relative to the episode start, so a full
turn is continuous, and the command is the commanded reference's own velocity, free of servo error (definitions and
the measured consistency on a real episode in `scripts/data/format_conversion/README_RLDS.md`). v1 is kept because
keeping it costs one key map; use v2 for new work. The loader reads each contract's own RLDS keys
(`mobile_base.RLDS_FIELDS`) and refuses a component that carries the other version's fields.

**Design.** The loader keeps the 10-wide arm vector (`actions`, `state`, their masks, gripper canonicalization and
the terminal gripper hold) exactly as it is and emits the base beside it as `base_actions` (chunked),
`base_action_mask`, `base_state` and `base_state_mask`. The base gets its own global `base_state`/`base_actions`
statistics over mobile frames only and is normalized separately (in the normalization's quantile mode; a scheme
that splits modes, such as `GroupedStandardNormQuantileState`, applies its state mode to `base_state` and its action
mode to `base_actions`); `transforms.AppendMobileBase` then writes it into
the model slots after the arm block. Appending (not prepending) is what keeps fixed arms identical: the arm slots
never move, the gripper stays the highest valid slot of the arm block, and a fixed arm's base slots are zero, which
is what padding already puts there.

| model slot | action (normalized) | state (normalized) | fixed arm |
|---|---|---|---|
| 0-9 | unchanged arm vector: joint deltas, absolute gripper (slot 7 canonical, or the layout's slot), padding | unchanged arm state | unchanged |
| 10, 11, 12 | base action of the contract (v2: v_forward, v_left, yaw_rate) | base state of the contract | zero (state: not appended) |
| 13-31 | padding | padding | unchanged |

The Pi0.5 state text has 13 numbers for a mobile sample and the usual 10 for a fixed arm. `action_semantic_mask`
(metrics only) becomes 13 wide in such a recipe, false at 10-12 for a fixed arm; a physical-coordinate
`action_loss_mask` gets the base mask on valid timesteps; the default model-coordinate loss already covers every
slot. The training static-chunk filter also counts base motion, so a mobile episode's drive-only chunks are kept
(v2 adds velocities in m/s to the arm's per-step deltas, so any driving chunk clears the threshold). Past the
episode end the base chunk holds: zero offset (v1) or zero velocity (v2).

**Enable.** Set `Pi05InputConfig(mobile_base=MobileBaseSlotsV2(dt_s=...))` in a new named config (it requires the
fixed 10-wide `LegacyFixedLayoutState`). `dt_s` is the control period of the data (`episode_metadata/
base_velocity_dt_s`, `policy_dt_ms / 1000` at collection): a velocity is a displacement per period, so the loader
refuses a mobile episode recorded at another period and the server a client stepping at another one. A population
whose robots run at different control periods (xsrc40: 5-50 Hz, one per robot) uses `MobileBaseSlotsV2AnyRate()`:
the same contract, slots and fields with no single `dt_s`, so neither the loader nor the server checks the period;
each label stays the velocity at its own robot's rate, which the client executes with its own `policy_dt_ms`. Every mixture component is read with the base fields; a build without them gets
zeros with false masks. Compute normalization statistics as usual (`third_party/openpi/scripts/compute_norm_stats.py`
or `molmo_spaces.training.run norm-stats`); the scan adds `base_state`/`base_actions` and refuses a mixture with no
mobile frames. Registered example: native config `october25_mobilebase_gen3_smoke`, a pipeline smoke on the
one-episode Gen3 build `20261025_dgi_mobile-data/rlds_mobile_feature_v2_465e1681` (state-only unit, blank images;
not a training dataset), statistics in `20261025_dgi_pi_mobile/norm_assets_v2`. It used v1 (build
`rlds_mobile_feature_v1_465e1681`, statistics in `norm_assets`, both still on disk) until BoAI moved it to v2 the
same day; no run used it. Measured 2026-10-25 on v2 (Pi GPU env on CPU, script
`/iris/u/boai/artifact/h5_data/20261025_dgi_decisions/pi_mobile_v2/pipeline_smoke_v2.py`): statistics over 541
frames (`base_actions` q01/q99 = (0.0, -0.123, -1.0)/(0.4, 0.0, 1.0)); one B8/H32 batch through the production
transforms (actions nonzero in slots 0-7 and 10-12 only, 13-number state text); one dummy-width Gemma loss; one
dummy-model RTC sample (below). No optimizer step and no training result.

**Stationary-arm guarantee.** The default `NoMobileBase` emits no base fields and takes none of the new code
paths, so existing configs, normalization assets and checkpoints produce byte-identical batches (and
`Pi05InputConfig.to_record()` records `mobile_base` only when enabled). With either option on,
`tests/test_pi05_mobile_base_slots.py` checks, for v1 and v2 builds, on real converter-written TFDS builds that every
fixed-arm frame has identical loader tensors, identical model inputs and targets after the full transform stack
(state, state text tokens, actions, loss mask, images), and that the `state`/`actions` statistics are identical; and
that a mobile episode's base lands in slots 10-12 with true masks and the expected normalized values.

**Serving and evaluation.** For a mobile-base checkpoint the server additionally reads optional `base_state` and
`base_state_mask` (absent = fixed arm), the client's `base_contract` (a v2 checkpoint requires it with
`base_state` and refuses a mismatch, so a v1 client cannot drive a v2 checkpoint or the reverse) and, for v2, the
client's `base_dt_s` (refused unless equal to the checkpoint's `dt_s`), and returns
`base_actions` (H, 3), unnormalized with the base table and zero for a fixed-arm row
(`action_boundary.unnormalize_base_actions`); the dynamic batcher carries all of them. Other checkpoints neither read
nor return them. In MuJoCo, `Pi05PolicyConfig(mobile_base_execution=True,
mobile_base_contract="mobile_base_slots_v2")` makes `RemoteChunkPolicy` buffer `[arm 10 | base 3]` chunks and, per
simulator step:

- v2: send the measured body-frame velocity since the previous step (`mobile_base.body_velocity`, previous-frame
  convention, zero at the first step), and execute a base velocity as the world `[x, y, theta]` target
  `anchor + R(yaw_t) v dt` of the `base` move group (`mobile_base.velocity_base_target`, dt = `policy_dt_ms`).
  `mobile_base_velocity_anchor="reference"` (default) integrates from the previous executed target: the exact inverse
  of the label `(target_t - target_(t-1)) / dt`. `"measured"` integrates from the measured pose (target = measured
  pose + R v dt): no wind-up when the base is blocked, but in the recorded episode the servo trails the reference by
  up to 9 mm per step at 0.4 m/s, so it commands a slower base than the data.
- v1: send the base pose relative to its pose at the episode's first query and execute a base action as the
  offset from the current measured pose (`mobile_base.compose_base_target`).

`mobile_base_contract` defaults to v2 (no checkpoint was trained on v1); a v2 client also sends `base_dt_s` =
`policy_dt_ms / 1000`. Off by default (`mobile_base_execution=False`); fixed-arm evaluation is unchanged.

**RTC.** RTC is orthogonal to what the action dims mean: the prefix is the same model-space action vector, base
slots included. The client's RTC strategy already buffers the 13-wide chunk, and `_query_server` sends its arm part
as `action_prefix` (unchanged wire width) and its base part as `base_action_prefix`. The server normalizes the base
prefix with the `base_actions` table into slots 10-12 and guides them with the same per-timestep weights as the arm
(`Pi05Server._add_base_prefix`); a fixed-arm row's base slots stay zero and unguided, and a mobile row without
`base_action_prefix` is refused rather than left unguided. Before 2026-10-25 the server guided only the arm's valid
wire slots (`weights * model_action_mask` covered slots 0-9), so the base would have been free; the client refused
RTC for that reason, and that refusal is removed. Stationary checkpoints and fixed-arm rows get byte-identical
guidance (`tests/test_pi05_mobile_base_slots.py::test_rtc_guides_the_base_slots_of_a_mobile_checkpoint_and_nothing_else_changes`).
Dummy-model sample on the smoke batch (4 committed steps toward the batch's own targets, max guidance weight 5):
committed-step mean |sample - prefix| in the base slots 1.20 unguided, 1.23 with arm-only guidance, 0.27 with the
base slots guided; arm slots 0.20 to 0.24 with guidance.

**Limitations and TODOs.** No trained checkpoint exists yet; the first registered training cell is the xsrc40
exclusion-v2 cell (`november05_xsrc40_exclv2_qanchor_tact_lbox_gclose_b1024`, `molmo_spaces/training/nov05_xsrc40_exclv2_training_contract.py`). The eval client still refuses
`mobile_base_execution` with a query-anchored arm action (`Pi05PolicyConfig`), so no such checkpoint can be
evaluated closed-loop yet. Base
statistics are global (not per embodiment or group). Compact state serialization is
refused. The v2 measured velocity at an exact-state replay that starts mid-episode is zero at the first step (the
client has no previous pose), where the source episode had the base's real velocity; the v1 base-state origin there
is the restored pose. Pi0.5's state text discretizes normalized state into 256 bins over [-1, 1]; under z-score
normalization (`GroupedStandardNorm`) about a quarter of arm state values saturate (measured 2026-10-25). The base
state goes through the same tokenizer, so a z-scored base state would saturate the same way; quantile normalization
(the smoke config's) maps q01..q99 into the range.

### Manifest-driven behavior prompts

`PiExperimentConfig.behavior_prompt_overlays` can add a task-language suffix for every episode
without republishing RLDS. Each `BehaviorPromptOverlay` pins one component identity, the raw
manifest SHA-256, and the accepted global taxonomy-manifest SHA-256. The manifest contains an
exact episode-key mapping, structured global behavior factors, and the final semantic suffix.
The suffix must equal the deterministic rendering of its factor record; arbitrary text is
rejected.

Loader startup verifies the manifest hash, taxonomy identity, component identity, duplicate keys,
and exact missing/extra key coverage for the complete source split. It then broadcasts only the
overlaid prompt before tokenization. Images, actions, state, masks, terminals, episode order,
sampling, split, and normalization assets are unchanged. Normalization scans explicitly disable
the overlay because they consume only the original numeric state/action view. The scientific
record, `input_contract.json`, and dataset-mixture audit retain the manifest and taxonomy hashes;
the mixture audit also reports suffix occupancy.

W&B now logs one consolidated `data_visualization/training_samples` HTML panel at the configured
interval. It replaces the former `camera/*`, `input_visualization/*`, and `io_viz` panels. Each
refresh uses current training samples and shows train-mode augmented camera inputs, the actual
token IDs/pieces and decoded model prefix, normalized physical/embodiment/padding state partitions,
the selected normalization and AdaRMS routes, action masks, and target-versus-sampled chunks in
physical units. Predictions are sampled from the same augmented observation displayed in the panel.

The performance-oriented nine-seen-embodiment recipe is registered alongside it:

```bash
python -m molmo_spaces.training.main show \
  pi05_9emb_n60_idtext_state_fullft_h32_40k
python -m molmo_spaces.training.run train \
  --experiment pi05_9emb_n60_idtext_state_fullft_h32_40k --runtime pi05_gpu \
  --exp-name 20260714_pi05_9emb_n60_idtext_state_fullft_h32_40k
```

It full-fine-tunes on exact-terminal N60, checkpoints every 4k updates, and conditions Pi through
`MorphologyTextPrompt(include_id=True)` plus a fixed nine-way `EmbodimentOneHotState` appended after
physical-state normalization. The ID codebook is explicit because training IDs are non-contiguous.
Unknown IDs are errors rather than silent fallback. See
`notes/20260714_pi05_lowvar70_to_9emb_n60.md`.

The homogeneous-current-planner data control has the identical model/training contract:

```bash
python -m molmo_spaces.training.main show \
  pi05_9emb_n60_currentplanner_idtext_state_fullft_h32_40k
```

Only the N60 RLDS family and normalization asset ID change. This isolates mixed-planner mode
conflict from model capacity, conditioning, compute, and inference.

Two further controlled candidates use the same launcher and 4k checkpoint cadence:

```bash
# Homogeneous current-planner N30 data-scale control.
python -m molmo_spaces.training.run train \
  --experiment pi05_9emb_n30_currentplanner_idtext_state_fullft_h32_40k \
  --runtime pi05_gpu --exp-name <unique-name>

# Mixed N60 with explicit ID-selected state/action normalization tables.
python -m molmo_spaces.training.run train \
  --experiment pi05_9emb_n60_idtext_state_perembnorm_fullft_h32_40k \
  --runtime pi05_gpu --exp-name <unique-name>

# Homogeneous N30 with training observations balanced 50/50 before/after gripper close.
python -m molmo_spaces.training.run train \
  --experiment pi05_9emb_n30_currentplanner_phasebalanced_idtext_state_fullft_h32_40k \
  --runtime pi05_gpu --exp-name <unique-name>

# Combined structural candidate: clean N60 + per-robot normalization + 50/50 phase sampling.
python -m molmo_spaces.training.run train \
  --experiment pi05_9emb_n60_currentplanner_phasebalanced_perembnorm_idtext_state_fullft_h32_40k \
  --runtime pi05_gpu --exp-name \
  20260714_pi05_9emb_n60_currentplanner_phasebalanced_perembnorm_idtext_state_fullft_h32_40k

# Single-factor prompt ablation: short ID text + the same direct state identity.
python -m molmo_spaces.training.run train \
  --experiment pi05_9emb_n60_currentplanner_phasebalanced_perembnorm_id_state_fullft_h32_40k \
  --runtime pi05_gpu --exp-name \
  20260714_pi05_9emb_n60_currentplanner_phasebalanced_perembnorm_id_state_fullft_h32_40k

# Single-factor horizon ablation of the short-ID combined candidate.
python -m molmo_spaces.training.run train \
  --experiment pi05_9emb_n60_currentplanner_phasebalanced_perembnorm_id_state_fullft_h16_40k \
  --runtime pi05_gpu --exp-name \
  20260714_pi05_9emb_n60_currentplanner_phasebalanced_perembnorm_id_state_fullft_h16_40k

# Direct action-expert conditioning: learned stable-ID embedding in Pi0.5 AdaRMS.
python -m molmo_spaces.training.run train \
  --experiment pi05_9emb_n60_currentplanner_phasebalanced_perembnorm_adaemb_id_state_fullft_h16_40k \
  --runtime pi05_gpu --exp-name \
  20260714_pi05_9emb_n60_currentplanner_phasebalanced_perembnorm_adaemb_id_state_fullft_h16_40k
```

For the same registered GPU recipe on one large-memory GPU, set
`MOLMOSPACES_FSDP_DEVICES=1`. This changes runtime sharding; the `sbatch` request must independently
select a device with sufficient memory. Do not reduce the registered global batch size merely
because runtime sharding changed.

Per-embodiment normalization is opt-in and stores both global `state`/`actions` statistics and
flat ID-specific entries in the checkpoint asset. Training, RTC prefix normalization, and sampled
action unnormalization select the same table from `embodiment_id`; an unknown ID is an error.
Phase balancing is training-only and occurs before image decode; held-out validation and unchunked
normalization retain the natural frame distribution. The measured clean-N60 source is 19.6%
post-close, while the configured target is 50%.

The combined candidate is screened every 4k updates with the registered
`...phasebalanced_perembnorm..._{no_rtc_k4,rtc}_checkpoint_screen` cells. Its primary promotion
path uses RTC off with K=4, the best matched setting at mixed-N60 step 7,999; only a checkpoint that
passes the 27-state screen proceeds to the equal-robot
5-state x 2-repeat acceptance cell and wide-video evidence run.
The short-ID sibling changes only `MorphologyTextPrompt(include_id=True)` to
`EmbodimentIdPrompt()`: it preserves the nine-way state code, per-ID normalization, phase sampler,
data, model, and optimizer.
Its H=16 sibling changes only the trained prediction horizon and is evaluated coherently at K=4.
The `adaemb` sibling changes the H=16 model by adding a zero-initialized learned embedding of the
stable raw embodiment ID to Pi0.5's AdaRMS action-expert condition. The default vocabulary size is
zero, so upstream Pi/OpenPI recipes are unchanged. This candidate retains the compact prompt ID,
discrete-state one-hot, per-ID normalization, clean N60 data, and phase-balanced sampler.

## Audit the mixture and exposure

Every W&B-enabled training run logs `dataset/mixture` automatically at startup. The table groups
episodes and frames by dataset component, task, house (`scene_id`), embodiment, and their useful
intersections. It also repeats the run-level `batch size` and `steps to repeat` columns on every row
so exposure remains visible when the table is filtered. The default `dataset/repeat_budget` W&B
table contains the same two columns. `batch size` is the global batch size and `steps to repeat` is
the exact floating-point ratio `training_frames / global_batch_size`.

The row audit remains bounded for startup throughput, while the repeat denominator comes from the
immutable normalization manifest's exact training-frame count when available. Without a manifest,
the repeat columns are populated only when the bounded audit happened to cover the entire selected
training view; they are left null rather than estimated from a partial scan. Because MolmoSpaces
uses an infinite shuffled stream and may rebalance phases, this is a nominal sample-exposure repeat,
not a claim that the loader has a conventional finite epoch boundary.

### ID355 contiguous-action low-LR reference

The current single-embodiment UR5e reference uses pretrained Pi0.5, LoRA, H32, global B128,
seed 42, per-dataset full-support Gaussian normalization, normalized action clipping `[-5,5]`,
AdamW with zero weight decay, gradient clipping at 1, no EMA, and a 300-update warmup to constant
LR `1e-5`. Its model-facing action is
`[j1,j2,j3,j4,j5,j6,gripper,0,0,0]` with mask
`[T,T,T,T,T,T,T,F,F,F]`. Historical sparse UR5e sources must be transformed to that layout before
mixture sampling and before fitting normalization; do not pool sparse and contiguous tensors as if
they shared semantics.

This is a documented study reference, not a launcher default. Use the registered experiment config
for each dataset and preserve its normalization asset. Exact Camera-D and Mixed430 configs, W&B
lineages, checkpoints, and results are recorded in
`notes/20260820_id355_single_embodiment_study.md`.

The independent 2026-08-29 follow-up registers an exact Camera-D `H=50` ablation and an
equal-component Slowdwell200 + Unique230 + Camera-D400 aggregate. The H50 config changes only the
trained action horizon relative to its registered H32 sibling. The aggregate uses component
probabilities `(1/3,1/3,1/3)`, B384, LR `3e-5`, and a separately fitted equal-component Gaussian
asset; it must not reuse the natural-frame-weighted Mixed430 statistics. Exact names, promotion
rules, evaluation panels, and artifact ledger live in
`notes/20260829_id355_camd_h50_threeway_viewpoint_study.md`.

## Evaluate

```bash
python -m molmo_spaces.evaluation.run --config lowvar_x0_compare_noensemble_h32_k16
python -m molmo_spaces.evaluation.run --config lowvar_x0_compare_ensemble_h32_k16
python -m molmo_spaces.evaluation.run --config lowvar_pi05_openloop_h32_k16
python -m molmo_spaces.evaluation.run --config lowvar_pi05_rtc_h32_k16
```

The accepted lowvar recipe is frozen as
`20260714_lowvar_pi05_fullft_rtc_w10_acceptance`. On IRIS, launch it with the H100 contract:

```bash
sbatch --partition=iris-hi --gres=gpu:h100:1 \
  --export=ALL,CONFIG=20260714_lowvar_pi05_fullft_rtc_w10_acceptance \
  scripts/cluster/eval.sbatch
```

All four comparison cells use process-parallel exact training-state replay with two repeats.
The x0 no-ensemble and Pi open-loop cells are the common-executor comparison; x0 ensemble and
Pi RTC report their native continuity mechanisms. RTC uses H=32, K=16, a one-step simulated
inference delay, request-local deterministic noise, and the training-free pseudoinverse guidance
sampler. Override checkpoint fields with `--set`.

The shared client retains Octo's leading `<act:joint>` / `<act:xyz>` protocol token. The Pi server
removes that one known prefix before tokenization because MolmoSpaces RLDS trains Pi on the bare
authoritative task prompt (`open the drawer`). Embodiment conditioning, when enabled, is appended
after this canonicalization so training and inference strings remain identical.

The standardized runner supports either a local accelerator process or an already-running
remote server. For a TPU server launched separately on the VM:

```bash
python -m molmo_spaces.evaluation.run --config lowvar_pi05_rtc_h32_k16 --set \
  server_mode=remote server_host=<tpu-host> server_port=8770 \
  pi05_runtime_platform=tpu
```

Override the repository-local interpreters with `MOLMOSPACES_PI05_GPU_PY` or
`MOLMOSPACES_PI05_TPU_PY` only when explicitly testing another environment.

For publishable closed-loop comparisons, pin one concrete GPU class for both the Pi server and
MuJoCo EGL renderer. An untyped `--gres=gpu:1` request is insufficient: the same frozen lowvar
state produces different first-frame pixels on H100, H200, and A6000, and those small image/JAX
differences compound into different contact outcomes. The checkpoint watcher defaults to
`EVAL_GPU_TYPE=h100`; override it only for a labelled hardware ablation. Completed Pi runs record
the concrete JAX `device_kind`, hostname, and Slurm identity in `pi05_runtime` provenance.

For nine-embodiment checkpoint selection and acceptance:

```bash
python -m molmo_spaces.evaluation.run --config \
  20260714_pi05_9emb_n60_idtext_state_rtc_checkpoint_screen
python -m molmo_spaces.evaluation.run --config \
  20260714_pi05_9emb_n60_idtext_state_rtc_acceptance
python scripts/analysis/summarize_multiemb_eval.py \
  --cell candidate=/path/to/result.json --target 0.60
```

The screen is balanced across three exact N30 simulator states per training embodiment. The
acceptance cell uses five states x two stable seeds per embodiment (n=90) and gates on equal-robot
macro SR. `scripts/cluster/pi05_eval_local_checkpoint_when_ready.sbatch` can wait for an async
local Orbax checkpoint and submit the registered screen only after metadata finalization. It
requests an H100 by default; set `EVAL_GPU_TYPE` and optionally `EVAL_NODELIST` explicitly when a
different pinned renderer/inference device is intended.

Local server startup is phase-timed in `<out_dir>/_policy_server.log`. Large full-model
checkpoints can set `server_start_timeout_s` in their registered eval config; do not diagnose a
checkpoint as corrupt from a generic readiness timeout alone. Every completed result records a
cheap checkpoint metadata digest. If an experiment name is reused after `--overwrite`, set an
explicit `checkpoint_lineage`; it becomes part of the output path and prevents the new numerical
step from colliding with artifacts from the old lineage.

Pi evals also record `flow_noise_mode`. The default `independent_query` uses a fresh reproducible
flow latent at every replan. The diagnostic `fixed_rollout` mode reuses one latent within a rollout
while keeping cells/repeats independent, testing mode persistence without RTC prefix constraints
or temporal averaging.

Video-mode rollouts also save a compressed `*.actions.npz` beside each MP4. Receding-horizon
episodes report `chunk_boundary_jerk`, `within_chunk_jerk`, their ratio, and the exact transfer
steps in `result.json`; for RTC the transfer is aligned at `nK+d`, after its committed old action.
The non-recording SR rollouts remain fully process-parallel. Selected repeat-zero video rollouts use
at most two failure-aware worker processes because a larger set of concurrent MuJoCo/EGL recorders
can exit natively. Both passes detect a broken worker pool and retry unfinished cells in isolated
one-cell processes; a cell that exits twice is recorded as an explicit scored failure. This avoids
`multiprocessing.Pool` waiting forever for a lost work item without risking the parent eval process.

To freeze a comparison set, set `init_selection=explicit` and
`initial_state_manifest=/path/to/states.json`. Schema v1 references a source run and
`traj_N/obs/extra/sim_state[0]`; see `molmo_spaces.evaluation.initial_states`.

A socket-free real-checkpoint smoke (vanilla plus guided inference) is available as:

```bash
.venv/pi05-gpu/bin/python -m pi05_integration.smoke --checkpoint /path/to/checkpoint/step
```

## Tests

```bash
/iris/u/boai/miniconda3/envs/mlspaces/bin/python -m pytest -q \
  tests/test_pi05_integration.py tests/test_eval_chunk_boundaries.py tests/test_mobile_base_eval_client.py \
  tests/test_arm_action_eval_client.py

.venv/pi05-gpu/bin/python -m pytest -q \
  tests/test_pi05_input_config.py \
  tests/test_pi05_mobile_base_slots.py \
  tests/test_pi05_arm_action_forms.py \
  tests/test_nov05_xsrc40_exclv2_training.py \
  tests/test_pi05_training_registry.py \
  third_party/openpi/src/openpi/training/config_molmospaces_test.py \
  third_party/openpi/src/openpi/models/pi0_rtc_test.py
```

Checkpoint compatibility was hardware-verified on 2026-07-14 after the typed-input refactor. The
new registered config restored the pre-refactor full-FT checkpoint
`pi05_9emb_n60_currentplanner_idtext_state_fullft_h32_40k/.../7999` on an H200, loaded 6.2 GiB of
parameters and checkpoint normalization assets, and constructed the server with H=32,
`MorphologyTextPrompt`, and the expected global normalization route. Slurm job `16170629` completed
successfully. The dummy-model regression additionally proves numerical equivalence between the old
integer AdaRMS vocabulary field and `LearnedEmbodimentAdaRMS` under identical initialization/noise.
