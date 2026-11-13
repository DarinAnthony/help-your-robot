# MolmoSpaces openpi snapshot

This directory vendors Bo Ai's openpi fork at commit `07457a7` (`pi05-eval-tools`) and overlays the
working MolmoSpaces core/training changes that were present in `/iris/u/boai/openpi_gpu` on
2026-07-13. Runtime checkpoints, backup files, caches, and one-off diagnostics are not included.

Repository-local integration code belongs in `pi05_integration/`. Changes required inside openpi
(masked loss, canonical statistics, shared augmentation, and RTC sampling) are kept here so training
and serving use one versioned implementation rather than patching an external checkout at runtime.

This source is accelerator-neutral JAX, but upstream's lock selects `jax[cuda12]==0.5.3`.
`scripts/setup/pi05_env.sh` creates isolated repository-local GPU and TPU environments without
changing those pins: GPU uses the frozen lock directly and TPU replaces only the JAX hardware
plugin with `jax[tpu]==0.5.3`. External OpenPI and MonoPi copies are not runtime dependencies and
are never modified by repository entrypoints.
