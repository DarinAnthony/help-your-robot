# Provenance of pi05/

Copied on 2026-11-13 from `BoAi01/xemb-vla` (a private repository), local branch `help/kitchen-latest` at
`1d8cdefba8c0043d7f60a1b0fefe4c74cc87a495`: `origin/main` `c71a1199b` plus four SRC kitchen twin commits
cherry-picked from `feat/src-kitchen-twin-axis-aligned`. Apache-2.0 (Allen Institute for AI, `LICENSE`);
`third_party/openpi` keeps its own license.

| Here | Source path |
|---|---|
| `pi05_integration/` | `pi05_integration/` |
| `openpi_integration/` | `openpi_integration/` |
| `third_party/openpi/` | `third_party/openpi/` (vendored OpenPI, JAX 0.5.3) |
| `scripts/setup/pi05_env.sh` | `scripts/setup/pi05_env.sh` (`OPENPI` pinned to this folder's copy) |
| `scripts/cluster/pi05_*` | `scripts/cluster/pi05_*` |
| `scripts/data/mlspaces_to_rlds.py`, `README_RLDS.md` | `scripts/data/format_conversion/` |

Not copied: the launcher and experiment registry `molmo_spaces.training` (`python -m molmo_spaces.training.run ...`)
and the modules these files import (`molmo_spaces.embodiment`, `molmo_spaces.dynamic_urma`, `molmo_spaces.paths`).
They come from `$XEMB` on `PYTHONPATH`. `env.sh` puts `pi05/` first, so this copy of `pi05_integration` is the one
imported. Training needs a CUDA GPU or TPU (`scripts/setup/pi05_env.sh gpu|tpu`); it does not run on this Mac.
