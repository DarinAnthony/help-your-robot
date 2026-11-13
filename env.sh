# Source from any shell: `source ~/research/help!/env.sh`
# Local (macOS, Apple silicon) setup for the xemb-vla simulation stack used by help!.
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
export PY="$MAMBA_ROOT_PREFIX/envs/mlspaces/bin/python"
# Worktree of xemb-vla: origin/main plus the SRC kitchen twin commits from feat/src-kitchen-twin-axis-aligned
# (local branch help/kitchen-latest). The plain ~/research/xemb-vla checkout is on main without them.
export XEMB="$HOME/research/xemb-vla-kitchen"
export HOMEBODY_ROOT="$HOME/research/third_party/homebody"

# xemb-vla defaults these to the Stanford cluster (/iris/...); keep them in a local, non-iCloud folder.
export MLSPACES_ARTIFACT_ROOT="$HOME/mlspaces_artifact"
export MLSPACES_ASSETS_DIR="$MLSPACES_ARTIFACT_ROOT/assets"
export MLSPACES_CACHE_DIR="$MLSPACES_ARTIFACT_ROOT/resource_cache"
export SRC_KITCHEN_TWIN_DIR="$MLSPACES_ARTIFACT_ROOT/src_kitchen_twin"
export NLTK_DATA="$MLSPACES_ARTIFACT_ROOT/nltk_data"

export MUJOCO_GL=cgl
# pi05/ first: its pi05_integration copy shadows xemb's (pi05/PROVENANCE.md)
export PYTHONPATH="$HOME/research/help!/pi05:$XEMB:$HOME/research/help!${PYTHONPATH:+:$PYTHONPATH}"
