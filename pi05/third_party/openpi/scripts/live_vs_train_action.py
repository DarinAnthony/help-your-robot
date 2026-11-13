"""At a datagen init state, does the model predict the SAME (correct) action from the
LIVE eval render as from the TRAINING image? Disambiguates residual-image/obs-gap from
closed-loop compounding drift.

Runs INSIDE mlspaces env is not possible (needs jax/openpi). So we DUMP live obs +
train image + GT action to an npz from the mlspaces side, then this (openpi side) loads
the trained policy and compares. This file is the openpi-side consumer.
Usage: python live_vs_train_action.py <config> <ckpt> <npz_glob>
"""
import sys, glob, numpy as np, jax
from openpi.training import config as _config
from openpi.policies import policy_config as _pc

CFG, CKPT, NPZG = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = _config.get_config(CFG)
policy = _pc.create_trained_policy(cfg, CKPT)
# Replicate the EVAL SERVER's output transform: model_transforms.outputs (NOT LiberoOutputs
# [:7]) + Unnormalize -> keep the FULL action incl gripper at index 7.
import openpi.transforms as _tf
from openpi.training import checkpoints as _ckpts
from openpi.shared import download as _dl
_ck = _dl.maybe_download(str(CKPT))
_dc = cfg.data.create(cfg.assets_dirs, cfg.model)
_ns = _ckpts.load_norm_stats(__import__("pathlib").Path(_ck) / "assets", _dc.asset_id)
policy._output_transform = _tf.compose([*_dc.model_transforms.outputs,
                                        _tf.Unnormalize(_ns, use_quantiles=_dc.use_quantile_norm)])
print(f"[lvt] policy ready (SERVER output transform) config={CFG}", flush=True)

def infer(img, wrist, state, prompt):
    out = policy.infer({
        "observation/image": np.asarray(img, np.uint8),
        "observation/wrist_image": np.asarray(wrist, np.uint8),
        "observation/state": np.asarray(state, np.float32),
        "prompt": str(prompt).lower(),
    })
    return np.asarray(out["actions"], np.float32)  # (horizon, D) raw/unnormalized

for f in sorted(glob.glob(NPZG)):
    d = np.load(f, allow_pickle=True)
    gt = d["gt_action"]                 # (D,) raw GT first action
    st = d["state"]; pr = str(d["prompt"])
    a_live = infer(d["img_live"], d["wrist_live"], st, pr)
    a_train = infer(d["img_train"], d["wrist_train"], st, pr)
    D = a_live.shape[-1]
    k = min(D, 8)
    e_live = np.abs(a_live[0,:k] - gt[:k]).mean()
    e_train = np.abs(a_train[0,:k] - gt[:k]).mean()
    print(f"[lvt] {f.split('/')[-1]}: outdim={D}  MAE(live,GT)={e_live:.4f}  MAE(train,GT)={e_train:.4f}  "
          f"GRIPPER live={a_live[0,7] if D>7 else 'NA'}  train={a_train[0,7] if D>7 else 'NA'}  GTgrip={gt[7]:.3f}", flush=True)
    print(f"        live[:8] ={np.round(a_live[0,:8],3)}")
    print(f"        gt[:8]   ={np.round(gt[:8],3)}")
