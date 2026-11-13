"""Does the trained pi0.5 model REPRODUCE ground-truth actions on TRAINING data?

Distinguishes "model fits (flow sampling correct)" from "low flow-loss but bad samples".
Loads the trained ckpt, pulls normalized (obs, GT-action) batches through the exact
training data pipeline, runs sample_actions (10-step flow), compares in normalized space.
"""
import os, sys, numpy as np, jax, jax.numpy as jnp
from openpi.training import config as _config, data_loader as _dl
from openpi.models import model as _model

CFG = sys.argv[1] if len(sys.argv) > 1 else "pi05_v13_half919"
CKPT = sys.argv[2]
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 4

cfg = _config.get_config(CFG)
print(f"[fid] config={CFG} horizon={cfg.model.action_horizon} bs={cfg.batch_size} ckpt={CKPT}", flush=True)
model = cfg.model.load(_model.restore_params(os.path.join(CKPT, "params"), dtype=jnp.bfloat16))
loader = _dl.create_data_loader(cfg, shuffle=False, num_batches=NB, skip_norm_stats=False)

rng = jax.random.key(0)
all_pred, all_gt = [], []
for i, (obs, act) in enumerate(loader):
    r = jax.random.fold_in(rng, i)
    pred = model.sample_actions(r, obs, num_steps=10)
    all_pred.append(np.asarray(pred.astype(jnp.float32)))
    all_gt.append(np.asarray(act.astype(jnp.float32)) if hasattr(act, "astype") else np.asarray(act, np.float32))
    print(f"[fid] batch {i}: pred{all_pred[-1].shape} gt{all_gt[-1].shape}", flush=True)

pred = np.concatenate(all_pred, 0)   # (N, ah, ad) normalized
gt = np.concatenate(all_gt, 0)
N, AH, AD = pred.shape
mae = np.abs(pred - gt)
gt_std = gt.std(axis=(0, 1))                          # per-dim GT spread (normalized ~[-1,1])
print(f"\n[fid] N={N} horizon={AH} action_dim={AD}")
print(f"[fid] OVERALL MAE (normalized) = {mae.mean():.4f}   (GT std mean = {gt_std.mean():.4f})")
print(f"[fid] first-step (t=0) MAE     = {mae[:,0,:].mean():.4f}")
print(f"[fid] last-step  (t=-1) MAE    = {mae[:,-1,:].mean():.4f}")
print("[fid] per-dim  t=0:  MAE / GTstd")
for d in range(AD):
    print(f"   dim{d}: {mae[:,0,d].mean():.4f} / {gt_std[d]:.4f}")
# correlation per dim (flattened)
print("[fid] per-dim corr(pred,gt) over all (sample,step):")
for d in range(AD):
    a = pred[:,:,d].ravel(); b = gt[:,:,d].ravel()
    c = np.corrcoef(a, b)[0,1] if a.std()>1e-6 and b.std()>1e-6 else float('nan')
    print(f"   dim{d}: corr={c:.3f}  pred[mean={a.mean():.3f} std={a.std():.3f}] gt[mean={b.mean():.3f} std={b.std():.3f}]")
print("\n[fid] VERDICT:", "MODEL REPRODUCES GT (sampling OK)" if mae.mean() < 0.15 else "SAMPLES DIVERGE FROM GT (flow sampling / training broken)")
