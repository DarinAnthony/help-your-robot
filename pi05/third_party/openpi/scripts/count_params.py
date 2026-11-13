"""Count total vs trainable params for a pi0.5 config+checkpoint, replicating
get_freeze_filter's regex logic on the restored param tree.
Usage: python count_params.py <config> <ckpt_dir>
"""
import sys, numpy as np, jax
from openpi.training import config as _config
from openpi.models import model as _model

CFG, CKPT = sys.argv[1], sys.argv[2]
cfg = _config.get_config(CFG)
pv = cfg.model.paligemma_variant
av = cfg.model.action_expert_variant
lora_pg = "lora" in pv
lora_ae = "lora" in av
print(f"config={CFG}  paligemma={pv}  action_expert={av}", flush=True)

params = _model.restore_params(CKPT + "/params", dtype=None)
leaves = jax.tree_util.tree_flatten_with_path(params)[0]

def pstr(path):
    out = []
    for k in path:
        out.append(getattr(k, "key", None) or getattr(k, "name", None) or str(k))
    return "/".join(str(x) for x in out)

def frozen(s):
    is_llm = "llm" in s
    is_lora = "lora" in s.lower()
    if lora_pg or lora_ae:
        # our configs set BOTH lora -> freeze .*llm.* AND NOT .*lora.*
        return is_llm and (not is_lora)
    return False  # full FT: nothing frozen

tot = 0; tr = 0
buckets = {}
for path, v in leaves:
    s = pstr(path); n = int(np.prod(v.shape))
    tot += n
    fr = frozen(s)
    if not fr: tr += n
    # bucket by top module
    top = s.split("/")[0]
    key = ("img" if "img" in s else ("llm_1" if "llm/" in s and "_1" in s else ("llm" if "llm" in s else top)))
    lk = "lora" if "lora" in s.lower() else "base"
    buckets.setdefault((key, lk), 0)
    buckets[(key, lk)] += n

print(f"TOTAL params    : {tot/1e6:8.2f} M")
print(f"TRAINABLE params: {tr/1e6:8.2f} M   ({100*tr/tot:.1f}% of total)")
print(f"FROZEN params   : {(tot-tr)/1e6:8.2f} M")
print("--- breakdown (module, lora/base -> M params) ---")
for (k, lk), n in sorted(buckets.items(), key=lambda x: -x[1]):
    print(f"   {k:10} {lk:5}: {n/1e6:8.2f} M")
