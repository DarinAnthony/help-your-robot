"""xemb's ``scripts/datagen/run_pipeline.py`` with help!'s hooks installed (``helpsim/xemb_hooks.py``); same arguments.

    source env.sh && $PY scripts/run_pipeline_hooked.py --task_type open ...
"""

import os
import runpy
import sys

from helpsim import xemb_hooks

xemb_hooks.install()
script = os.path.join(os.environ["XEMB"], "scripts", "datagen", "run_pipeline.py")
sys.argv = [script, *sys.argv[1:]]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
