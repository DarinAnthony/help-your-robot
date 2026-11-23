"""xemb's ``run_pipeline.py`` with help!'s hooks (``helpsim/xemb_hooks.py``) and pi0.5 as every episode's policy
(``helpsim/pi05_drawer.py``, source ``$HELP_PI05_SOURCE``); same arguments. Run through ``scripts/run_drawer_pi05.sh``."""

import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from helpsim import pi05_drawer, xemb_hooks  # noqa: E402

xemb_hooks.install()
pi05_drawer.install()
script = os.path.join(os.environ["XEMB"], "scripts", "datagen", "run_pipeline.py")
sys.argv = [script, *sys.argv[1:]]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
