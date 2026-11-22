"""Pipe openpi training output through this to time it: prints each line, then seconds per update and a projection.

    uv run python scripts/openpi_run.py train ... 2>&1 | python scripts/step_timer.py --steps 10000 --units-per-hour 13

The first logged interval includes XLA compilation and data-loader warm-up, so the rate comes from the later ones.
Run training with PYTHONUNBUFFERED=1: piped, Python buffers its output and the lines arrive here all at once.
"""

import argparse
import json
import re
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=10_000, help="project a run of this many updates")
p.add_argument("--units-per-hour", type=float, default=None, help="Colab compute units per hour of this runtime")
p.add_argument("--json", default=None, help="also write the result here")
p.add_argument("--quiet", action="store_true", help="print only 'Step' lines, progress landmarks and errors")
a = p.parse_args()

marks = []
pat = re.compile(r"^Step (\d+):")
keep = re.compile(r"Error|Traceback|Exception|RESOURCE_EXHAUSTED|Killed|Resuming|Restoring|Downloading|"
                  r"Initialized data loader|Initialized train state|Waiting for checkpoint")
for line in sys.stdin:
    line = line.rsplit("\r", 1)[-1]   # tqdm's bar (stderr) shares these lines; keep what follows its last \r
    m = pat.match(line)
    if m:
        marks.append((int(m.group(1)), time.time()))
    if not a.quiet or m or keep.search(line):
        sys.stdout.write(line)
        sys.stdout.flush()

if len(marks) >= 3:
    (s0, t0), (s1, t1) = marks[1], marks[-1]
    sec = (t1 - t0) / (s1 - s0)
    if t1 - t0 < 0.5:
        print("TIMING the Step lines arrived together: run training with PYTHONUNBUFFERED=1", flush=True)
        sys.exit(1)
    out = {"sec_per_update": round(sec, 3), "updates_timed": s1 - s0,
           "hours_for_steps": round(sec * a.steps / 3600, 2), "steps": a.steps}
    if a.units_per_hour:
        out["units_for_steps"] = round(out["hours_for_steps"] * a.units_per_hour, 1)
    print("TIMING " + json.dumps(out), flush=True)
    if a.json:
        open(a.json, "w").write(json.dumps(out))
else:
    print(f"TIMING not enough logged steps ({len(marks)}) to time", flush=True)
