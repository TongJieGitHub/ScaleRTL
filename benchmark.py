from pathlib import Path
from perflib import run_task
import json
import os
import re
from tqdm import tqdm

root = Path('.')
sims = re.split(r'\s+', os.environ['SIMS'].strip())
runs_dir = root.joinpath('runs')
results_dir = root.joinpath('results/runs')

num_runs = 5
run_cycle = 10000

tasksets = {
  'scaleRTL-C': 'taskset -c 0-11',
  'scaleRTL-G': 'taskset -c 0-11',
}

# benchmarks = [f.stem for f in root.joinpath('cases').glob('*.fir')]
with open('case_list.txt') as f:
    benchmarks = [line.strip() for line in f if line.strip()]
for s in sims:
  assert s in tasksets, f"taskset command of simulator {s} is not set"

results_dir.mkdir(exist_ok=True, parents=True)
runs = [(bench, sim) for bench in benchmarks for sim in sims]
for bench, sim in tqdm(runs):
  run_task(
    bench, sim, 
    exe=runs_dir.joinpath(sim, 'bin', bench + '.out'),
    taskset=tasksets[sim], 
    results_dir=results_dir,
    num_runs=num_runs,
    run_cycle=run_cycle
  )

import pandas as pd
df = []
for res in results_dir.glob('*.json'):
  with res.open() as f:
    df += json.load(f)
pd.DataFrame(df).to_csv(root.joinpath('results', 'result.csv'), index=False)
