# Standalone driver that regenerates the diagnostic / illustrative JSON files
# consumed by the paper's figure panels. The reported comparisons and tables
# come from rerun_fair_protocol.py; this script only rebuilds the optional side
# analyses (attention patterns, embeddings, degree-stratified gain, sensitivity
# surface, training curves, efficiency). It never touches the fair-protocol
# pipeline and does not run the deprecated run_all.py.
#
#   Usage:
#       python code/gen_diagnostics.py
#
#   Note: requires figures/main_results.json, which rerun_fair_protocol.py
#   produces (used for the efficiency table). Precomputed copies of every
#   output already ship in figures/, so this step is optional.
import os, sys, json

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_gen import load_dataset
from train import run_once
import main as M
import main_part2 as P2

ROOT = os.path.dirname(_HERE)
FIG = os.path.join(ROOT, 'figures')


def main():
    if not os.path.exists(os.path.join(FIG, 'dataset_stats.json')):
        M.dataset_stats()
        print('wrote dataset_stats.json', flush=True)
    p = os.path.join(FIG, 'main_results.json')
    if not os.path.exists(p):
        sys.exit(f'ERROR: {p} not found - run "python code/rerun_fair_protocol.py" first')
    with open(p, 'r', encoding='utf-8') as f:
        main_tab = json.load(f)
    P2.run_part2(M.save, load_dataset, run_once, M.DS, M.SEEDS, M.MODELS, main_tab)
    print('diagnostic JSON files (re)written under figures/', flush=True)


if __name__ == '__main__':
    main()
