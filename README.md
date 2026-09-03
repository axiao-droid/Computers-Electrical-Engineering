# Region-Adaptive Graph Attention for Citation-Network Link Prediction

Code and numerical results accompanying the manuscript submitted to *Computers & Electrical Engineering*.

## Main entry point

The paper results are produced by the **fair-protocol** pipeline:

```
pip install -r code/requirements.txt
python code/rerun_fair_protocol.py
```

This script runs every encoder (GCN, VGAE, GraphSAGE, GAT, GATv2, RA-GAT) under a shared,
pre-registered search space on the synthetic (syn) and public (Cora / CiteSeer) datasets, and
writes the numerical summaries to `figures/*.json`.

- Expected runtime is on the order of hours (extensive grid over multiple seeds).
- Outputs are written as JSON under `figures/` (e.g. `main_results.json`,
  `main_results_raw.json`, `real_citation_results.json`, `ablation_results.json`,
  `sensitivity_results.json`, `attention_results.json`, `embedding_results.json`,
  `degree_gain_results.json`, `training_curves_results.json`, `efficiency_results.json`).
- The public Cora / CiteSeer datasets are small and are staged locally by `code/data_gen.py`.

Scripts outside the main entry point (e.g. `run_all.py`, `run_real.py`, `retrain_*`,
`sweep_*`, `main.py`, `main_part2.py`) are part of the earlier exploratory workflow.
They are kept for reproducibility only and are **not** used for the reported paper results.

## Directory layout

```
.
├── README.md
├── LICENSE
├── requirements-lock.txt
├── code/              # all Python source (entry: rerun_fair_protocol.py)
├── figures/           # numerical summaries (*.json) referenced by the paper
└── user_data/         # local working data (paper source kept private; not part of repo)
```

## Data

- **Synthetic datasets**: generated locally by `code/data_gen.py` (no external download).
- **Public datasets**: Cora (`code/data/cora`), CiteSeer (`code/data/citeseer`).
  See the README inside each dataset folder for provenance and citation requirements.
