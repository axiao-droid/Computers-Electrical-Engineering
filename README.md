# Region-Adaptive Graph Attention for Citation-Network Link Prediction

Code and numerical results accompanying the manuscript submitted to *Computers & Electrical Engineering*.

## Main entry point

The paper results are produced by the **fair-protocol** pipeline:

```
pip install -r code/requirements.txt
python code/rerun_fair_protocol.py
```

This script runs every encoder (GCN, VGAE, GraphSAGE, GAT, GATv2, RA-GAT) under a
controlled shared-setting hyperparameter search space on the synthetic (syn) and public
(Cora / CiteSeer) datasets, and writes the main numerical summaries to `figures/*.json`.

- Expected runtime is on the order of hours (extensive grid over multiple seeds).
- It produces the following files under `figures/`:
  `main_results.json`, `main_results_raw.json`, `ablation_results.json`,
  `public_channel_ablation.json`, `real_citation_results.json`,
  `neg_exclusivity_audit.json`.
- The public Cora / CiteSeer datasets are small and are staged locally by `code/data_gen.py`.

## Diagnostic / illustrative data (optional)

The figure-only diagnostics
(`attention_results`, `embedding_results`, `degree_gain_results`,
`sensitivity_results`, `training_curves_results`, `efficiency_results`, `dataset_stats`)
are not part of the fair-protocol comparisons or the reported tables. They are generated
by a separate driver, which uses the same `data_gen` / `train` pipeline and never runs the
deprecated `run_all.py`:

```
python code/gen_diagnostics.py
```

Running it writes the corresponding JSON files under `figures/`. Precomputed copies
already ship in `figures/`, so this step is optional.

Scripts outside the main entry point (e.g. `run_all.py`, `run_real.py`, `retrain_*`,
`sweep_*`, `main.py`, `main_part2.py`) are part of the earlier exploratory workflow.
They are kept for reproducibility only and are **not** used for the reported paper results.

## Directory layout

```
.
├── README.md
├── LICENSE
├── requirements-lock.txt
├── code/              # all Python source (entries: rerun_fair_protocol.py, gen_diagnostics.py)
└── figures/           # numerical summaries (*.json) referenced by the paper
```

## Data

- **Synthetic datasets**: generated locally by `code/data_gen.py` (no external download).
- **Public datasets**: Cora (`code/data/cora`), CiteSeer (`code/data/citeseer`).
  See the README inside each dataset folder for provenance and citation requirements.
