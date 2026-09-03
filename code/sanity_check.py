# Deprecated: not used for reported paper results. See rerun_fair_protocol.py
# __mh_autobootstrap_syspath__
import os as _mh_os, sys as _mh_sys
_mh_here = _mh_os.path.dirname(_mh_os.path.abspath(__file__))
if _mh_here and _mh_here not in _mh_sys.path:
    _mh_sys.path.insert(0, _mh_here)

# Automated "too-perfect" / invalid-value detector over figures/*.json.
import json, os, sys, math

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
results = {}
for f in sorted(os.listdir(FIG)):
    if f.endswith('_results.json') or f in ('all_results.json', 'main_results.json',
                                            'ablation_results.json'):
        try:
            with open(os.path.join(FIG, f), 'r', encoding='utf-8') as fh:
                results[f] = json.load(fh)
        except Exception:
            pass

errors, suspicious = [], []

def check(name, val):
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        return
    k = name.lower()
    if any(w in k for w in ['auc', 'ap_', '_ap', 'accuracy', 'f1', 'precision', 'recall', 'hits']):
        if val > 0.999:
            suspicious.append(f"{name} = {val:.4f} (>0.999) — check overfitting/leakage")
        elif val < 0:
            errors.append(f"{name} = {val} negative metric")
    if any(w in k for w in ['rmse', 'mae', 'mse', 'loss']):
        if val < 0:
            errors.append(f"{name} = {val} negative error")
    if 'p_value' in k or 'pvalue' in k:
        if val > 1:
            errors.append(f"{name} = {val} p>1 impossible")

PAIRED_LEAF_KEYS = {'mean', 'lo', 'hi', 'n'}

def _is_paired_leaf(o):
    """Paired-difference record like {'mean':..., 'lo':..., 'hi':..., 'n':...}
    (e.g. 'RA-GAT - GAT / AUC' under real_citation_results['paired']). Such
    differences may legitimately be negative, so sign/range checks must not
    apply to their mean/lo/hi fields - only absolute metrics are sign-checked."""
    return isinstance(o, dict) and o.keys() <= PAIRED_LEAF_KEYS and 'mean' in o

def walk(o, p=""):
    if isinstance(o, dict):
        paired_leaf = _is_paired_leaf(o)
        for k, v in o.items():
            if paired_leaf:
                # Skip sign/range checks for paired-difference mean/lo/hi.
                if k != 'n' and isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    errors.append(f"{p}.{k} is NaN/Inf")
                continue
            walk(v, f"{p}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            if isinstance(v, (dict, list)): walk(v, f"{p}[{i}]")
    elif isinstance(o, float):
        if math.isnan(o): errors.append(f"{p} is NaN")
        elif math.isinf(o): errors.append(f"{p} is Inf")
        else: check(p, o)
    elif isinstance(o, int):
        check(p, o)

for fn, data in results.items():
    walk(data, fn)

for e in errors: print("ERROR:", e)
for s in suspicious: print("SUSPECT:", s)
if errors:
    print(f"\n{len(errors)} hard errors"); sys.exit(1)
if suspicious:
    print(f"\n{len(suspicious)} suspicious values — confirm manually")
else:
    print("All values pass sanity check")
