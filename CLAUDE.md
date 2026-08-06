# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Three academic ML deliverables (Especialización en Analítica de Datos,
Universidad Piloto de Colombia; course finished 2026-02-18) built around one
recurring business case: predicting churn for a fictional telecom
("Telmex Claro"), using a synthetic dataset generated at runtime (no data
files ship with the repo). Written with two classmates, then restructured
from three ~1,300–1,900 line flat scripts into a shared `src/churn/`
package. See `README.md` for the full story, badges, and methodology notes
— read that before this file if you need the "why," not just the "where."

## Running things

```bash
pip install -r requirements.txt
python scripts/run_unit1_baseline.py       # LR + RF (30/60-day churn) + K-Means
python scripts/run_unit2_supervised.py     # + Decision Tree, GridSearchCV, generalization analysis
python scripts/run_unit3_unsupervised.py   # DBSCAN/hierarchical, PCA/t-SNE, anomaly detection, imputation
```

No test suite, linter, or build step exists — this is coursework, not a
package meant to be installed. Each `scripts/run_unit*.py` does its own
`sys.path.insert(0, ".../src")` at the top; there's no `pip install -e .`
step. Unit 2 and 3 save figures to `outputs/unit2/` and `outputs/unit3/`
(created automatically via `churn.viz.output_dir()` — git-ignored, safe to
delete and regenerate).

To run a single piece of logic in isolation (e.g. while debugging one
function), import directly from `src/churn/`:
```python
import sys; sys.path.insert(0, "src")
from churn.data import generate_telecom_dataset
df = generate_telecom_dataset(n_samples=1000)
```

## Architecture

`src/churn/` is a shared library, not three copies of the same code:

- `config.py` — `RANDOM_STATE`, feature-name lists, `configure_environment()`, `set_plot_style()`.
- `data.py` — `generate_telecom_dataset()`, the canonical dataset generator, **used by Units 1 and 2 only** (see divergence note below).
- `preprocessing.py` — `preprocess_data()`: one-hot encode, stratified 80/20 split, scale (Units 1 & 2).
- `evaluation.py`, `clustering.py`, `supervised_baseline.py` — Unit 1's metrics helpers, K-Means segmentation, and Logistic Regression + Random Forest models.
- `supervised_deep.py`, `visualization_unit2.py` — Unit 2's 3-algorithm training (+ Decision Tree), `GridSearchCV`, generalization analysis, and its 4 saved plots.
- `missing_data.py`, `clustering_advanced.py`, `dimensionality.py`, `anomaly.py`, `association_rules.py` — Unit 3's imputation, DBSCAN/hierarchical clustering, PCA/t-SNE, anomaly detection, and optional (mlxtend-gated) association rules.
- `viz.py` — `output_dir(unit)`, the single place that resolves/creates `outputs/<unit>/`.

`scripts/run_unit{1,2,3}_*.py` are thin orchestrators: they wire the above
modules together in the original scripts' section order and reproduce their
(Spanish) console narration. Unit 3's orchestrator also contains its own
local `_generar_dataset_clientes()` — **not** a call into `churn.data`, see
below.

### Important: Unit 3's dataset generator is genuinely different, not a bug

Units 1 and 2 share `churn.data.generate_telecom_dataset()`. Unit 3's
original script generates its own dataset inline instead — different
distributions (uniform vs. exponential/normal/Poisson), a different risk
score formula, a single `abandono_30` target instead of `churn_30`/`churn_60`,
and Spanish column names (`antiguedad`, `cargo_mensual`, `tipo_contrato`, ...)
instead of English ones. This is a real inconsistency in the *original*
coursework — Unit 3's own docstring claims it reuses "the same dataset
generation logic" as Units 1/2, but the code doesn't. It's preserved as-is
(kept local to `scripts/run_unit3_unsupervised.py`) rather than retrofitted
onto the shared generator, since virtually every line in that ~1,900-line
pipeline references the Spanish column names — swapping the generator would
have meant rewriting the whole thing and risking behavior changes. If you
touch Unit 3, don't assume it produces the same columns as Units 1/2.

### Windows console encoding

Unit 1 and 2 print ASCII-only Spanish (e.g. "profundizacion", no accents) —
this was a deliberate compatibility fix, not a style choice: Windows'
default terminal codepage is `cp1252`, which raises `UnicodeEncodeError` on
accented characters or emoji. Unit 3 keeps the original's accented text and
emoji, and instead forces UTF-8 on `sys.stdout`/`sys.stderr` at the top of
`scripts/run_unit3_unsupervised.py`, before any module (including
`churn.association_rules`, which prints a warning at import time when
`mlxtend` is missing) has a chance to print. If you add new print statements
to Unit 1/2 code, stay ASCII-only to match; if you add them to Unit 3 code,
the UTF-8 reconfiguration already covers you.

## Conventions worth knowing before editing

- Console output is Spanish, narrative, and banner-heavy (`print("="*80)`,
  numbered section headers) by design — this is graded coursework meant to
  be read top-to-bottom, not something to "clean up" into terse logging.
- `mlxtend` (Unit 3 association rules only) is optional. `churn.association_rules`
  guards the import in `try/except ImportError` and the pipeline must keep
  running end-to-end without it — don't make it a hard dependency.
- `reports/*.pdf` are the written deliverables actually submitted for the
  course — don't regenerate or overwrite them. `reports/sample_console_output.txt`
  is a captured run of Unit 1, kept for reference.
- `notebooks/KMeans_Churn_Segmentacion.ipynb` is a narrated, notebook-native
  walkthrough of Unit 1's K-Means section — independent of the `scripts/`/`src/`
  split, not generated from it.
