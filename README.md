# Self-Calibrating Physics-Guided Digital Twin for PV Fault Identification and Soiling Quantification — code

Code accompanying the manuscript *"A Self-Calibrating Physics-Guided Digital Twin with Explainable Machine Learning
for Robust Fault Identification and Soiling Loss Quantification in Photovoltaic Systems"* (submitted to *Energy and AI*).

Corresponding author: Khaled Mohamad Abdel Aziz — Khaled.ai_0047@ai.kfs.edu.eg

## 1. Folder layout

```
code/        all scripts (run from the work/ folder)
splits/      exact data splits used in the paper (see Section 6)
data/        put the three public datasets here (not redistributed; layout in data/README.md)
work/        intermediate files, results (*.json, *.csv), figs/ and TABLES.md are written here
run_all.sh   end-to-end pipeline in the order used for the paper
requirements.txt
LICENSE      MIT licence (code only; the datasets keep their own licences)
```

## 2. Environment

Python 3.11–3.13, single CPU core is sufficient. `pip install -r requirements.txt`
(versions listed are the ones used for the paper; nearby versions should work).

## 3. Data (public, download from the original providers)

| Dataset | Source | Place in |
|---|---|---|
| Field plant, Lazzaretti et al. (Sensors 2020) — Apache-2.0 | https://github.com/clayton-h-costa/pv_fault_dataset | `data/pv_fault_dataset/dataset_elec.mat`, `dataset_amb.mat` |
| GPVS-Faults (Bakdi et al.) | https://doi.org/10.17632/n76t439f65.1 (use the CSV files) | `data/GPVS/CSV_Files/F0L.csv … F7M.csv` |
| TECNALIA / Univ. of Ljubljana PV performance dataset — CC BY 4.0 | https://doi.org/10.5281/zenodo.18306410 | `data/TECNALIA/data_<Module>/data_<Module>.csv` |

`code/lazz_day_starts.csv` stores the sample index at which each recorded day of the field-plant dataset starts
(the day segmentation used for leave-one-day-out validation).

## 4. Run

```
bash run_all.sh
```
Runtime on one CPU core is several hours, dominated by leave-one-day-out training (MLP, random forest) and the
deep-learning baselines.

## 5. Script → manuscript map

| Script | Produces | Manuscript |
|---|---|---|
| `twin.py` | CEC single-diode twin (pvlib) | Sec. 4.1 |
| `lazz_twin.py`, `lazz_features.py`, `common.py` | twin predictions, gated online calibration, twin features, physics-consistent augmentation | Sec. 4.2–4.3, Tables 3–4, Figs. 2–3 |
| `lazz_classify.py`, `stats.py` | leave-one-day-out and random-split classification, bootstrap CIs, Wilcoxon tests | Tables 5, 6, 8; Figs. 4–5 |
| `lazz_rawrf.py` | RAW random-forest row of Table 5 (predictions saved, day-bootstrap CI, Wilcoxon vs PHYS+PA RF) | Tables 5–6 |
| `lazz_dataeff.py` | data efficiency | Table 9, Fig. 6 |
| `ablation_ms.py`, `ablation_A5.py`, `abl_robust_eval.py`, `lazz_robust.py` | ablation A0–A5 and sensor-gain robustness | Table 10, Fig. 7 |
| `gate_sensitivity.py` | one-at-a-time sensitivity of ε_I, ε_V, Δ_I, half-life and re-anchoring buffer (`PART=a`/`b` splits the grid over two processes) | Table 11 |
| `lazz_xai_open.py`, `openset_v2.py` | SHAP, physics-consistency score, open-set AUROC | Tables 12–13, Figs. 8–9 |
| `gpvs_features.py`, `gpvs_cls.py`, `gpvs_random.py`, `gpvs_da.py`, `gpvs_detect_v2.py` | converter-level twin, protocols, transfer, unsupervised detection | Tables 8, 14, 15; Figs. 10–11 |
| `dl_baselines.py` | 1D-CNN and LSTM baselines | Table 7 |
| `tec_params.py` | online single-diode state identification of aged modules | Table 16, Fig. 12 |
| `soiling.py` | semi-synthetic soiling benchmark (twin, naive PR, RdTools SRR), scenarios S, S+Sh, S+D, S+D+Sh | Table 17, Fig. 13 |
| `soiling_ext.py` | scenarios that violate the I_sc assumption: non-uniform soiling (S+NU) and I_sc-reducing shading (S+ShI) | Table 17, Fig. 13 |
| `cleaning_opt.py` | renewal-optimal cleaning rule vs fixed schedules and oracle (500 runs, common random numbers; `RUNS` env) | Table 18, Fig. 14 |
| `latency.py` | computational cost | Table 20 |
| `figs1.py`–`figs4.py`, `fig_framework.py`, `fig_graphical_abstract.py`, `make_tables.py` | figures and `TABLES.md` | all figures/tables |

`TABLES.md` uses the table numbers of the manuscript.

## 6. Notes

* All steps are causal: the calibration state used at minute *m* only depends on data before *m*.
* Final twin configuration: `CALMODE=gated REANCHOR=1 BUF=10 MORNING=0` (A3) with physics-consistent augmentation (A5).
* Calibration settings can be changed through environment variables of `lazz_features.py`: `EPS_I` (0.015), `EPS_V` (0.02),
  `DELTA_I` (0.15), `HALF` (240 accepted minutes) and `BUF` (10 min); defaults reproduce the paper.
* Random seeds are fixed in every script. Results for the paper were computed in several batches with these scripts;
  small differences (third decimal) can arise from library versions and thread scheduling.

## 7. Data splits (`splits/`)

* `lazz_lodo_folds.csv` — leave-one-day-out folds of the field plant: fold *k* holds out recorded day *k*
  (day boundaries in `code/lazz_day_starts.csv`), with the number of samples of each class per day.
  Training folds subsample normal and shading samples to 20 % with the seed given in each script; test folds are complete.
* `gpvs_window_splits.csv` — one row per 0.1 s GPVS-Faults window (run, tracking mode, window end time, label, onset)
  with its role in every protocol: blocked temporal split (`train`/`gap`/`test`), both cross-mode directions, and the five
  leaky stratified 70/30 random splits (seeds 0–4).
* Monte-Carlo seeds: soiling benchmark seeds 0–19 per scenario; cleaning simulation seeds 0–499 (plant) and 100000 + seed
  (observation noise).

## 8. Licence and citation

Code: MIT licence (see `LICENSE`). Please cite the article and the original datasets when using this code.
Citation metadata for this software is given in `CITATION.cff`. Archived release: https://doi.org/10.5281/zenodo.XXXXXXX (to be updated after the first release).
