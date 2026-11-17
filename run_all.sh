#!/usr/bin/env bash
# Reproduces all results, tables and figures of the manuscript.
# Usage:  bash run_all.sh      (raw data must be placed in ./data, see README.md)
set -e
cd "$(dirname "$0")/work"
C=../code
F="CALMODE=gated REANCHOR=1 BUF=10 MORNING=0"   # final twin configuration (A3)

# 0) raw data -> intermediate files
python $C/prepare_data.py

# 1) Field plant: digital twin + calibrated features (final configuration)
python $C/lazz_twin.py
env $F OUTF=lazz_feat_A3.parquet python $C/lazz_features.py
python -c "import sys; sys.path.insert(0,'../code'); import pandas as pd; from common import add_raw_window; add_raw_window(pd.read_parquet('lazz_feat_A3.parquet')).to_parquet('lazz_feat2.parquet')"

# 2) Twins under +/-10 % irradiance-sensor gain error (robustness)
python - <<'P'
import sys; sys.path.insert(0,'../code')
import numpy as np, pandas as pd
from twin import twin_mpp
d0=pd.read_parquet('lazz_twin.parquet')
for tag,g in [('g090',0.9),('g110',1.1)]:
    d=d0.copy(); d['irr']=d['irr']*g; tw=twin_mpp(d.irr.values,d.pvt.values)
    for k,v in tw.items(): d['tw_'+k]=np.where(d.irr>50,v,np.nan)
    d.to_parquet(f'lazz_twin_{tag}.parquet')
P
for g in g090 g110; do
  env $F TWINF=lazz_twin_$g.parquet OUTF=lazz_feat_$g.parquet python $C/lazz_features.py
  CALMODE=none TWINF=lazz_twin_$g.parquet OUTF=lazz_feat_abl_A0_$g.parquet python $C/lazz_features.py
done

# 3) Ablation feature sets A0-A4
CALMODE=none                                 OUTF=lazz_feat_abl_A0.parquet python $C/lazz_features.py
CALMODE=ungated                              OUTF=lazz_feat_abl_A1.parquet python $C/lazz_features.py
CALMODE=gated REANCHOR=0                     OUTF=lazz_feat_abl_A2.parquet python $C/lazz_features.py
CALMODE=gated REANCHOR=1 BUF=10 MORNING=0    OUTF=lazz_feat_abl_A3.parquet python $C/lazz_features.py
CALMODE=gated REANCHOR=1 BUF=5  MORNING=1    OUTF=lazz_feat_abl_A4.parquet python $C/lazz_features.py

# 4) Field-plant experiments
python $C/lazz_classify.py        # Tables 5, 8 (LODO + leaky random split)
python $C/stats.py                # bootstrap CIs + Wilcoxon tests (Tables 5, 6)
python $C/lazz_rawrf.py           # RAW random forest row of Tables 5-6
python $C/lazz_dataeff.py         # Table 9
python $C/lazz_robust.py          # Table 10 (raw/twin under sensor error)
python $C/ablation_ms.py          # Table 10 (ablation A0-A4, 3 seeds)
python $C/ablation_A5.py          # Table 10 (final A5)
python $C/abl_robust_eval.py      # Table 10 (A0 vs A5 under sensor error)
PART=a python $C/gate_sensitivity.py &   # Table 11 (needs lazz_twin_g090/g110.parquet from step 2)
PART=b python $C/gate_sensitivity.py; wait
python $C/lazz_xai_open.py        # Table 12 (SHAP, PCS)
python $C/openset_v2.py           # Table 13 (open set)
python $C/latency.py              # Table 20

# 5) GPVS-Faults
python $C/gpvs_features.py
python $C/gpvs_cls.py             # blocked + cross-mode
python $C/gpvs_random.py          # leaky random split (5 seeds)
python $C/gpvs_detect_v2.py       # Table 14
python $C/gpvs_da.py              # Table 15
python $C/dl_baselines.py         # Table 7 (1D-CNN, LSTM)

# 6) TECNALIA: state identification, soiling, cleaning economics
python $C/tec_params.py           # Table 16
python $C/soiling.py              # Table 17 (S, S+Sh, S+D, S+D+Sh)
python $C/soiling_ext.py          # Table 17 (S+NU, S+ShI)
RUNS=500 python $C/cleaning_opt.py # Table 18

# 7) Figures and tables
mkdir -p figs
python $C/figs1.py; python $C/figs2.py; python $C/figs3.py; python $C/figs4.py
python $C/fig_framework.py; python $C/fig_graphical_abstract.py
python $C/make_tables.py          # -> TABLES.md
