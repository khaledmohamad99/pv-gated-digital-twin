"""Convert the public raw datasets into the intermediate files used by the pipeline.
Run from the work/ folder:  python ../code/prepare_data.py
Expected raw data layout (download links in README.md):
  ../data/pv_fault_dataset/dataset_elec.mat, dataset_amb.mat   (Lazzaretti et al., GitHub)
  ../data/GPVS/CSV_Files/F0L.csv ... F7M.csv                    (GPVS-Faults, Mendeley Data)
  ../data/TECNALIA/data_<Module>/data_<Module>.csv              (TECNALIA, Zenodo; read directly by tec_params.py)
"""
import numpy as np, pandas as pd, scipy.io as sio, os
# ---- Lazzaretti field plant -> lazz_raw.parquet (1 Hz)
L='../data/pv_fault_dataset/'
e=sio.loadmat(L+'dataset_elec.mat'); a=sio.loadmat(L+'dataset_amb.mat')
d=pd.DataFrame({k:e[k].ravel() for k in ['vdc1','vdc2','idc1','idc2']})
d['irr']=a['irr'].ravel(); d['pvt']=a['pvt'].ravel(); d['y']=a['f_nv'].ravel().astype(int)
d['t']=np.arange(len(d))
# day index: sample indices at which each recorded day starts (segmentation used in the paper)
starts=pd.read_csv('../code/lazz_day_starts.csv').start.values
d['day']=np.searchsorted(starts,d.t.values,side='right')-1
d['ep']=1+np.r_[0,np.cumsum(np.diff(d.y.values)!=0)]          # label-run (episode) index
d.to_parquet('lazz_raw.parquet'); print('lazz_raw',d.shape)
# ---- GPVS-Faults -> gpvs_<run>.pkl (10 kHz)
G='../data/GPVS/CSV_Files/'
for f in sorted(os.listdir(G)):
    if f.endswith('.csv'):
        pd.read_csv(G+f).to_pickle('gpvs_'+f.replace('.csv','.pkl')); print(f)
