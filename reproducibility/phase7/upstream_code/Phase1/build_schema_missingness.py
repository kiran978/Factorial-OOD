import importlib.util, os, struct, numpy as np, pandas as pd, lz4.frame
# import module dynamically
import sys
sys.path.insert(0,'/mnt/data')
import read_feather_min as rf

fields=rf.fields; nrows=rf.nrows; null_counts=rf.null_counts; label_cols=[n for n,t in fields if n.startswith('TOX21_')]
meta={'chnm','epa_sample_id','cell_type','outcome_id'}
rows=[]
httr_indices=[]
for j,(name,t) in enumerate(fields):
    if name in meta: fam='metadata'
    elif name.startswith('MACCS_'): fam='MACCS'
    elif name.startswith('TOX21_'): fam='Tox21_endpoint'
    else: fam='HTTr'; httr_indices.append(j)
    metric=''
    base=name
    if fam=='HTTr':
        for suf,m in [('_dose_at_max','dose_at_max'),('_AUC_neg','AUC_neg'),('_max','max')]:
            if name.endswith(suf): metric=m; base=name[:-len(suf)]; break
    dtype='utf8' if t==5 else ('float64' if t==3 else f'arrow_type_{t}')
    nc=int(null_counts[j])
    rows.append({'column_index_0based':j,'column_name':name,'family':fam,'HTTr_base_signal':base if fam=='HTTr' else '','HTTr_summary_metric':metric,'arrow_type':dtype,'null_count':nc,'missing_pct':100*nc/nrows})
schema=pd.DataFrame(rows)
schema.to_csv('/mnt/data/factorial_ood_phase1/P1_02_data_dictionary.csv',index=False)

# Exact row-wise HTTr missingness by validity bitmap only
rowmiss=np.zeros(nrows,dtype=np.int32)
nonzero_cols=0
for j in httr_indices:
    if null_counts[j]==0: continue
    nonzero_cols+=1
    raw=rf.get_raw_buffer(rf.buf_idx[j])
    valid=rf.validity_mask(raw,nrows)
    rowmiss += (~valid).astype(np.int32)
row_df=pd.DataFrame({'matrix_row_0based':np.arange(nrows),'httr_missing_n':rowmiss,'httr_missing_pct':100*rowmiss/len(httr_indices)})
row_df.to_csv('/mnt/data/factorial_ood_phase1/P1_04_row_HTTr_missingness.csv',index=False)

fam_sum=schema.groupby('family').agg(columns=('column_name','size'),total_null_cells=('null_count','sum'),columns_with_any_missing=('null_count',lambda x:int((x>0).sum())),mean_column_missing_pct=('missing_pct','mean'),max_column_missing_pct=('missing_pct','max')).reset_index()
fam_sum.to_csv('/mnt/data/factorial_ood_phase1/P1_04_missingness_family_summary.csv',index=False)
httr_metric=schema[schema.family=='HTTr'].groupby('HTTr_summary_metric').agg(columns=('column_name','size'),columns_with_any_missing=('null_count',lambda x:int((x>0).sum())),total_null_cells=('null_count','sum'),mean_missing_pct=('missing_pct','mean'),max_missing_pct=('missing_pct','max')).reset_index()
httr_metric.to_csv('/mnt/data/factorial_ood_phase1/P1_04_HTTr_missingness_by_metric.csv',index=False)
print(fam_sum.to_string(index=False))
print('\nHTTr row missing summary',row_df.httr_missing_n.describe().to_dict(),'rows any missing',int((rowmiss>0).sum()),'nonzero cols',nonzero_cols)
print('\nHTTr metrics\n',httr_metric.to_string(index=False))
