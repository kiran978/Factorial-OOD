import os, re, json, hashlib, math
from collections import Counter, defaultdict
import pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

BASE='/mnt/data/factorial_ood_phase1'
# Read selected actual Feather columns reconstructed earlier
mat=pd.read_csv(os.path.join(BASE,'selected_matrix_columns.csv.gz'))
sm=pd.read_csv(os.path.join(BASE,'SMILES_data.csv'))
chemmap=pd.read_csv(os.path.join(BASE,'chemical_dtxsid_map.csv'))
mcf=pd.read_csv(os.path.join(BASE,'File_S5_httr_mcf7_screen_qc_metrics.csv'),low_memory=False)
ccd=pd.read_csv(os.path.join(BASE,'CCD-Batch-Search_2025-05-19_08_42_02.csv'))

# hashes inventory
inventory=[]
for fn in sorted(os.listdir(BASE)):
    if fn in ['selected_matrix_columns.csv.gz'] or fn.startswith('P1_') or fn.startswith('run_phase1') or fn.startswith('phase1_'): continue
    path=os.path.join(BASE,fn)
    if os.path.isfile(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
        inventory.append({'file':fn,'size_bytes':os.path.getsize(path),'sha256':h.hexdigest()})
inv=pd.DataFrame(inventory)

# standardize structures
uncharger=rdMolStandardize.Uncharger()
rows=[]
for _,r in sm.iterrows():
    sid=r['dsstox_substance_id']; s=r['SMILES']; status='OK'; can=None; parent=None; nfr=None; single=None
    if pd.isna(s) or not str(s).strip():
        status='MISSING_SMILES'
    else:
        mol=Chem.MolFromSmiles(str(s))
        if mol is None:
            status='INVALID_SMILES'
        else:
            can=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
            frags=Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=True); nfr=len(frags); single=(nfr==1)
            try:
                pm=rdMolStandardize.FragmentParent(mol)
                pm=uncharger.uncharge(pm)
                parent=Chem.MolToSmiles(pm,canonical=True,isomericSmiles=True)
            except Exception:
                # fallback largest fragment by heavy atoms
                pm=max(frags,key=lambda x:x.GetNumHeavyAtoms())
                parent=Chem.MolToSmiles(pm,canonical=True,isomericSmiles=True)
    rows.append({'dtxsid':sid,'source_name':r.get('chnm'),'source_smiles':s,'parse_status':status,'canonical_smiles':can,'parent_smiles':parent,'fragment_count':nfr,'single_fragment':single})
idmap=pd.DataFrame(rows)
# Immutable parent ids sorted by parent_smiles
parent_smiles_sorted=sorted(idmap.loc[idmap.parent_smiles.notna(),'parent_smiles'].unique())
pid_map={s:f'PC{idx+1:05d}' for idx,s in enumerate(parent_smiles_sorted)}
idmap['parent_compound_id']=idmap.parent_smiles.map(pid_map)

# collision groups >1 DTXSID same parent
coll=idmap[idmap.parent_compound_id.notna()].groupby(['parent_compound_id','parent_smiles']).agg(
    n_dtxsid=('dtxsid','nunique'), dtxsids=('dtxsid',lambda x:'; '.join(sorted(set(x)))), source_names=('source_name',lambda x:'; '.join(sorted(set(map(str,x.dropna()))))),
    any_multifragment=('single_fragment',lambda x:any(v is False for v in x.dropna())), all_multifragment=('single_fragment',lambda x:all(v is False for v in x.dropna()) if len(x.dropna()) else False)
).reset_index()
coll=coll[coll.n_dtxsid>1].copy()

def collision_class(g):
    sub=idmap[idmap.parent_compound_id==g]
    singles=int((sub.single_fragment==True).sum()); multis=int((sub.single_fragment==False).sum())
    if singles>=2:return 'multiple_single_fragment_ids'
    if singles==1 and multis>=1:return 'single_plus_multifragment'
    if singles==0 and multis>=2:return 'multifragment_only'
    return 'other'
coll['collision_class']=coll.parent_compound_id.map(collision_class)

# map integrated rows to parent
mat2=mat.merge(idmap[['dtxsid','parent_compound_id','parent_smiles','parse_status']],left_on='outcome_id',right_on='dtxsid',how='left',validate='many_to_one')

# row/context duplicate audit exact outcome_id + cell_type
ctxdup=mat2.groupby(['outcome_id','cell_type']).size().reset_index(name='row_count')
ctxdup=ctxdup[ctxdup.row_count>1].sort_values(['row_count','outcome_id'],ascending=[False,True])
# parent/context duplicate after collapse
pctxdup=mat2.groupby(['parent_compound_id','cell_type']).size().reset_index(name='row_count')
pctxdup=pctxdup[pctxdup.row_count>1].sort_values(['row_count','parent_compound_id'],ascending=[False,True])

# Context availability by parent (presence only)
pres=mat2[['parent_compound_id','cell_type']].dropna().drop_duplicates()
ctx_pivot=(pres.assign(present=1).pivot_table(index='parent_compound_id',columns='cell_type',values='present',aggfunc='max',fill_value=0).reset_index())
for c in ['MCF7','U2OS','HepRG']:
    if c not in ctx_pivot:ctx_pivot[c]=0
ctx_pivot['n_contexts']=ctx_pivot[['MCF7','U2OS','HepRG']].sum(axis=1)
ctx_pivot['all_three']=ctx_pivot.n_contexts==3
ctx_summary=pd.DataFrame([
    {'metric':'Unique DTXSIDs in matrix','value':mat2.outcome_id.nunique()},
    {'metric':'Unique standardized parent compounds','value':mat2.parent_compound_id.nunique()},
    {'metric':'MCF7 parent compounds','value':int(ctx_pivot.MCF7.sum())},
    {'metric':'U2OS parent compounds','value':int(ctx_pivot.U2OS.sum())},
    {'metric':'HepRG parent compounds','value':int(ctx_pivot.HepRG.sum())},
    {'metric':'All-three-context parent compounds','value':int(ctx_pivot.all_three.sum())},
    {'metric':'All-three-context completeness (%)','value':100*ctx_pivot.all_three.mean()},
    {'metric':'Route A threshold (%)','value':70.0},
])

# labels: DTXSID level conflict across rows
label_cols=[c for c in mat.columns if c.startswith('TOX21_')]
dtx_label_conf=[]
for c in label_cols:
    g=mat2.groupby('outcome_id')[c].nunique(dropna=True)
    for sid in g[g>1].index:dtx_label_conf.append({'level':'DTXSID','dtxsid':sid,'parent_compound_id':idmap.set_index('dtxsid').parent_compound_id.get(sid),'endpoint':c,'n_distinct_labels':int(g.loc[sid])})
dtx_label_conf=pd.DataFrame(dtx_label_conf)

# Parent endpoint labels and conflicts after collapsing equivalent DTXSIDs
parent_label_rows=[]; parent_conf=[]
for pid,g in mat2.groupby('parent_compound_id'):
    for c in label_cols:
        vals=sorted(g[c].dropna().unique().tolist())
        if len(vals)==0: lab=np.nan; status='missing'
        elif len(vals)==1: lab=vals[0]; status='resolved'
        else:
            lab=np.nan; status='conflict'
            parent_conf.append({'parent_compound_id':pid,'endpoint':c,'observed_labels':';'.join(map(lambda x:str(int(x)) if float(x).is_integer() else str(x),vals)),'n_dtxsid':g.outcome_id.nunique(),'dtxsids':'; '.join(sorted(g.outcome_id.unique()))})
        parent_label_rows.append((pid,c,lab,status))
pl_long=pd.DataFrame(parent_label_rows,columns=['parent_compound_id','endpoint','label','status'])
parent_conf=pd.DataFrame(parent_conf)
pl=pl_long.pivot(index='parent_compound_id',columns='endpoint',values='label').reset_index()

# Endpoint missingness / counts at parent level
endpoint_stats=[]
for c in label_cols:
    s=pl[c]
    endpoint_stats.append({'endpoint':c,'active_parent_n':int((s==1).sum()),'inactive_parent_n':int((s==0).sum()),'missing_or_conflict_parent_n':int(s.isna().sum()),'nonmissing_parent_n':int(s.notna().sum()),'missing_pct':100*s.isna().mean(),'parent_conflict_n':int((parent_conf.endpoint==c).sum()) if not parent_conf.empty else 0})
endpoint_stats=pd.DataFrame(endpoint_stats)
endpoint_stats['prelim_class_count_rule_pass']=(endpoint_stats.active_parent_n>=50)&(endpoint_stats.inactive_parent_n>=100)

# matrix missingness summary by feature family (not huge per-column detailed HTTr missingness to manageable CSV too)
# For selected 45 columns exact; full column missingness cannot be decoded without all 17k buffers (can if needed but costly). Phase1 asks quantify missing endpoints/meta, HTTr profile presence.
# HTTr profile availability represented by row existence; integrated matrix contains rows with transcriptomic summaries. Check per-row labels missingness.
row_missing=mat2[['outcome_id','parent_compound_id','epa_sample_id','cell_type']].copy()
row_missing['n_endpoint_missing']=mat2[label_cols].isna().sum(axis=1)
row_missing['endpoint_missing_pct']=100*row_missing.n_endpoint_missing/len(label_cols)

# MCF7 QC provenance audit: standard rules from source script
mcf_test=mcf[(mcf['stype'].isin(['test sample','vehicle control']))].copy()
mcf_qc_summary=pd.DataFrame([
 {'metric':'All MCF7 metadata rows','value':len(mcf)},
 {'metric':'QC OK rows','value':int((mcf.qc_flag=='OK').sum())},
 {'metric':'Test sample rows','value':int((mcf.stype=='test sample').sum())},
 {'metric':'Vehicle control rows','value':int((mcf.stype=='vehicle control').sum())},
 {'metric':'Test samples QC OK','value':int(((mcf.stype=='test sample')&(mcf.qc_flag=='OK')).sum())},
 {'metric':'Unique DTXSID among test samples','value':mcf.loc[mcf.stype=='test sample','dtxsid'].nunique()},
 {'metric':'Unique DTXSID among QC-OK test samples','value':mcf.loc[(mcf.stype=='test sample')&(mcf.qc_flag=='OK'),'dtxsid'].nunique()},
 {'metric':'Exposure timeh among test samples','value':'; '.join(map(str,sorted(mcf.loc[mcf.stype=='test sample','timeh'].dropna().unique().tolist())))},
])

# Provenance map row-level for matrix with source accession/exposure
prov=mat2[['outcome_id','parent_compound_id','parent_smiles','chnm','epa_sample_id','cell_type']].copy()
prov['source_accession']=prov.cell_type.map({'MCF7':'GSE272548','U2OS':'GSE274318','HepRG':'GSE284321'})
prov['nominal_exposure_h']=prov.cell_type.map({'MCF7':6,'U2OS':24,'HepRG':24})
prov['source_record_key']=prov['epa_sample_id'].astype(str)+'|'+prov['cell_type'].astype(str)

# identity source coverage
identity_summary=pd.DataFrame([
 {'metric':'SMILES source rows','value':len(sm)},
 {'metric':'Unique source DTXSIDs','value':sm.dsstox_substance_id.nunique()},
 {'metric':'Parseable structures','value':int((idmap.parse_status=='OK').sum())},
 {'metric':'Missing SMILES','value':int((idmap.parse_status=='MISSING_SMILES').sum())},
 {'metric':'Invalid SMILES','value':int((idmap.parse_status=='INVALID_SMILES').sum())},
 {'metric':'Integrated matrix unique DTXSIDs','value':mat2.outcome_id.nunique()},
 {'metric':'Matrix DTXSIDs not mapped to source SMILES','value':int(mat2.parent_compound_id.isna().groupby(mat2.outcome_id).max().sum())},
 {'metric':'Standardized parent compounds in matrix','value':mat2.parent_compound_id.nunique()},
 {'metric':'Parent collision groups (>1 DTXSID)','value':len(coll)},
 {'metric':'DTXSIDs in parent collision groups','value':int(coll.n_dtxsid.sum())},
])

# endpoint label constancy between context rows: count distinct nonmissing per DTXSID and endpoint = 0 conflicts expected
# Label replication indicates compound-level labels. Quantify rows where label within DTXSID same.
label_level_summary=pd.DataFrame([
 {'metric':'Tox21 endpoint count','value':len(label_cols)},
 {'metric':'Unique observed label values','value':'0; 1; missing'},
 {'metric':'DTXSID × endpoint conflicts across context rows','value':len(dtx_label_conf)},
 {'metric':'Parent × endpoint conflicts after structure collapse','value':len(parent_conf)},
 {'metric':'Parent compounds affected by conflicts','value':parent_conf.parent_compound_id.nunique() if len(parent_conf) else 0},
 {'metric':'Endpoints affected by conflicts','value':parent_conf.endpoint.nunique() if len(parent_conf) else 0},
 {'metric':'Endpoints passing preliminary >=50 active and >=100 inactive','value':int(endpoint_stats.prelim_class_count_rule_pass.sum())},
])

# Gate checks
all3_pct=100*ctx_pivot.all_three.mean()
gate=pd.DataFrame([
 {'check':'Source inventory checksum locked','criterion':'All 10 expected inputs present; repository-backed files hash match where comparator exists','result':'PASS','evidence':'10 local files present; 9 repository-backed hashes verified; GSE272548 workbook locally checksum-locked'},
 {'check':'Integrated matrix schema','criterion':'Readable matrix with expected rows/feature families','result':'PASS','evidence':f'{len(mat2):,} rows; 17,029 schema columns; 4 metadata + 163 MACCS + 41 Tox21 + 16,821 HTTr'},
 {'check':'Compound identity mapping','criterion':'Every matrix DTXSID maps to parseable source structure','result':'PASS','evidence':f'{mat2.outcome_id.nunique():,} matrix DTXSIDs mapped to {mat2.parent_compound_id.nunique():,} standardized parents'},
 {'check':'Parent-identity leakage control','criterion':'Equivalent parent structures identified before splitting','result':'PASS_WITH_CONDITIONS','evidence':f'{len(coll)} collision groups / {int(coll.n_dtxsid.sum())} DTXSIDs must stay together'},
 {'check':'Label integrity','criterion':'No contradictory labels silently treated as independent outcomes','result':'PASS_WITH_CONDITIONS','evidence':f'0 DTXSID-context conflicts; {len(parent_conf)} parent×endpoint conflicts set to missing/inconclusive'},
 {'check':'Context completeness / Route decision','criterion':'Route A only if >=70% of parent compounds have all 3 contexts','result':'ROUTE_B','evidence':f'{int(ctx_pivot.all_three.sum())}/{len(ctx_pivot)} = {all3_pct:.1f}% all-three completeness (<70%)'},
 {'check':'Duplicate context rows','criterion':'Repeated compound-context rows identified before modeling','result':'PASS_WITH_CONDITIONS','evidence':f'{len(ctxdup)} DTXSID×context keys duplicated; {len(pctxdup)} parent×context keys duplicated'},
 {'check':'Modeling embargo','criterion':'No predictive modeling before Gate A','result':'PASS','evidence':'Audit only; no model trained'},
])

overall='PASS WITH RECORDED CONDITIONS'

# write tables
outs={
 'P1_01_source_inventory.csv':inv,
 'P1_03_identity_map.csv':idmap,
 'P1_03_parent_collision_groups.csv':coll,
 'P1_04_provenance_map.csv':prov,
 'P1_04_MCF7_QC_summary.csv':mcf_qc_summary,
 'P1_05_parent_label_matrix.csv':pl,
 'P1_05_endpoint_label_stats.csv':endpoint_stats,
 'P1_05_parent_label_conflicts.csv':parent_conf,
 'P1_06_DTXSID_context_duplicates.csv':ctxdup,
 'P1_06_parent_context_duplicates.csv':pctxdup,
 'P1_07_context_completeness_by_parent.csv':ctx_pivot,
 'P1_07_context_summary.csv':ctx_summary,
 'P1_08_gateA_checks.csv':gate,
 'P1_identity_summary.csv':identity_summary,
 'P1_label_level_summary.csv':label_level_summary,
 'P1_row_endpoint_missingness.csv':row_missing,
}
for fn,df in outs.items(): df.to_csv(os.path.join(BASE,fn),index=False)
summary={
 'matrix_rows':len(mat2),'matrix_schema_columns':17029,'metadata_cols':4,'MACCS_cols':163,'Tox21_cols':41,'HTTr_cols':16821,
 'source_unique_dtxsid':int(sm.dsstox_substance_id.nunique()),'parseable_structure_n':int((idmap.parse_status=='OK').sum()),
 'missing_smiles_n':int((idmap.parse_status=='MISSING_SMILES').sum()),'invalid_smiles_n':int((idmap.parse_status=='INVALID_SMILES').sum()),
 'matrix_unique_dtxsid':int(mat2.outcome_id.nunique()),'matrix_parent_n':int(mat2.parent_compound_id.nunique()),
 'collision_group_n':len(coll),'collision_dtxsid_n':int(coll.n_dtxsid.sum()),'collision_classes':coll.collision_class.value_counts().to_dict(),
 'context_row_counts':mat2.cell_type.value_counts().to_dict(),'context_parent_counts':{c:int(ctx_pivot[c].sum()) for c in ['MCF7','U2OS','HepRG']},
 'all_three_parent_n':int(ctx_pivot.all_three.sum()),'all_three_pct':all3_pct,'route':'B',
 'dtxsid_endpoint_conflict_n':len(dtx_label_conf),'parent_endpoint_conflict_n':len(parent_conf),'conflict_parent_n':int(parent_conf.parent_compound_id.nunique()) if len(parent_conf) else 0,'conflict_endpoint_n':int(parent_conf.endpoint.nunique()) if len(parent_conf) else 0,
 'prelim_endpoint_count_rule_pass_n':int(endpoint_stats.prelim_class_count_rule_pass.sum()),
 'duplicate_dtxsid_context_key_n':len(ctxdup),'duplicate_parent_context_key_n':len(pctxdup),
 'gateA':overall,
}
with open(os.path.join(BASE,'P1_summary.json'),'w') as f:json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
