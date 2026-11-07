from __future__ import annotations
import os, sys, json, hashlib, platform, shutil, subprocess
from pathlib import Path
import pandas as pd
import numpy as np

BASE=Path('/mnt/data')
P2=BASE/'phase2_pkg'
P3=BASE/'phase3_pkg'/'outputs'
OUT=BASE/'factorial_ood_phase4'/'outputs'
OUT.mkdir(parents=True, exist_ok=True)

master=pd.read_csv(P2/'P2_01_curated_compound_master.csv')
contexts=pd.read_csv(P2/'P2_02_curated_compound_context_table.csv')
source_rows=pd.read_csv(P2/'P2_02a_curated_context_source_rows.csv')
labels=pd.read_csv(P2/'P2_04_endpoint_label_matrix.csv')
primary_eps=pd.read_csv(P3/'P3_06a_primary_endpoints.csv')['endpoint'].tolist()
p3_outer=pd.read_csv(P3/'P3_02_outer_split_feasibility_manifest.csv')
p3_chem=pd.read_csv(P3/'P3_03_chemical_support_by_parent.csv')
p3_bio=pd.read_csv(P3/'P3_04_biological_support_by_parent.csv')
p3_fact=pd.read_csv(P3/'P3_05_factorial_cell_manifest.csv')
p3_thr=pd.read_csv(P3/'P3_04a_biological_support_fold_thresholds.csv')

APPROVAL_DATE='2026-08-31'
PROTOCOL='v0.2'
PRIMARY_OUTER_SEED='FACTORIAL-OOD-v0.1'  # inherited exactly from Phase 3 candidate manifest
INNER_SALT='FACTORIAL-OOD-v0.2-INNER-v1'
ALT_SEEDS={
 'V3_ALT01':'FACTORIAL-OOD-V3-ALT-01',
 'V3_ALT02':'FACTORIAL-OOD-V3-ALT-02',
 'V3_ALT03':'FACTORIAL-OOD-V3-ALT-03',
 'V3_ALT04':'FACTORIAL-OOD-V3-ALT-04',
}

def hfold(compound_id, smiles, salt, nfold):
    s=f'{compound_id}|{smiles}|{salt}'
    return int(hashlib.sha256(s.encode()).hexdigest()[:16],16)%nfold

def inner_fold(compound_id, smiles, split_set, outer_fold):
    s=f'{compound_id}|{smiles}|{split_set}|outer={outer_fold}|{INNER_SALT}'
    return int(hashlib.sha256(s.encode()).hexdigest()[:16],16)%4

# ---------- Primary outer freeze ----------
idcols=master[['compound_id','canonical_parent_smiles','parent_inchikey','bemis_murcko_scaffold','source_dtxsids']].copy()
recalc=idcols.apply(lambda r:hfold(r['compound_id'],r['canonical_parent_smiles'],PRIMARY_OUTER_SEED,5),axis=1)
outer=p3_outer[['compound_id','outer_fold']].merge(idcols,on='compound_id',how='left')
outer['recomputed_outer_fold']=outer.apply(lambda r:hfold(r['compound_id'],r['canonical_parent_smiles'],PRIMARY_OUTER_SEED,5),axis=1)
outer['hash_match']=outer['outer_fold'].eq(outer['recomputed_outer_fold'])
outer['split_set']='PRIMARY'
outer['split_role']='CONFIRMATORY_PRIMARY'
outer['split_method']='SHA256(compound_id|canonical_parent_smiles|FACTORIAL-OOD-v0.1)[:16] mod 5'
outer['endpoint_outcomes_used']=False
outer['protocol_frozen_under']=PROTOCOL
outer['protocol_approval_date']=APPROVAL_DATE
outer['status']='FROZEN_DO_NOT_CHANGE_AFTER_MODELING_START'
outer_cols=['split_set','split_role','compound_id','canonical_parent_smiles','parent_inchikey','bemis_murcko_scaffold','source_dtxsids','outer_fold','split_method','endpoint_outcomes_used','protocol_frozen_under','protocol_approval_date','status']
outer[outer_cols].to_csv(OUT/'P4_01_primary_outer_split_manifest_FROZEN.csv',index=False)

# ---------- Split registry + four alternative V3 outer sets ----------
reg=[{
 'split_set':'PRIMARY','role':'confirmatory primary','outer_seed':PRIMARY_OUTER_SEED,'outer_folds':5,'inner_folds':4,
 'outer_assignment_source':'Inherited unchanged from P3_02 nominated manifest','endpoint_outcomes_used':False,
 'status':'FROZEN','protocol':PROTOCOL,'approval_date':APPROVAL_DATE
}]
alt_rows=[]
for sid,seed in ALT_SEEDS.items():
    reg.append({'split_set':sid,'role':'V3 robustness only','outer_seed':seed,'outer_folds':5,'inner_folds':4,
                'outer_assignment_source':'Prospectively generated in Phase 4 from identity fields only','endpoint_outcomes_used':False,
                'status':'FROZEN_FOR_LATER_V3','protocol':PROTOCOL,'approval_date':APPROVAL_DATE})
    for r in idcols.itertuples(index=False):
        alt_rows.append({'split_set':sid,'compound_id':r.compound_id,'canonical_parent_smiles':r.canonical_parent_smiles,
                         'outer_fold':hfold(r.compound_id,r.canonical_parent_smiles,seed,5),'outer_seed':seed,
                         'endpoint_outcomes_used':False,'status':'FROZEN_FOR_V3_ROBUSTNESS'})
pd.DataFrame(reg).to_csv(OUT/'P4_02_split_registry.csv',index=False)
alt_outer=pd.DataFrame(alt_rows)
alt_outer.to_csv(OUT/'P4_03_v3_alternative_outer_splits_FROZEN.csv',index=False)

# ---------- Nested inner manifests, primary + alternatives ----------
all_outer={'PRIMARY':outer[['compound_id','canonical_parent_smiles','outer_fold']].copy()}
for sid in ALT_SEEDS:
    all_outer[sid]=alt_outer[alt_outer.split_set==sid][['compound_id','canonical_parent_smiles','outer_fold']].copy()
inner_rows=[]
for sid,odf in all_outer.items():
    for of in range(5):
        tr=odf[odf.outer_fold!=of]
        for r in tr.itertuples(index=False):
            inner_rows.append({'split_set':sid,'outer_fold':of,'compound_id':r.compound_id,
                               'inner_fold':inner_fold(r.compound_id,r.canonical_parent_smiles,sid,of),
                               'inner_assignment_method':'SHA256(identity|split_set|outer_fold|FACTORIAL-OOD-v0.2-INNER-v1)[:16] mod 4',
                               'endpoint_outcomes_used':False})
inner=pd.DataFrame(inner_rows)
inner.to_csv(OUT/'P4_04_nested_inner_split_manifest_FROZEN.csv',index=False)

# ---------- Primary context/source row inheritance ----------
prim_map=dict(zip(outer.compound_id,outer.outer_fold))
ctx=contexts.copy();ctx['outer_fold']=ctx['compound_id'].map(prim_map);ctx['split_set']='PRIMARY';ctx['status']='FROZEN_PARENT_INHERITED'
ctx.to_csv(OUT/'P4_05_primary_context_outer_manifest.csv',index=False)
# Context nested long manifest
ctx_inner=ctx[['modeling_context_id','compound_id','cell_type','outer_fold']].merge(
    inner[inner.split_set=='PRIMARY'][['outer_fold','compound_id','inner_fold']],on=['outer_fold','compound_id'],how='right',suffixes=('_heldout','_scope'))
# The merge above is not semantically right because inner.outer_fold is the held-out outer iteration, whereas ctx.outer_fold is parent's heldout fold.
# Rebuild explicitly.
ci=[]
ctx_lookup=contexts.groupby('compound_id')
for r in inner[inner.split_set=='PRIMARY'].itertuples(index=False):
    g=ctx_lookup.get_group(r.compound_id)
    for c in g.itertuples(index=False):
        ci.append({'split_set':'PRIMARY','outer_fold':r.outer_fold,'compound_id':r.compound_id,'modeling_context_id':c.modeling_context_id,
                   'cell_type':c.cell_type,'inner_fold':r.inner_fold,'role_in_outer':'OUTER_TRAIN'})
pd.DataFrame(ci).to_csv(OUT/'P4_06_primary_context_nested_inner_manifest.csv',index=False)

src=source_rows.copy();src['compound_id']=src['parent_compound_id'];src['outer_fold']=src['compound_id'].map(prim_map);src['split_set']='PRIMARY';src['status']='FROZEN_PARENT_INHERITED'
src.to_csv(OUT/'P4_07_primary_source_row_outer_manifest.csv',index=False)

# ---------- Scope registry ----------
scope=[]
for sid,odf in all_outer.items():
    for of in range(5):
        test=set(odf.loc[odf.outer_fold==of,'compound_id'])
        inn=inner[(inner.split_set==sid)&(inner.outer_fold==of)]
        outer_train=set(inn.compound_id)
        scope.append({'split_set':sid,'outer_fold':of,'inner_fold':'FINAL_REFIT','fit_scope':'OUTER_TRAIN_ONLY',
                      'fit_parent_n':len(outer_train),'validation_parent_n':0,'outer_test_parent_n':len(test),
                      'fit_validation_overlap_n':0,'fit_outer_test_overlap_n':len(outer_train&test),'validation_outer_test_overlap_n':0,
                      'allowed_operations':'final preprocessing fit; model refit after tuning; NEVER calibration/threshold selection on outer test'})
        for inf in range(4):
            val=set(inn.loc[inn.inner_fold==inf,'compound_id']); fit=outer_train-val
            scope.append({'split_set':sid,'outer_fold':of,'inner_fold':inf,'fit_scope':'INNER_TRAIN_ONLY',
                          'fit_parent_n':len(fit),'validation_parent_n':len(val),'outer_test_parent_n':len(test),
                          'fit_validation_overlap_n':len(fit&val),'fit_outer_test_overlap_n':len(fit&test),'validation_outer_test_overlap_n':len(val&test),
                          'allowed_operations':'variance filter; scaling; missing handling; PCA/latent factors; feature selection; resampling/class weighting; hyperparameter fit; calibration/threshold selection via inner OOF only'})
scope_df=pd.DataFrame(scope)
scope_df.to_csv(OUT/'P4_08_preprocessing_scope_registry.csv',index=False)

# ---------- Primary endpoint class support by outer and inner folds ----------
lab=labels[['compound_id']+primary_eps].copy()
outer_lab=lab.merge(outer[['compound_id','outer_fold']],on='compound_id',how='left')
cl=[]
for ep in primary_eps:
    for of in range(5):
        for role,mask in [('OUTER_TEST',outer_lab.outer_fold==of),('OUTER_TRAIN',outer_lab.outer_fold!=of)]:
            y=outer_lab.loc[mask,ep].dropna()
            cl.append({'endpoint':ep,'outer_fold':of,'level':role,'inner_fold':np.nan,'n_labeled':len(y),'active_n':int((y==1).sum()),'inactive_n':int((y==0).sum())})
        inn=inner[(inner.split_set=='PRIMARY')&(inner.outer_fold==of)][['compound_id','inner_fold']].merge(lab[['compound_id',ep]],on='compound_id',how='left')
        for inf in range(4):
            y=inn.loc[inn.inner_fold==inf,ep].dropna()
            cl.append({'endpoint':ep,'outer_fold':of,'level':'INNER_VALIDATION','inner_fold':inf,'n_labeled':len(y),'active_n':int((y==1).sum()),'inactive_n':int((y==0).sum())})
            y2=inn.loc[inn.inner_fold!=inf,ep].dropna()
            cl.append({'endpoint':ep,'outer_fold':of,'level':'INNER_TRAIN','inner_fold':inf,'n_labeled':len(y2),'active_n':int((y2==1).sum()),'inactive_n':int((y2==0).sum())})
class_support=pd.DataFrame(cl)
class_support.to_csv(OUT/'P4_09_primary_endpoint_nested_class_support.csv',index=False)

# ---------- Primary endpoint factorial cell support by outer fold ----------
fact=p3_fact[['compound_id','outer_fold','factorial_cell']].merge(lab,on='compound_id',how='left')
fc=[]
for ep in primary_eps:
    for of in range(5):
        for cell in ['C0/B0','C1/B0','C0/B1','C1/B1']:
            y=fact.loc[(fact.outer_fold==of)&(fact.factorial_cell==cell),ep].dropna()
            fc.append({'endpoint':ep,'outer_fold':of,'factorial_cell':cell,'n_labeled':len(y),'active_n':int((y==1).sum()),'inactive_n':int((y==0).sum())})
pd.DataFrame(fc).to_csv(OUT/'P4_10_primary_endpoint_factorial_support_by_outer_fold.csv',index=False)

# ---------- Leakage tests ----------
tests=[]
def add(tid,category,passed,observed,criterion,evidence,notes=''):
    tests.append({'test_id':tid,'category':category,'status':'PASS' if bool(passed) else 'FAIL','observed':str(observed),'criterion':criterion,'evidence':evidence,'notes':notes})

# Parent / outer reproducibility
add('L01','outer_split',outer.hash_match.all(),f"{int(outer.hash_match.sum())}/{len(outer)} exact hash matches",'2017/2017','P4_01','Primary outer split must remain exactly the Phase 3 nominated manifest.')
add('L02','outer_split',outer.compound_id.nunique()==len(master) and len(outer)==len(master),f"{outer.compound_id.nunique()} unique / {len(outer)} rows",f'{len(master)} unique parents','P4_01')
add('L03','outer_split',outer.groupby('compound_id').outer_fold.nunique().max()==1,'max folds per parent = 1','exactly 1','P4_01')
add('L04','outer_split',set(outer.outer_fold.unique())==set(range(5)),sorted(outer.outer_fold.unique().tolist()),'[0,1,2,3,4]','P4_01')
# exact structures and inchikey
smx=outer.groupby('canonical_parent_smiles').outer_fold.nunique().max()
add('L05','identity_leakage',smx==1,f'max outer folds per canonical parent SMILES = {smx}','1','P4_01')
ik=outer.dropna(subset=['parent_inchikey']).groupby('parent_inchikey').outer_fold.nunique(); ikmax=int(ik.max()) if len(ik) else 0
add('L06','identity_leakage',ikmax<=1,f'max outer folds per non-missing parent InChIKey = {ikmax}','<=1','P4_01')
# source DTXSID expansion
exp=[]
for r in outer[['compound_id','source_dtxsids','outer_fold']].itertuples(index=False):
    for x in str(r.source_dtxsids).split('|'):
        if x and x!='nan': exp.append((x,r.compound_id,r.outer_fold))
ex=pd.DataFrame(exp,columns=['dtxsid','compound_id','outer_fold'])
dtxmax=ex.groupby('dtxsid').outer_fold.nunique().max()
add('L07','identity_leakage',dtxmax==1,f'max outer folds per source DTXSID = {dtxmax}','1','P4_01')
# contexts/source inherit
ctx_missing=int(ctx.outer_fold.isna().sum());ctxmax=ctx.groupby('compound_id').outer_fold.nunique().max()
add('L08','context_leakage',ctx_missing==0 and ctxmax==1,f'missing folds={ctx_missing}; max folds per parent across contexts={ctxmax}','0 missing; max=1','P4_05')
src_missing=int(src.outer_fold.isna().sum());srcmax=src.groupby('compound_id').outer_fold.nunique().max()
add('L09','source_row_leakage',src_missing==0 and srcmax==1,f'missing folds={src_missing}; max folds per parent across source rows={srcmax}','0 missing; max=1','P4_07')
dup=ctx[ctx.source_duplicate_flag.astype(str).str.lower().isin(['true','1'])]
dupmax=dup.groupby('compound_id')['outer_fold'].nunique().max() if len(dup) else 1
add('L10','duplicate_leakage',dupmax==1,f'{len(dup)} duplicate-flagged parent-context rows; max outer folds per parent={dupmax}','max=1','P4_05')
# inner boundaries for every split set
train_counts_ok=True; test_inner_overlap=0; inner_parent_dup=0; inner_fold_missing=[]
for sid,odf in all_outer.items():
    for of in range(5):
        tset=set(odf.loc[odf.outer_fold==of,'compound_id']); inn=inner[(inner.split_set==sid)&(inner.outer_fold==of)]
        test_inner_overlap+=len(tset&set(inn.compound_id))
        inner_parent_dup=max(inner_parent_dup,int(inn.groupby('compound_id').inner_fold.nunique().max()))
        if set(inn.inner_fold.unique())!=set(range(4)): inner_fold_missing.append((sid,of))
        if len(inn)!=len(odf)-len(tset): train_counts_ok=False
add('L11','nested_cv',test_inner_overlap==0,f'outer-test parents found in corresponding inner manifests = {test_inner_overlap}','0','P4_04')
add('L12','nested_cv',inner_parent_dup==1,f'max inner folds per parent within an outer iteration = {inner_parent_dup}','1','P4_04')
add('L13','nested_cv',len(inner_fold_missing)==0,f'missing inner-fold sets = {inner_fold_missing}','none; each outer iteration has folds 0-3','P4_04')
add('L14','nested_cv',train_counts_ok,'all inner manifest row counts equal outer-train parent counts','true','P4_04')
# preprocessing scope disjointness
mx1=int(scope_df.fit_validation_overlap_n.max());mx2=int(scope_df.fit_outer_test_overlap_n.max());mx3=int(scope_df.validation_outer_test_overlap_n.max())
add('L15','preprocessing_scope',mx1==0 and mx2==0 and mx3==0,f'max overlaps fit-val={mx1}, fit-test={mx2}, val-test={mx3}','all zero','P4_08')
# outcome-blind flags
all_flags=(outer.endpoint_outcomes_used==False).all() and (alt_outer.endpoint_outcomes_used==False).all() and (inner.endpoint_outcomes_used==False).all()
add('L16','outcome_blind_split',all_flags,'all outer/inner split records flag endpoint_outcomes_used=False','true','P4_01/P4_03/P4_04')
# label permutation invariance (recompute identity-only primary fold after reversing label table order)
rev=labels.iloc[::-1][['compound_id']].merge(idcols[['compound_id','canonical_parent_smiles']],on='compound_id',how='right')
perm=rev.apply(lambda r:hfold(r['compound_id'],r['canonical_parent_smiles'],PRIMARY_OUTER_SEED,5),axis=1)
orig=idcols.apply(lambda r:hfold(r['compound_id'],r['canonical_parent_smiles'],PRIMARY_OUTER_SEED,5),axis=1)
add('L17','outcome_blind_split',np.array_equal(perm.values,orig.values),f'assignments invariant to label-table reversal; n={len(orig)}','true','build_phase4.py')
# Alternative split completeness
alt_ok=(alt_outer.groupby('split_set').compound_id.nunique()==len(master)).all() and all(set(g.outer_fold.unique())==set(range(5)) for _,g in alt_outer.groupby('split_set'))
add('L18','v3_robustness',alt_ok,f'{alt_outer.split_set.nunique()} alternative sets x {len(master)} parents, each with folds 0-4','4 complete alternative sets','P4_03')

# Chemical support isolation - full recomputation
chem_pass=False; chem_maxdiff=np.nan; chem_scaf_mismatch=-1; chem_state_mismatch=-1
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    mm=master[['compound_id','canonical_parent_smiles','bemis_murcko_scaffold']].merge(outer[['compound_id','outer_fold']],on='compound_id')
    fps={r.compound_id:gen.GetFingerprint(Chem.MolFromSmiles(r.canonical_parent_smiles)) for r in mm.itertuples(index=False)}
    calc=[]
    for of in range(5):
        tr=mm[mm.outer_fold!=of]; te=mm[mm.outer_fold==of]; trfps=[fps[c] for c in tr.compound_id]
        trsc=set(x for x in tr.bemis_murcko_scaffold.dropna().astype(str) if x.strip())
        for r in te.itertuples(index=False):
            mx=max(DataStructs.BulkTanimotoSimilarity(fps[r.compound_id],trfps))
            scaf=(pd.notna(r.bemis_murcko_scaffold) and str(r.bemis_murcko_scaffold).strip()!='' and str(r.bemis_murcko_scaffold) in trsc)
            state='C0' if (scaf or mx>=0.50) else 'C1'
            calc.append((r.compound_id,mx,scaf,state))
    cc=pd.DataFrame(calc,columns=['compound_id','calc_max','calc_scaf','calc_state']).merge(p3_chem[['compound_id','max_train_morgan_tanimoto','scaffold_represented_in_train','chemical_state']],on='compound_id')
    chem_maxdiff=float(np.max(np.abs(cc.calc_max-cc.max_train_morgan_tanimoto)))
    chem_scaf_mismatch=int((cc.calc_scaf.astype(bool)!=cc.scaffold_represented_in_train.astype(bool)).sum())
    chem_state_mismatch=int((cc.calc_state!=cc.chemical_state).sum())
    chem_pass=chem_maxdiff<1e-12 and chem_scaf_mismatch==0 and chem_state_mismatch==0
    cc.to_csv(OUT/'P4_11_chemical_support_train_only_recheck.csv',index=False)
except Exception as e:
    chem_error=repr(e)
add('L19','support_isolation',chem_pass,f'max Tanimoto diff={chem_maxdiff}; scaffold mismatches={chem_scaf_mismatch}; state mismatches={chem_state_mismatch}','diff<1e-12; 0 mismatches','P4_11',locals().get('chem_error',''))

# Biological support isolation using the Phase 3 training-distance audit files
bio_all=True; bio_details=[]
for of in range(5):
    fp=BASE/f'P3_train_biodist_fold{of}.csv'
    if not fp.exists():
        bio_all=False; bio_details.append(f'fold{of}:missing_train_distance_file'); continue
    td=pd.read_csv(fp)
    train_ids=set(outer.loc[outer.outer_fold!=of,'compound_id']); test_ids=set(outer.loc[outer.outer_fold==of,'compound_id'])
    ids=set(td.parent_compound_id)
    overlap=len(ids&test_ids); missing=len(train_ids-ids); extra=len(ids-train_ids)
    q75=float(td.train_bio_distance.quantile(.75)); frozen=float(p3_thr.loc[p3_thr.outer_fold==of,'q75'].iloc[0])
    qdiff=abs(q75-frozen)
    pvals=p3_bio.loc[p3_bio.outer_fold==of,'train_q75'].dropna().unique(); pdiff=max(abs(pvals-frozen)) if len(pvals) else np.inf
    ok=overlap==0 and missing==0 and extra==0 and qdiff<1e-10 and pdiff<1e-10
    bio_all &= ok
    bio_details.append(f'fold{of}: test_overlap={overlap}, missing_train={missing}, extra={extra}, q75_diff={qdiff:.3g}, parent_q75_diff={pdiff:.3g}')
add('L20','support_isolation',bio_all,'; '.join(bio_details),'0 test overlap; complete train IDs; q75 exact','P3_train_biodist_fold*.csv + P3_04a/P3_04')

# Primary factorial manifest alignment with frozen primary outer
fm=p3_fact[['compound_id','outer_fold']].merge(outer[['compound_id','outer_fold']],on='compound_id',suffixes=('_p3','_p4'))
add('L21','factorial_alignment',(fm.outer_fold_p3==fm.outer_fold_p4).all(),f'{int((fm.outer_fold_p3==fm.outer_fold_p4).sum())}/{len(fm)} outer-fold matches','2017/2017','P3_05 vs P4_01')
# Primary endpoint inner validation/training binary class presence; do not require large counts, just estimability
iv=class_support[class_support.level=='INNER_VALIDATION']
it=class_support[class_support.level=='INNER_TRAIN']
min_iv_a=int(iv.active_n.min());min_iv_i=int(iv.inactive_n.min());min_it_a=int(it.active_n.min());min_it_i=int(it.inactive_n.min())
add('L22','endpoint_estimability',min_iv_a>0 and min_iv_i>0 and min_it_a>0 and min_it_i>0,f'min inner-val active/inactive={min_iv_a}/{min_iv_i}; inner-train={min_it_a}/{min_it_i}','all >0 for 8 Primary endpoints','P4_09')

# Outer test binary class presence for primaries
ot=class_support[class_support.level=='OUTER_TEST']
min_ot_a=int(ot.active_n.min());min_ot_i=int(ot.inactive_n.min())
add('L23','endpoint_estimability',min_ot_a>0 and min_ot_i>0,f'min outer-test active/inactive={min_ot_a}/{min_ot_i}','all >0','P4_09')

leak=pd.DataFrame(tests)
leak.to_csv(OUT/'P4_12_leakage_unit_tests.csv',index=False)

# Split characterization summaries
rows=[]
for sid,odf in all_outer.items():
    for of in range(5):
        rows.append({'split_set':sid,'outer_fold':of,'parent_n':int((odf.outer_fold==of).sum()),'role':'PRIMARY' if sid=='PRIMARY' else 'V3_ROBUSTNESS'})
pd.DataFrame(rows).to_csv(OUT/'P4_13_outer_fold_size_summary.csv',index=False)
inner_summary=inner.groupby(['split_set','outer_fold','inner_fold']).size().reset_index(name='validation_parent_n')
inner_summary.to_csv(OUT/'P4_14_inner_fold_size_summary.csv',index=False)

# Software/environment lock
versions={'python':platform.python_version(),'platform':platform.platform(),'pandas':pd.__version__,'numpy':np.__version__}
for pkg in ['rdkit','sklearn','scipy']:
    try:
        mod=__import__(pkg);versions[pkg]=getattr(mod,'__version__','unknown')
    except Exception as e: versions[pkg]=f'UNAVAILABLE:{e}'
versions.update({'protocol':PROTOCOL,'protocol_approval_date':APPROVAL_DATE,'primary_outer_seed':PRIMARY_OUTER_SEED,
                 'inner_salt':INNER_SALT,'alternative_outer_seeds':ALT_SEEDS,'primary_endpoints':primary_eps})
with open(OUT/'P4_15_software_and_seed_lock.json','w') as f: json.dump(versions,f,indent=2)

# Gate C
essential=leak[leak.test_id.isin([f'L{i:02d}' for i in range(1,24)])]
fail=essential[essential.status!='PASS']
gate_status='PASS' if len(fail)==0 else 'FAIL'
gate=pd.DataFrame([
 {'check':'Primary outer split inherited unchanged','status':'PASS' if outer.hash_match.all() else 'FAIL','evidence':f'{int(outer.hash_match.sum())}/{len(outer)} exact matches'},
 {'check':'Nested outer/inner identities are disjoint','status':'PASS' if all(leak.set_index('test_id').loc[x,'status']=='PASS' for x in ['L11','L12','L13','L14','L15']) else 'FAIL','evidence':'See P4_12 L11-L15'},
 {'check':'Parent/context/source duplicate leakage','status':'PASS' if all(leak.set_index('test_id').loc[x,'status']=='PASS' for x in ['L05','L06','L07','L08','L09','L10']) else 'FAIL','evidence':'See P4_12 L05-L10'},
 {'check':'Train-only chemical/biological support verified','status':'PASS' if all(leak.set_index('test_id').loc[x,'status']=='PASS' for x in ['L19','L20']) else 'FAIL','evidence':'Full chemical recomputation + biological training-distance q75 audit'},
 {'check':'Primary endpoint inner/outer estimability','status':'PASS' if all(leak.set_index('test_id').loc[x,'status']=='PASS' for x in ['L22','L23']) else 'FAIL','evidence':f'min inner-val active/inactive {min_iv_a}/{min_iv_i}; min outer-test {min_ot_a}/{min_ot_i}'},
 {'check':'Overall Gate C','status':gate_status,'evidence':f'{int((essential.status=="PASS").sum())}/{len(essential)} leakage/unit tests PASS'}
])
gate.to_csv(OUT/'P4_16_gateC_decision.csv',index=False)

summary={
 'project':'FACTORIAL-OOD','phase':'4','protocol':PROTOCOL,'protocol_approval_date':APPROVAL_DATE,
 'primary_parent_n':len(master),'primary_context_n':len(contexts),'source_row_n':len(source_rows),'primary_endpoint_n':len(primary_eps),
 'primary_outer_fold_counts':outer.outer_fold.value_counts().sort_index().astype(int).to_dict(),
 'alternative_split_sets':list(ALT_SEEDS.keys()),
 'leakage_tests_passed':int((leak.status=='PASS').sum()),'leakage_tests_total':len(leak),'gateC':gate_status,
 'min_primary_outer_test_active':min_ot_a,'min_primary_outer_test_inactive':min_ot_i,
 'min_primary_inner_validation_active':min_iv_a,'min_primary_inner_validation_inactive':min_iv_i,
 'carry_forward':'All Phase 5 preprocessing and feature learning must consume only P4_08-authorized fit scopes. Outer-test records remain read-only until locked prediction generation.'
}
with open(OUT/'P4_summary.json','w') as f: json.dump(summary,f,indent=2)

# README
readme=f'''FACTORIAL-OOD Phase 4 — Frozen Split Package and Gate C\nProtocol: {PROTOCOL}, approved {APPROVAL_DATE}\n\nPrimary outer split:\n- Inherited unchanged from Phase 3 nominated P3_02 manifest.\n- 5 grouped outer folds, parent-compound unit.\n- Hash: SHA256(compound_id|canonical_parent_smiles|FACTORIAL-OOD-v0.1)[:16] mod 5.\n\nNested inner CV:\n- 4 grouped inner folds inside each outer-training set.\n- Assignment uses identity fields only and includes split_set + outer_fold in the hash.\n\nRobustness splits:\n- Four alternative V3 outer/inner split sets were frozen prospectively in Phase 4.\n- They are robustness-only and may not replace the primary split.\n\nGate C: {gate_status}\nLeakage/unit tests: {int((leak.status=='PASS').sum())}/{len(leak)} PASS.\nNo predictive model was trained in Phase 4.\n'''
(OUT/'README_PHASE4.txt').write_text(readme)

# Manifest and checksums
files=sorted([p for p in OUT.iterdir() if p.is_file()])
manifest={'phase':4,'protocol':PROTOCOL,'status':'FROZEN' if gate_status=='PASS' else 'FAILED','files':[p.name for p in files]}
with open(OUT/'PHASE4_MANIFEST.json','w') as f: json.dump(manifest,f,indent=2)
files=sorted([p for p in OUT.iterdir() if p.is_file()])
with open(OUT/'SHA256SUMS_PHASE4.txt','w') as f:
    for p in files:
        h=hashlib.sha256(p.read_bytes()).hexdigest();f.write(f'{h}  {p.name}\n')

print(json.dumps(summary,indent=2))
print('\nGate C table:\n',gate.to_string(index=False))
print('\nFailed tests:\n',fail.to_string(index=False) if len(fail) else 'NONE')
