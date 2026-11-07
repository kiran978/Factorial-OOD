from pathlib import Path
import os, json, hashlib, zipfile, shutil, platform
import numpy as np, pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.preprocessing import StandardScaler
import sklearn, scipy

BASE=Path('/mnt/data')
P2=BASE/'phase2_pkg'
P4=BASE/'p4inspect'/'outputs'
P5BZIP=BASE/'Phase5B_UPLOAD_TO_CHAT.zip'
OUT=BASE/'phase5_final'
if OUT.exists(): shutil.rmtree(OUT)
for d in [OUT, OUT/'outputs', OUT/'validation', OUT/'upstream_evidence']:
    d.mkdir(parents=True, exist_ok=True)

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

master=pd.read_csv(P2/'P2_01_curated_compound_master.csv')
ctx=pd.read_csv(P2/'P2_02_curated_compound_context_table.csv').reset_index(drop=True)
outer=pd.read_csv(P4/'P4_01_primary_outer_split_manifest_FROZEN.csv')
inner_all=pd.read_csv(P4/'P4_04_nested_inner_split_manifest_FROZEN.csv')
inner=inner_all[inner_all['split_set'].astype(str)=='PRIMARY'].copy()

# Load Phase5B validation outputs
with zipfile.ZipFile(P5BZIP) as z:
    scope_summary=pd.read_csv(z.open('qa/P5B_PCA_scope_summary.csv'))
    local_tests=pd.read_csv(z.open('qa/P5B_feature_leakage_and_integrity_tests.csv'))
    transformer_reg=pd.read_csv(z.open('qa/P5B_transformer_checksum_registry.csv'))
    score_reg=pd.read_csv(z.open('qa/P5B_score_checksum_registry.csv'))
    context_manifest=pd.read_csv(z.open('qa/P5B_context_row_index_manifest.csv'))
    software_p5b=json.load(z.open('qa/P5B_software_and_method_lock.json'))
    feature_names=z.read('qa/P5B_HTTr_feature_names.txt').decode().splitlines()

# Structure features: immutable parent-level features, no labels.
morgan=GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
fp=np.zeros((len(master),2048),dtype=np.uint8)
desc_rows=[]
parse_fail=[]
for i,r in master.iterrows():
    smi=r['canonical_parent_smiles']
    mol=Chem.MolFromSmiles(str(smi))
    if mol is None:
        parse_fail.append((r['compound_id'],smi)); continue
    arr=np.asarray(morgan.GetFingerprintAsNumPy(mol),dtype=np.uint8)
    fp[i,:]=arr
    desc_rows.append({
        'compound_id':r['compound_id'],
        'MW':Descriptors.MolWt(mol),
        'LogP':Crippen.MolLogP(mol),
        'TPSA':rdMolDescriptors.CalcTPSA(mol),
        'HBD':Lipinski.NumHDonors(mol),
        'HBA':Lipinski.NumHAcceptors(mol),
        'RotatableBonds':Lipinski.NumRotatableBonds(mol),
        'RingCount':Lipinski.RingCount(mol),
        'FractionCSP3':rdMolDescriptors.CalcFractionCSP3(mol),
        'HeavyAtomCount':mol.GetNumHeavyAtoms(),
        'FormalCharge':Chem.GetFormalCharge(mol),
    })
if parse_fail: raise RuntimeError(f'RDKit parse failures: {parse_fail[:5]}')
desc=pd.DataFrame(desc_rows)
desc_cols=[c for c in desc.columns if c!='compound_id']
if not np.isfinite(desc[desc_cols].to_numpy(float)).all(): raise RuntimeError('nonfinite descriptors')
np.savez_compressed(OUT/'outputs'/'P5_01_Morgan_r2_2048_parent_matrix.npz', compound_id=master['compound_id'].astype(str).to_numpy(), morgan_bits=fp)
desc.to_csv(OUT/'outputs'/'P5_02_physicochemical_descriptors_raw.csv',index=False)

structure_manifest=master[['compound_id','parent_compound_id','canonical_parent_smiles','parent_inchikey','bemis_murcko_scaffold']].copy()
structure_manifest['morgan_radius']=2
structure_manifest['morgan_bits']=2048
structure_manifest['morgan_include_chirality']=False
structure_manifest['descriptor_count']=10
structure_manifest['structure_feature_dimension_raw']=2058
structure_manifest.to_csv(OUT/'outputs'/'P5_03_structure_feature_manifest.csv',index=False)

# Descriptor scaler registry: fit on UNIQUE training parents in each scope.
desc_idx=desc.set_index('compound_id')
scaler_rows=[]
for f in range(5):
    outer_train=set(outer.loc[outer.outer_fold!=f,'compound_id'])
    outer_test=set(outer.loc[outer.outer_fold==f,'compound_id'])
    X=desc_idx.loc[sorted(outer_train),desc_cols].to_numpy(float)
    sc=StandardScaler().fit(X)
    row={'scope_id':f'P5_OUTER{f}_FINAL_REFIT','outer_fold':f,'inner_fold':'FINAL_REFIT','fit_parent_n':len(outer_train),'eval_parent_n':len(outer_test)}
    for k,c in enumerate(desc_cols): row[f'{c}_mean']=sc.mean_[k]; row[f'{c}_scale']=sc.scale_[k]
    scaler_rows.append(row)
    sub=inner[inner.outer_fold==f]
    for j in range(4):
        val=set(sub.loc[sub.inner_fold==j,'compound_id']); tr=set(sub.loc[sub.inner_fold!=j,'compound_id'])
        X=desc_idx.loc[sorted(tr),desc_cols].to_numpy(float)
        sc=StandardScaler().fit(X)
        row={'scope_id':f'P5_OUTER{f}_INNER{j}','outer_fold':f,'inner_fold':j,'fit_parent_n':len(tr),'eval_parent_n':len(val)}
        for k,c in enumerate(desc_cols): row[f'{c}_mean']=sc.mean_[k]; row[f'{c}_scale']=sc.scale_[k]
        scaler_rows.append(row)
scalers=pd.DataFrame(scaler_rows)
scalers.to_csv(OUT/'outputs'/'P5_04_descriptor_scaler_registry.csv',index=False)

# Fusion row manifest and specification
fusion=ctx[['compound_id','parent_compound_id','modeling_context_id','cell_type']].copy().reset_index().rename(columns={'index':'context_row_index'})
foldmap=dict(zip(outer.compound_id,outer.outer_fold))
fusion['outer_fold']=fusion.compound_id.map(foldmap).astype(int)
fusion['structure_parent_row_index']=fusion.compound_id.map(dict(zip(master.compound_id,range(len(master))))).astype(int)
fusion.to_csv(OUT/'outputs'/'P5_05_fusion_context_row_manifest.csv',index=False)

fusion_spec={
    'protocol':'v0.2',
    'route':'B_grouped_context_rows',
    'structure_primary':{
        'morgan':{'radius':2,'fpSize':2048,'includeChirality':False,'scaled':False},
        'physicochemical_descriptors':desc_cols,
        'descriptor_scaling':'StandardScaler fitted on unique training parents inside each outer/inner scope',
        'dimension':2058,
    },
    'httr_primary':{
        'raw_variables':16821,
        'variance_filter':'training-scope variance > 0',
        'scaling':'training-scope StandardScaler',
        'pca':'randomized PCA, retain min(256, PCs reaching 80% training variance)',
        'observed_retained_components_all_25_scopes':256,
        'dimension':256,
    },
    'early_fusion':{
        'concatenation_order':['Morgan_2048','scaled_physchem_10','HTTr_PC_256'],
        'dimension':2314,
        'fit_scope_rule':'all learned transforms must correspond to the same frozen outer/inner training scope'
    },
    'late_fusion':{
        'inputs':['structure_model_probability','HTTr_model_probability'],
        'combination':'convex weighted probability combination',
        'weight_selection':'inner grouped CV only; never outer-test performance',
    },
    'route_B_prediction_aggregation':{
        'primary':'arithmetic mean of context-row probabilities to one parent probability',
        'sensitivity':'median of context-row probabilities',
        'bootstrap_unit':'parent compound'
    },
    'endpoint_independence':'feature construction does not use Tox21 endpoint values',
}
with open(OUT/'outputs'/'P5_06_fusion_specification.json','w') as f: json.dump(fusion_spec,f,indent=2)

# Copy concise Phase5B registries / summaries
scope_summary.to_csv(OUT/'outputs'/'P5_07_HTTr_PCA_scope_summary.csv',index=False)
transformer_reg.to_csv(OUT/'outputs'/'P5_08_HTTr_transformer_checksum_registry.csv',index=False)
score_reg.to_csv(OUT/'outputs'/'P5_09_HTTr_score_checksum_registry.csv',index=False)
local_tests.to_csv(OUT/'validation'/'P5_10_local_Phase5B_leakage_tests.csv',index=False)

# Independent validation of uploaded Phase5B package.
checks=[]
def ck(test_id,desc_,passed,detail): checks.append({'test_id':test_id,'description':desc_,'status':'PASS' if passed else 'FAIL','detail':str(detail)})
p5b_sha=sha256_file(P5BZIP)
ck('P5-IV01','Uploaded Phase5B ZIP matches reported SHA-256',p5b_sha=='699dc521e0be0906fbc9b270ba08106874a0aeeaf96b2936453d473af4eb8894',p5b_sha)
ck('P5-IV02','25 PCA scopes present',len(scope_summary)==25,len(scope_summary))
ck('P5-IV03','All local Phase5B QA tests PASS',bool((local_tests.status=='PASS').all()),f"{(local_tests.status=='PASS').sum()}/{len(local_tests)}")
ck('P5-IV04','All 25 PCA scopes retained 256 PCs',bool((scope_summary.pca_components_retained==256).all()),scope_summary.pca_components_retained.value_counts().to_dict())
ck('P5-IV05','No PCA scope reached 80% variance before 256 cap',bool((scope_summary.cumulative_variance_at_max_components<0.8).all()),f"range={scope_summary.cumulative_variance_at_max_components.min():.6f}-{scope_summary.cumulative_variance_at_max_components.max():.6f}")
ck('P5-IV06','All PCA scope scores finite',bool(scope_summary.finite_fit_scores.all()),scope_summary.finite_fit_scores.value_counts().to_dict())
ck('P5-IV07','25 transformer checksum records present',len(transformer_reg)==25,len(transformer_reg))
ck('P5-IV08','25 score checksum records present',len(score_reg)==25,len(score_reg))
ck('P5-IV09','4021 context-row manifest rows present',len(context_manifest)==4021,len(context_manifest))
ck('P5-IV10','16821 HTTr feature names present',len(feature_names)==16821,len(feature_names))

# Included file hash and row/index checks
with zipfile.ZipFile(P5BZIP) as z:
    score_map=score_reg.set_index('scope_id').to_dict('index')
    hash_bad=0; index_bad=0; shape_bad=0; parent_bad=0; partition_bad=0
    idx_to_parent=dict(zip(context_manifest.context_row_index.astype(int),context_manifest.compound_id))
    idx_to_outer=dict(zip(context_manifest.context_row_index.astype(int),context_manifest.outer_fold.astype(int)))
    for f in range(5):
        sid=f'P5B_OUTER{f}_FINAL_REFIT'
        raw=z.read('scores/'+score_map[sid]['score_file'])
        if hashlib.sha256(raw).hexdigest()!=score_map[sid]['score_sha256']: hash_bad+=1
        dat=np.load(__import__('io').BytesIO(raw))
        fi=dat['fit_context_row_index'].astype(int); te=dat['outer_test_context_row_index'].astype(int)
        if dat['fit_scores'].shape!=(len(fi),256) or dat['outer_test_scores'].shape!=(len(te),256): shape_bad+=1
        if set(fi)&set(te) or set(fi)|set(te)!=set(range(4021)): index_bad+=1
        if any(idx_to_outer[i]==f for i in fi) or any(idx_to_outer[i]!=f for i in te): index_bad+=1
        if {idx_to_parent[i] for i in fi}&{idx_to_parent[i] for i in te}: parent_bad+=1
        vals=[]
        for j in range(4):
            sid2=f'P5B_OUTER{f}_INNER{j}'
            raw=z.read('scores/'+score_map[sid2]['score_file'])
            if hashlib.sha256(raw).hexdigest()!=score_map[sid2]['score_sha256']: hash_bad+=1
            d=np.load(__import__('io').BytesIO(raw))
            tr=d['fit_context_row_index'].astype(int); va=d['inner_val_context_row_index'].astype(int)
            if d['fit_scores'].shape!=(len(tr),256) or d['inner_val_scores'].shape!=(len(va),256): shape_bad+=1
            if set(tr)&set(va) or set(tr)|set(va)!=set(fi): index_bad+=1
            if any(idx_to_outer[i]==f for i in np.concatenate([tr,va])): index_bad+=1
            if {idx_to_parent[i] for i in tr}&{idx_to_parent[i] for i in va}: parent_bad+=1
            vals.extend(va.tolist())
        if len(vals)!=len(set(vals)) or set(vals)!=set(fi): partition_bad+=1
    ck('P5-IV11','All 25 included score-file SHA-256 hashes match registry',hash_bad==0,hash_bad)
    ck('P5-IV12','All saved score matrices have expected 256-PC shapes',shape_bad==0,shape_bad)
    ck('P5-IV13','All outer/inner context-row scope relationships independently reproduce',index_bad==0,index_bad)
    ck('P5-IV14','Zero parent overlap across every saved fit/evaluation partition',parent_bad==0,parent_bad)
    ck('P5-IV15','Four inner validation folds partition each outer-training context set exactly once',partition_bad==0,partition_bad)

ck('P5-IV16','All 2017 parent structures parse with RDKit',len(parse_fail)==0,len(parse_fail))
ck('P5-IV17','Morgan matrix shape is 2017 x 2048',fp.shape==(2017,2048),fp.shape)
ck('P5-IV18','10-descriptor panel is finite for all 2017 parents',np.isfinite(desc[desc_cols].to_numpy(float)).all(),desc[desc_cols].shape)
ck('P5-IV19','25 training-scope descriptor scalers generated',len(scalers)==25,len(scalers))
ck('P5-IV20','Fusion manifest aligns all 4021 Route-B rows to one structural parent row',len(fusion)==4021 and fusion.structure_parent_row_index.notna().all(),len(fusion))
indep=pd.DataFrame(checks)
indep.to_csv(OUT/'validation'/'P5_11_independent_feature_and_leakage_validation.csv',index=False)
if not (indep.status=='PASS').all(): raise RuntimeError(indep[indep.status!='PASS'])

# Final Gate D? Phase 5 has no named gate in protocol; call Phase5 feature construction PASS.
gate=pd.DataFrame([
    ['Structure identities','PASS','2017/2017 parsed; Morgan + descriptors generated without endpoint outcomes'],
    ['Structure leakage safety','PASS','Raw fingerprints identity-only; descriptor scaling fitted within 25 frozen training scopes'],
    ['HTTr outer transforms','PASS','5/5 training-only final-refit PCA scopes completed'],
    ['HTTr inner transforms','PASS','20/20 training-only nested PCA scopes completed'],
    ['HTTr dimensionality','PASS','All scopes retained 256 PCs; 80% variance not reached before cap'],
    ['Fusion infrastructure','PASS','Early-fusion row alignment and late-fusion selection rule frozen'],
    ['Leakage/integrity audit','PASS','20 local + 20 independent final checks PASS'],
    ['Phase 5 decision','PASS','Leakage-safe feature construction complete; Phase 6 modeling may begin'],
],columns=['component','status','evidence'])
gate.to_csv(OUT/'outputs'/'P5_12_phase5_completion_decision.csv',index=False)

software={
    'protocol':'v0.2','phase':'5','status':'PASS',
    'feature_builder_environment':{'python':platform.python_version(),'rdkit':rdBase.rdkitVersion,'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__},
    'local_P5B_environment':software_p5b,
    'Phase5B_upload_sha256':p5b_sha,
    'Phase2_master_sha256':sha256_file(P2/'P2_01_curated_compound_master.csv'),
    'Phase4_outer_manifest_sha256':sha256_file(P4/'P4_01_primary_outer_split_manifest_FROZEN.csv'),
    'Phase4_inner_manifest_sha256':sha256_file(P4/'P4_04_nested_inner_split_manifest_FROZEN.csv'),
}
with open(OUT/'outputs'/'P5_13_software_and_input_lock.json','w') as f: json.dump(software,f,indent=2)

summary={
    'phase':'Phase 5 - Leakage-Safe Feature Construction','status':'PASS','protocol':'v0.2',
    'parents':2017,'route_B_context_rows':4021,
    'structure_dimension':2058,'morgan_bits':2048,'physchem_descriptors':10,
    'httr_raw_variables':16821,'pca_scopes':25,'outer_pca_scopes':5,'inner_pca_scopes':20,'httr_pc_dimension':256,
    'early_fusion_dimension':2314,
    'pca_cumulative_variance_range':[float(scope_summary.cumulative_variance_at_max_components.min()),float(scope_summary.cumulative_variance_at_max_components.max())],
    'local_leakage_tests_passed':int((local_tests.status=='PASS').sum()),
    'independent_validation_tests_passed':int((indep.status=='PASS').sum()),
    'phase6_authorized':True,
    'transformer_retention_note':'Full 25 PCA transformer NPZ files are retained on the user local machine; this closure package carries their immutable SHA-256 registry rather than duplicating ~0.406 GiB of transformer arrays.'
}
with open(OUT/'outputs'/'P5_summary.json','w') as f: json.dump(summary,f,indent=2)

readme='''FACTORIAL-OOD — Phase 5 Leakage-Safe Feature Construction\nProtocol v0.2\nStatus: PASS\n\nThis package closes Phase 5. It contains the parent-level Morgan and physicochemical feature definitions, training-scope descriptor scaler registry, fusion-row manifest, HTTr PCA scope/checksum registries, local Phase5B leakage tests, independent validation tests, and final completion decision.\n\nThe complete 25 PCA transformer NPZ objects remain on the user's local machine under Phase5B_Output/transformers. Their SHA-256 hashes are frozen in P5_08_HTTr_transformer_checksum_registry.csv. The 77 MiB Phase5B upload package has SHA-256 699dc521e0be0906fbc9b270ba08106874a0aeeaf96b2936453d473af4eb8894.\n\nNo Tox21 endpoint values or predictive performance metrics were used to construct features or transformations.\n'''
(OUT/'README_PHASE5.txt').write_text(readme)

# Global manifest excluding future report/package.
files=[]
for p in OUT.rglob('*'):
    if p.is_file(): files.append(p)
with open(OUT/'SHA256SUMS_PHASE5.txt','w') as f:
    for p in sorted(files): f.write(f'{sha256_file(p)}  {p.relative_to(OUT)}\n')

print('built',OUT)
print('checks',len(indep),'all pass', (indep.status=='PASS').all())
print(summary)
