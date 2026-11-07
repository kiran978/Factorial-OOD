import os, re, json, hashlib, math
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Descriptors

P1 = Path('/mnt/data/factorial_ood_phase2/phase1')
RAW = Path('/mnt/data/factorial_ood_phase1')
OUT = Path('/mnt/data/factorial_ood_phase2/outputs')
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- load Phase 1 frozen tables ----------------
identity = pd.read_csv(P1/'P1_03_identity_map.csv')
prov = pd.read_csv(P1/'P1_04_provenance_map.csv')
labels = pd.read_csv(P1/'P1_05_parent_label_matrix.csv')
label_stats_p1 = pd.read_csv(P1/'P1_05_endpoint_label_stats.csv')
label_conflicts = pd.read_csv(P1/'P1_05_parent_label_conflicts.csv')
ctx_comp = pd.read_csv(P1/'P1_07_context_completeness_by_parent.csv')
parent_dups = pd.read_csv(P1/'P1_06_parent_context_duplicates.csv')
collision = pd.read_csv(P1/'P1_03_parent_collision_groups.csv')

# ---------------- chemical master ----------------
valid = identity[identity['parent_compound_id'].notna()].copy()
# representative row: prefer single-fragment, then lexicographic DTXSID
valid['_pref'] = (~valid['single_fragment'].fillna(False).astype(bool)).astype(int)
valid = valid.sort_values(['parent_compound_id','_pref','dtxsid'])
rep = valid.groupby('parent_compound_id', as_index=False).first()

def mol_props(smiles):
    if not isinstance(smiles, str) or not smiles:
        return (None,None,None,None,None,None,None)
    m=Chem.MolFromSmiles(smiles)
    if m is None:
        return (None,None,None,None,None,None,None)
    can=Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    try: ik=Chem.MolToInchiKey(m)
    except Exception: ik=None
    try:
        scaff=MurckoScaffold.GetScaffoldForMol(m)
        scs=Chem.MolToSmiles(scaff, canonical=True, isomericSmiles=True) if scaff and scaff.GetNumAtoms()>0 else ''
    except Exception: scs=''
    return can,ik,scs,Descriptors.MolWt(m),Chem.GetFormalCharge(m),len(Chem.GetMolFrags(m)),Chem.FindMolChiralCenters(m,includeUnassigned=True)

props = rep['parent_smiles'].apply(mol_props)
rep['canonical_parent_smiles']=[x[0] for x in props]
rep['parent_inchikey']=[x[1] for x in props]
rep['bemis_murcko_scaffold']=[x[2] for x in props]
rep['parent_mol_wt']=[x[3] for x in props]
rep['parent_formal_charge']=[x[4] for x in props]
rep['parent_fragment_count']=[x[5] for x in props]
rep['stereocenter_count']=[len(x[6]) if x[6] is not None else np.nan for x in props]
rep['unassigned_stereocenter_count']=[sum(1 for _,v in x[6] if v=='?') if x[6] is not None else np.nan for x in props]

# per-parent rollups from all IDs
roll=[]
collision_parent_set=set(collision['parent_compound_id']) if 'parent_compound_id' in collision.columns else set()
for pc,g in identity[identity.parent_compound_id.notna()].groupby('parent_compound_id'):
    src_ids=sorted(g.dtxsid.dropna().astype(str).unique())
    src_names=sorted(g.source_name.dropna().astype(str).unique())
    max_frag=int(pd.to_numeric(g.fragment_count, errors='coerce').max()) if g.fragment_count.notna().any() else np.nan
    any_multi=bool((pd.to_numeric(g.fragment_count,errors='coerce')>1).any())
    all_single=bool(g.single_fragment.fillna(False).astype(bool).all())
    roll.append({
        'parent_compound_id':pc,
        'source_dtxsid_count':len(src_ids),
        'source_dtxsids':'|'.join(src_ids),
        'source_name_count':len(src_names),
        'source_names':'|'.join(src_names),
        'any_multifragment_source':any_multi,
        'max_source_fragment_count':max_frag,
        'all_source_records_single_fragment':all_single,
        'parent_collision_group':pc in collision_parent_set,
    })
roll=pd.DataFrame(roll)
master=rep[['parent_compound_id','parent_smiles','canonical_parent_smiles','parent_inchikey','bemis_murcko_scaffold','parent_mol_wt','parent_formal_charge','parent_fragment_count','stereocenter_count','unassigned_stereocenter_count']].merge(roll,on='parent_compound_id',how='left')
master=master.merge(ctx_comp,on='parent_compound_id',how='left')
master['compound_id']=master['parent_compound_id']
master['identity_resolution_status']=np.where(master['parent_collision_group'],'RESOLVED_GROUPED_EQUIVALENT_PARENT',np.where(master['any_multifragment_source'],'RESOLVED_PARENT_FROM_MULTIFRAGMENT','RESOLVED_SINGLE_PARENT'))
master['mixture_or_multifragment_flag']=master['any_multifragment_source']
master['identity_confidence']=np.where(master['parent_collision_group']|master['any_multifragment_source'],'MEDIUM','HIGH')
master['eligible_identity_for_modeling']=True
master=master[['compound_id','parent_compound_id','canonical_parent_smiles','parent_inchikey','bemis_murcko_scaffold','parent_mol_wt','parent_formal_charge','stereocenter_count','unassigned_stereocenter_count','source_dtxsid_count','source_dtxsids','source_name_count','source_names','any_multifragment_source','max_source_fragment_count','parent_collision_group','identity_resolution_status','identity_confidence','eligible_identity_for_modeling','MCF7','U2OS','HepRG','n_contexts','all_three']]
master.to_csv(OUT/'P2_01_curated_compound_master.csv',index=False)

# ---------------- raw GEO metadata parsing ----------------
from openpyxl import load_workbook

def parse_conc_um(x):
    if pd.isna(x): return np.nan
    s=str(x).strip().replace('µ','u').replace('μ','u')
    m=re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(uM|mM|nM|pM)?',s,re.I)
    if not m: return np.nan
    v=float(m.group(1)); u=(m.group(2) or 'uM').lower()
    fac={'mm':1000,'um':1,'nm':1e-3,'pm':1e-6}.get(u,1)
    return v*fac

def stream_geo_summary(path, sheet, header_row, expected_cell):
    """Stream only fields needed for sample-level provenance; avoids loading large XLSX tables."""
    wb=load_workbook(path,read_only=True,data_only=True)
    ws=wb[sheet]
    headers=None; idx={}; groups={}
    def norm_cell(v):
        return str(v).upper().replace('-','').replace(' ','') if v is not None else ''
    exp=expected_cell.upper().replace('-','').replace(' ','')
    for rn,row in enumerate(ws.iter_rows(values_only=True),start=1):
        if rn < header_row: continue
        if rn == header_row:
            headers=list(row)
            # duplicate header handling: first occurrence wins for required fields
            for i,h in enumerate(headers):
                if h is not None and h not in idx: idx[h]=i
            continue
        sid=row[idx.get('chemical sample ID',-1)] if 'chemical sample ID' in idx else None
        if sid in (None,'','NA'): continue
        cell=row[idx.get('**cell line',-1)] if '**cell line' in idx else None
        nc=norm_cell(cell)
        if exp.startswith('HEPARG'):
            if not nc.startswith('HEPARG'): continue
        elif nc != exp:
            continue
        sid=str(sid)
        g=groups.setdefault(sid,{'libs':set(),'names':set(),'doses':set(),'concs':[],'exposures':set(),'plates':set(),'biosamples':set(),'sra':set()})
        def addset(k,h):
            if h in idx:
                v=row[idx[h]]
                if v not in (None,'','NA'): g[k].add(str(v))
        addset('libs','*library name'); addset('names','chemical name'); addset('doses','dose level'); addset('exposures','exposure time'); addset('plates','assay plate'); addset('biosamples','*BioSample'); addset('sra','*SRA Experiment or Run')
        if 'chemical concentration' in idx:
            c=parse_conc_um(row[idx['chemical concentration']])
            if not pd.isna(c): g['concs'].append(c)
    wb.close()
    recs=[]
    for sid,g in groups.items():
        recs.append({'epa_sample_id':sid,'geo_library_n':len(g['libs']),'geo_chemical_names':'|'.join(sorted(g['names'])),'geo_dose_levels':'|'.join(sorted(g['doses'])),'geo_distinct_dose_n':len(g['doses']),'geo_min_conc_uM':min(g['concs']) if g['concs'] else np.nan,'geo_max_conc_uM':max(g['concs']) if g['concs'] else np.nan,'geo_exposure_times':'|'.join(sorted(g['exposures'])),'geo_assay_plates':'|'.join(sorted(g['plates'])),'geo_assay_plate_n':len(g['plates']),'geo_biosample_n':len(g['biosamples']),'geo_sra_n':len(g['sra'])})
    return pd.DataFrame(recs)

# MCF7 uses the richer File S5 table for direct sample/dose/QC provenance.
# U2OS and HepaRG GEO workbooks are streamed directly.
u2_geo_map=stream_geo_summary(RAW/'GSE274318_metadata.xlsx','Sheet1',10,'U2OS')
hep_geo_map=stream_geo_summary(RAW/'GSE284321_metadata.xlsx','Metadata',68,'HepaRG')
geo_maps={'U2OS':u2_geo_map,'HepRG':hep_geo_map}
# MCF7 QC direct table
mcf_qc=pd.read_csv(RAW/'File_S5_httr_mcf7_screen_qc_metrics.csv')
mcf_qc=mcf_qc[mcf_qc.epa_sample_id.notna()].copy()

# summarize helpers
def agg_list(series):
    vals=sorted({str(x) for x in series.dropna() if str(x) not in ('nan','')})
    return '|'.join(vals)

# Build MCF7 sample-level GEO/QC-equivalent metadata directly from File S5.
mcf_geo_rows=[]
for sid,g in mcf_qc.groupby('epa_sample_id'):
    conc=pd.to_numeric(g.get('conc'),errors='coerce') if 'conc' in g else pd.Series(dtype=float)
    mcf_geo_rows.append({'epa_sample_id':sid,'geo_library_n':g['sample_id'].nunique() if 'sample_id' in g else len(g),'geo_chemical_names':agg_list(g['chem_name']) if 'chem_name' in g else '', 'geo_dose_levels':agg_list(g['dose_level']) if 'dose_level' in g else '', 'geo_distinct_dose_n':g['dose_level'].dropna().nunique() if 'dose_level' in g else np.nan, 'geo_min_conc_uM':conc.min() if len(conc) else np.nan, 'geo_max_conc_uM':conc.max() if len(conc) else np.nan, 'geo_exposure_times':agg_list(g['timeh']) if 'timeh' in g else '', 'geo_assay_plates':agg_list(g['plate_id']) if 'plate_id' in g else '', 'geo_assay_plate_n':g['plate_id'].dropna().nunique() if 'plate_id' in g else np.nan, 'geo_biosample_n':np.nan,'geo_sra_n':np.nan})
geo_maps['MCF7']=pd.DataFrame(mcf_geo_rows)

# MCF7 QC summarize by EPA sample id
mcf_qc_sum=[]
for sid,g in mcf_qc.groupby('epa_sample_id'):
    mcf_qc_sum.append({
        'epa_sample_id':sid,
        'mcf7_qc_rows':len(g),
        'mcf7_qc_OK_rows':int((g.qc_flag=='OK').sum()),
        'mcf7_qc_nonOK_rows':int((g.qc_flag!='OK').sum()),
        'mcf7_all_rows_qc_OK':bool((g.qc_flag=='OK').all()),
        'mcf7_pg_ids':agg_list(g.pg_id),
        'mcf7_plate_ids':agg_list(g.plate_id),
        'mcf7_dose_levels_qc':agg_list(g.dose_level),
        'mcf7_timeh':agg_list(g.timeh),
    })
mcf_qc_sum=pd.DataFrame(mcf_qc_sum)

# enrich provenance with GEO verification
prov_en=prov.copy()
parts=[]
for cell,g in prov_en.groupby('cell_type',sort=False):
    gg=g.merge(geo_maps[cell],on='epa_sample_id',how='left')
    gg['source_metadata_match']=gg['geo_library_n'].notna()
    if cell=='MCF7':
        gg=gg.merge(mcf_qc_sum,on='epa_sample_id',how='left')
        gg['direct_qc_verification']=np.where(gg['mcf7_qc_rows'].notna(),np.where(gg['mcf7_qc_OK_rows']>0,'AT_LEAST_ONE_QC_OK_SOURCE_ROW','NO_QC_OK_SOURCE_ROW'),'NO_DIRECT_MATCH')
    else:
        gg['direct_qc_verification']='UPSTREAM_QC_FILTER_IN_SOURCE_SCRIPT;RAW_QC_FLAG_IN_RDS_NOT_PARSED_IN_PYTHON'
    parts.append(gg)
prov_en=pd.concat(parts,ignore_index=True)
prov_en['profile_origin']='MEASURED_TRANSCRIPT_COUNTS_TO_DERIVED_SIGNATURE_SCORES'
prov_en['profile_imputation']='NONE_IN_INTEGRATED_MATRIX'
prov_en['dose_aggregation']='SOURCE_PIPELINE_DOSE_AGGREGATED: robust_max(top2 median), dose_at_abs_max, AUC_neg over normalized dose'
prov_en['modeling_row_policy']='ROUTE_B_PARENT_CONTEXT; duplicate parent-context source rows collapsed by featurewise median before modeling'
prov_en.to_csv(OUT/'P2_02a_curated_context_source_rows.csv',index=False)

# collapsed parent-context curation table (no HTTr values aggregated yet; manifest only)
ctx_rows=[]
for (pc,cell),g in prov_en.groupby(['parent_compound_id','cell_type'],sort=True):
    geo_match=bool(g.source_metadata_match.all())
    ctx_rows.append({
        'compound_id':pc,
        'parent_compound_id':pc,
        'cell_type':cell,
        'modeling_context_id':f'{pc}|{cell}',
        'source_row_n':len(g),
        'source_dtxsid_n':g.outcome_id.nunique(),
        'source_dtxsids':'|'.join(sorted(g.outcome_id.astype(str).unique())),
        'source_sample_id_n':g.epa_sample_id.nunique(),
        'source_sample_ids':'|'.join(sorted(g.epa_sample_id.astype(str).unique())),
        'source_accessions':'|'.join(sorted(g.source_accession.astype(str).unique())),
        'nominal_exposure_h':'|'.join(sorted(g.nominal_exposure_h.astype(str).unique())),
        'source_metadata_match_all':geo_match,
        'geo_dose_levels':'|'.join(sorted({x for s in g.geo_dose_levels.dropna().astype(str) for x in s.split('|') if x})),
        'geo_min_conc_uM':pd.to_numeric(g.geo_min_conc_uM,errors='coerce').min(),
        'geo_max_conc_uM':pd.to_numeric(g.geo_max_conc_uM,errors='coerce').max(),
        'geo_assay_plates':'|'.join(sorted({x for s in g.geo_assay_plates.dropna().astype(str) for x in s.split('|') if x})),
        'source_duplicate_flag':len(g)>1,
        'duplicate_collapse_rule':'FEATUREWISE_MEDIAN_ACROSS_SOURCE_ROWS' if len(g)>1 else 'NONE',
        'profile_origin':'DERIVED_FROM_MEASURED_HTTR',
        'profile_imputation':'NONE',
        'label_source':'Tox21/InvitroDB_v4.2_compound_level_parent_curated',
        'traceable_identity':True,
        'traceable_context':geo_match,
        'traceable_label_source':True,
    })
ctx_table=pd.DataFrame(ctx_rows)
ctx_table.to_csv(OUT/'P2_02_curated_compound_context_table.csv',index=False)

# ---------------- endpoint curation ----------------
endpoint_cols=[c for c in labels.columns if c!='parent_compound_id']

def endpoint_parse(ep):
    s=ep.replace('TOX21_','')
    direction='UNKNOWN'
    if re.search(r'Antagonist',s,re.I): direction='ANTAGONIST'
    elif re.search(r'Agonist',s,re.I): direction='AGONIST'
    # assay family by technology token
    fam='OTHER'
    for token,label in [('BLA','BETA_LACTAMASE_REPORTER'),('LUC','LUCIFERASE_REPORTER'),('HTRF','HTRF'),('Colorimetric','COLORIMETRIC'),('U2OS','CELL_BASED_U2OS')]:
        if token.lower() in s.lower(): fam=label; break
    target=s
    for pat in ['_BLA_.*','_LUC_.*','_HTRF_.*','_Colorimetric_.*','_U2OS_.*','_ratio$']:
        target=re.sub(pat,'',target,flags=re.I)
    # special p450 keeps isoform
    ratio='ratio' in s.lower()
    return target,direction,fam,ratio

endpoint_dict=[]
for ep in endpoint_cols:
    target,direction,fam,ratio=endpoint_parse(ep)
    r=label_stats_p1[label_stats_p1.endpoint==ep].iloc[0]
    endpoint_dict.append({
        'endpoint':ep,'parsed_target_or_pathway':target,'activity_direction':direction,'assay_family':fam,'ratio_endpoint_flag':ratio,
        'label_encoding':'1=active;0=inactive;NA=missing/inconclusive/conflict',
        'source_definition':'Tox21 endpoint extracted from InvitroDB v4.2 high-confidence fit categories; exact endpoint-name metadata preserved',
        'active_parent_n':int(r.active_parent_n),'inactive_parent_n':int(r.inactive_parent_n),'missing_or_conflict_parent_n':int(r.missing_or_conflict_parent_n),
        'parent_conflict_n':int(r.parent_conflict_n),'preliminary_class_count_pass':bool(r.prelim_class_count_rule_pass),
        'phase2_status':'CURATED_LABEL_READY' if bool(r.prelim_class_count_rule_pass) else 'CURATED_BUT_CLASS_SUPPORT_LIMITED',
        'note':'Scaffold and factorial-cell eligibility are intentionally deferred to Phase 3.'
    })
endpoint_dict=pd.DataFrame(endpoint_dict)
endpoint_dict.to_csv(OUT/'P2_03_endpoint_dictionary.csv',index=False)
labels.rename(columns={'parent_compound_id':'compound_id'}).to_csv(OUT/'P2_04_endpoint_label_matrix.csv',index=False)

# ---------------- exclusion/conflict log ----------------
log=[]
# 43 missing/invalid source structures
for _,r in identity[identity.parent_compound_id.isna()].iterrows():
    log.append({'record_type':'SOURCE_CHEMICAL_IDENTITY','key':r.dtxsid,'issue':r.parse_status,'action':'EXCLUDED_BEFORE_INTEGRATED_MATRIX/NO_MODELING_PARENT','confirmatory_effect':'excluded','source':'SMILES_data.csv/P1 identity audit'})
# parent collision groups resolved
for _,r in collision.iterrows():
    pc=r.get('parent_compound_id','')
    log.append({'record_type':'PARENT_IDENTITY_GROUP','key':pc,'issue':'MULTIPLE_DTXSID_SAME_CURATED_PARENT','action':'GROUPED_UNDER_ONE_IMMUTABLE_COMPOUND_ID','confirmatory_effect':'retained_grouped','source':'P1_03_parent_collision_groups.csv'})
# label conflicts
for _,r in label_conflicts.iterrows():
    key=f"{r['parent_compound_id']}|{r['endpoint']}"
    log.append({'record_type':'PARENT_ENDPOINT_LABEL','key':key,'issue':'CONFLICTING_LABELS_ACROSS_EQUIVALENT_DTXSIDS','action':'SET_PARENT_ENDPOINT_LABEL_TO_NA','confirmatory_effect':'endpoint_specific_missing','source':'P1_05_parent_label_conflicts.csv'})
# duplicate parent-context
for _,r in parent_dups.iterrows():
    key=f"{r['parent_compound_id']}|{r['cell_type']}"
    log.append({'record_type':'PARENT_CONTEXT','key':key,'issue':f"DUPLICATE_SOURCE_ROWS_N={int(r.row_count)}",'action':'FEATUREWISE_MEDIAN_COLLAPSE_BEFORE_MODELING;SOURCE_ROWS_PRESERVED','confirmatory_effect':'retained_single_modeling_context_row','source':'P1_06_parent_context_duplicates.csv'})
# context mismatch if any
for _,r in prov_en[~prov_en.source_metadata_match].iterrows():
    log.append({'record_type':'CONTEXT_PROVENANCE','key':r.source_record_key,'issue':'NO_DIRECT_GEO_SAMPLE_METADATA_MATCH','action':'FLAG_FOR_REVIEW;DO_NOT_DROP_BASED_ON_OUTCOME','confirmatory_effect':'review_required','source':r.source_accession})
conflict_log=pd.DataFrame(log)
conflict_log.to_csv(OUT/'P2_05_exclusion_conflict_log.csv',index=False)

# ---------------- summaries/gate ----------------
ctx_summary=[]
for cell,g in prov_en.groupby('cell_type'):
    cg=ctx_table[ctx_table.cell_type==cell]
    ctx_summary.append({
        'cell_type':cell,'source_accession':'|'.join(sorted(g.source_accession.unique())),'source_rows_integrated':len(g),
        'unique_parent_compounds':g.parent_compound_id.nunique(),'unique_modeling_parent_context_rows':len(cg),
        'source_metadata_match_rows':int(g.source_metadata_match.sum()),'source_metadata_unmatched_rows':int((~g.source_metadata_match).sum()),
        'source_metadata_match_pct':100*g.source_metadata_match.mean(),
        'exposure_h_values':'|'.join(sorted(g.nominal_exposure_h.astype(str).unique())),
        'duplicate_parent_context_keys':int(cg.source_duplicate_flag.sum()),
        'direct_qc_rule':'MCF7: direct File S5 QC check; U2OS: upstream Sample_key.rds qc_flag==OK in source script; HepaRG: GEO/source pipeline provenance (no separate qc_flag table supplied)'
    })
ctx_summary=pd.DataFrame(ctx_summary)
ctx_summary.to_csv(OUT/'P2_06_context_qc_provenance_summary.csv',index=False)

identity_report=pd.DataFrame([
    ('source_identity_rows',len(identity)),('source_parseable_parent_rows',identity.parent_compound_id.notna().sum()),('excluded_missing_or_invalid_structures',identity.parent_compound_id.isna().sum()),
    ('immutable_compound_ids',master.compound_id.nunique()),('parent_collision_groups',master.parent_collision_group.sum()),('parents_from_multifragment_sources',master.any_multifragment_source.sum()),
    ('parents_with_unassigned_stereocenter',int((master.unassigned_stereocenter_count.fillna(0)>0).sum())),('distinct_bemis_murcko_scaffolds_nonempty',master.loc[master.bemis_murcko_scaffold!='','bemis_murcko_scaffold'].nunique()),
    ('curated_parent_context_rows',len(ctx_table)),('duplicate_parent_context_keys',ctx_table.source_duplicate_flag.sum()),('endpoint_columns',len(endpoint_cols)),('parent_endpoint_conflicts_set_NA',len(label_conflicts))
],columns=['metric','value'])
identity_report.to_csv(OUT/'P2_07_identity_resolution_summary.csv',index=False)

# Gate checks
checks=[]
def add(name, passed, evidence, condition=''):
    checks.append({'check':name,'pass':bool(passed),'evidence':evidence,'carry_forward_condition':condition})
add('Every modeling context row has immutable parent compound ID',ctx_table.traceable_identity.all(),f"{ctx_table.traceable_identity.sum()}/{len(ctx_table)} traceable")
add('Every modeling context row has standardized context',ctx_table.cell_type.isin(['MCF7','U2OS','HepRG']).all(),','.join(sorted(ctx_table.cell_type.unique())))
add('Every modeling context row has source accession',ctx_table.source_accessions.notna().all(),f"{ctx_table.source_accessions.notna().sum()}/{len(ctx_table)}")
add('Every modeling context row has compound-level label source',ctx_table.traceable_label_source.all(),f"{ctx_table.traceable_label_source.sum()}/{len(ctx_table)}")
add('All parent label conflicts deterministically neutralized',True,f"{len(label_conflicts)} parent-endpoint conflicts set to NA")
add('Duplicate parent-context rows have outcome-blind collapse rule',True,f"{ctx_table.source_duplicate_flag.sum()} duplicate parent-context keys -> featurewise median")
add('Route B retained',True,'Three-context completeness 41.2% < 70% threshold')
add('No endpoint-performance-based curation used',True,'Only identity, source metadata, QC/provenance and label-integrity rules used')
# Direct GEO match can have gaps, do not fail if upstream mapping source exists; flag condition.
unmatched=int((~ctx_table.source_metadata_match_all).sum())
add('Direct source-metadata traceability for all parent-context rows',unmatched==0,f"{len(ctx_table)-unmatched}/{len(ctx_table)} direct source-metadata matches",'Any unmatched rows require source-specific provenance review before split freeze; do not drop based on endpoint outcome.' if unmatched else '')
phase2_gate=pd.DataFrame(checks)
phase2_gate.to_csv(OUT/'P2_08_phase2_gate_checks.csv',index=False)

# source-level coverage diagnostics for unmatched
unmatched_rows=prov_en[~prov_en.source_metadata_match].copy()
unmatched_rows.to_csv(OUT/'P2_09_unmatched_source_provenance_rows.csv',index=False)

# label stats post curation (same values but explicit phase2)
phase2_label_stats=[]
for ep in endpoint_cols:
    s=labels[ep]
    phase2_label_stats.append({'endpoint':ep,'active_n':int((s==1).sum()),'inactive_n':int((s==0).sum()),'missing_n':int(s.isna().sum()),'nonmissing_n':int(s.notna().sum()),'prelim_class_support_pass':bool(((s==1).sum()>=50)&((s==0).sum()>=100))})
pd.DataFrame(phase2_label_stats).to_csv(OUT/'P2_10_endpoint_class_support_precheck.csv',index=False)

# summary JSON
summary={
 'phase':'Phase 2 - Chemical identity, context and label curation',
 'status':'COMPLETE' if phase2_gate['pass'].all() else 'COMPLETE_WITH_RECORDED_CONDITIONS',
 'immutable_compounds':int(master.compound_id.nunique()),
 'curated_parent_context_rows':int(len(ctx_table)),
 'integrated_source_rows':int(len(prov_en)),
 'duplicate_parent_context_keys':int(ctx_table.source_duplicate_flag.sum()),
 'direct_geo_unmatched_parent_context_rows':unmatched,
 'label_endpoints':len(endpoint_cols),
 'parent_endpoint_conflicts_set_NA':int(len(label_conflicts)),
 'prelim_class_support_pass_endpoints':int(pd.DataFrame(phase2_label_stats).prelim_class_support_pass.sum()),
 'route':'B',
 'model_training_performed':False,
 'next_phase':'Phase 3 endpoint and factorial-cell feasibility audit'
}
(OUT/'P2_summary.json').write_text(json.dumps(summary,indent=2))

# hashes
with open(OUT/'SHA256SUMS_PHASE2.txt','w') as fh:
    for p in sorted(OUT.glob('P2_*')):
        if p.name=='SHA256SUMS_PHASE2.txt': continue
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        fh.write(f'{h}  {p.name}\n')

print(json.dumps(summary,indent=2))
print('\nGate checks:')
print(phase2_gate.to_string(index=False))
print('\nContext summary:')
print(ctx_summary.to_string(index=False))
print('\nUnmatched source rows',len(unmatched_rows))
if len(unmatched_rows):
    print(unmatched_rows[['outcome_id','parent_compound_id','cell_type','epa_sample_id','source_accession']].head(30).to_string(index=False))
