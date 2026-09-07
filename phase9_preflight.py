#!/usr/bin/env python3
from pathlib import Path
import pandas as pd, json, hashlib, sys, zipfile

ROOT = Path("/home/lenovo/Documents/Kiran Gaikwad/Pharmacodia/Chem")
P9 = ROOT / "Phase9_Output"
P9.mkdir(exist_ok=True)
LOCK = ROOT / "P9_00_robustness_lock_v1.0.json"

EXPECTED_LOCK_SHA = "55f769210fa70d1be54904e1e59c7a639175811018a8373264fe3f3226027a01"
EXPECTED = {
 "P4_V3": "6fc68b0a51e1d696643969433dc6caea46a6eb6fd52075d00963f9113833aff1",
 "P4_REG": "c359b3c398b6fcc3420809daebc474be570adc7c1a07e9934dae7d3eaff38cc4",
 "P4_SEED": "d00a4b55dca922584ad69a8ea39a96e6a14935a9c110d4fb6af720d447aba707",
 "P6_REG": "10005b76d389dcc73466ef377cf4f5d2ee0787ff71b95523d08ab2b988276b57",
 "P6_CONTEXT": "e58e741dceb05730b699cc05f898468d8a964a07a7b333719232885e56bcb87f",
 "P6_PARENT": "45d1c87930de43b9a02a75eddf00b696fd2f2351dbc2955841e34bde42a2c965",
 "P3_THRESH": "bd39b12ca273f26926c2186d9c3e0e0d8ee02f99bf7ce678f64f18abe7175a80",
}

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def locate(name, preferred=()):
    for x in preferred:
        p=ROOT/x
        if p.exists(): return p
    hits=[p for p in ROOT.rglob(name) if "Phase9_Output" not in str(p)]
    if len(hits)==1:return hits[0]
    if len(hits)>1:
        # Prefer canonical Phase folders over extracted scratch directories.
        hits=sorted(hits,key=lambda p:(0 if any(str(p).startswith(str(ROOT/d)) for d in ["Phase3","Phase4","Phase6_Output","Phase7_Output","Phase7B_Output","Phase8_Output"]) else 1,len(str(p))))
        return hits[0]
    raise FileNotFoundError(name)

tests=[]
def T(i,desc,ok,detail=""):
    tests.append({"test_id":i,"description":desc,"status":"PASS" if ok else "FAIL","detail":str(detail)})

print("="*88)
print("FACTORIAL-OOD PHASE 9 — ROBUSTNESS PREFLIGHT (NO PERFORMANCE COMPUTATION)")
print("="*88)

T("P9-P01","Phase9 lock exists and checksum is frozen",LOCK.exists() and sha(LOCK)==EXPECTED_LOCK_SHA, sha(LOCK) if LOCK.exists() else "missing")
if not LOCK.exists() or sha(LOCK)!=EXPECTED_LOCK_SHA:
    pd.DataFrame(tests).to_csv(P9/"P9_PREFLIGHT_TESTS.csv",index=False); raise RuntimeError("Phase9 lock mismatch")

# Status locks.
for n,pref,tid in [
 ("P6_FINAL_STATUS.json",["Phase6_Output/P6_FINAL_STATUS.json"],"P9-P02"),
 ("P7_FINAL_STATUS.json",["Phase7_Output/P7_FINAL_STATUS.json"],"P9-P03"),
 ("P7B_FINAL_STATUS.json",["Phase7B_Output/P7B_FINAL_STATUS.json"],"P9-P04"),
 ("P8_FINAL_STATUS.json",["Phase8_Output/P8_FINAL_STATUS.json"],"P9-P05")]:
    p=locate(n,pref); st=json.load(open(p)); T(tid,f"{n.split('_')[0]} upstream PASS",st.get("status")=="PASS",st.get("status"))

p4v3=locate("P4_03_v3_alternative_outer_splits_FROZEN.csv",["Phase4/outputs/P4_03_v3_alternative_outer_splits_FROZEN.csv"])
p4reg=locate("P4_02_split_registry.csv",["Phase4/outputs/P4_02_split_registry.csv"])
p4seed=locate("P4_15_software_and_seed_lock.json",["Phase4/outputs/P4_15_software_and_seed_lock.json"])
p4inner=locate("P4_04_nested_inner_split_manifest_FROZEN.csv",["Phase4/outputs/P4_04_nested_inner_split_manifest_FROZEN.csv"])
p6reg=locate("P6_00_locked_computational_registry.json",["Phase6_Output/P6_00_locked_computational_registry.json"])
p6ctx=locate("P6_04_locked_context_OOF_raw_predictions.csv",["Phase6_Output/P6_04_locked_context_OOF_raw_predictions.csv"])
p6par=locate("P6_03_locked_parent_OOF_predictions.csv",["Phase6_Output/P6_03_locked_parent_OOF_predictions.csv"])
p6tun=locate("P6_05_tuning_calibration_threshold_registry.csv",["Phase6_Output/P6_05_tuning_calibration_threshold_registry.csv"])
p3bio=locate("P3_04_biological_support_by_parent.csv",["Phase3/outputs/P3_04_biological_support_by_parent.csv"])
p3chem=locate("P3_03_chemical_support_by_parent.csv",["Phase3/outputs/P3_03_chemical_support_by_parent.csv"])
p3thr=locate("P3_04a_biological_support_fold_thresholds.csv",["Phase3/outputs/P3_04a_biological_support_fold_thresholds.csv"])
p5ctx=locate("P5B_context_row_index_manifest.csv",["Phase5B_Output/qa/P5B_context_row_index_manifest.csv"])

for key,p,tid in [
 ("P4_V3",p4v3,"P9-P06"),("P4_REG",p4reg,"P9-P07"),("P4_SEED",p4seed,"P9-P08"),
 ("P6_REG",p6reg,"P9-P09"),("P6_CONTEXT",p6ctx,"P9-P10"),("P6_PARENT",p6par,"P9-P11"),
 ("P3_THRESH",p3thr,"P9-P12")]:
    T(tid,f"{key} frozen SHA256 matches",sha(p)==EXPECTED[key],sha(p))

v3=pd.read_csv(p4v3)
T("P9-P13","V3 manifest has exactly 4 x 2017 rows",len(v3)==8068 and v3.split_set.nunique()==4 and (v3.groupby("split_set").size()==2017).all(),v3.groupby("split_set").size().to_dict())
T("P9-P14","Every V3 set contains outer folds 0..4",all(set(g.outer_fold)==set(range(5)) for _,g in v3.groupby("split_set")),v3.groupby("split_set").outer_fold.nunique().to_dict())
T("P9-P15","V3 assignments are outcome-blind",(~v3.endpoint_outcomes_used.astype(bool)).all(),v3.endpoint_outcomes_used.value_counts().to_dict())

inn=pd.read_csv(p4inner)
alts=inn[inn.split_set.astype(str).str.startswith("V3_ALT")]
ok=(alts.split_set.nunique()==4 and all(set(g.inner_fold)==set(range(4)) for _,g in alts.groupby(["split_set","outer_fold"])))
T("P9-P16","Frozen V3 nested inner manifests exist for all 4 sets x 5 outer folds",ok,alts.groupby(["split_set","outer_fold"]).inner_fold.nunique().value_counts().to_dict())

bio=pd.read_csv(p3bio)
needbio={"B_state_amended_q90","B_state_primary_q95","B_state_q975","B_state_worst_q95","biological_state","bio_distance_median","bio_distance_worst"}
T("P9-P17","All frozen biological robustness states available",needbio.issubset(bio.columns),sorted(set(needbio)-set(bio.columns)))

chem=pd.read_csv(p3chem)
needchem={"max_train_morgan_tanimoto","scaffold_represented_in_train","chemical_state"}
T("P9-P18","Frozen chemical support inputs available",needchem.issubset(chem.columns),sorted(set(needchem)-set(chem.columns)))

ctx=pd.read_csv(p6ctx)
comb=ctx[["model_family","modality"]].drop_duplicates()
T("P9-P19","Phase6 context OOF retained for all 3 families x HTTR/EARLY_FUSION",len(ctx)==122478 and len(comb)==6,len(ctx))
T("P9-P20","Context OOF contains no duplicate endpoint/fold/model/modality/context-row keys",
  not ctx.duplicated(["endpoint","outer_fold","model_family","modality","context_row_index"]).any(),
  int(ctx.duplicated(["endpoint","outer_fold","model_family","modality","context_row_index"]).sum()))

cm=pd.read_csv(p5ctx)
T("P9-P21","Context-row manifest is 4021 unique rows with MCF7/U2OS/HepRG",len(cm)==4021 and cm.context_row_index.nunique()==4021 and set(cm.cell_type)=={"MCF7","U2OS","HepRG"},cm.cell_type.value_counts().to_dict())

par=pd.read_csv(p6par)
learn=par[par.model_family.isin(["ELASTIC_NET","RANDOM_FOREST","HIST_GRAD_BOOST"]) & par.modality.isin(["STRUCTURE","HTTR","EARLY_FUSION","LATE_FUSION"])]
T("P9-P22","Locked learned parent OOF is 120000 rows / 8 endpoints / 12 model combinations",len(learn)==120000 and learn.endpoint.nunique()==8 and len(learn[["model_family","modality"]].drop_duplicates())==12,len(learn))

tun=pd.read_csv(p6tun)
needtun={"platt_coef","platt_intercept","decision_threshold","selected_structure_weight"}
T("P9-P23","Frozen calibration/threshold/fusion registry supports Route-B sensitivity",needtun.issubset(tun.columns),sorted(set(needtun)-set(tun.columns)))

tdf=pd.DataFrame(tests)
tdf.to_csv(P9/"P9_PREFLIGHT_TESTS.csv",index=False)
summary={"phase":"9","type":"preflight","performance_computed":False,"pass_n":int((tdf.status=="PASS").sum()),"total_n":len(tdf),"status":"PASS" if (tdf.status=="PASS").all() else "FAIL","lock_sha256":EXPECTED_LOCK_SHA}
json.dump(summary,open(P9/"P9_PREFLIGHT_STATUS.json","w"),indent=2)

print(tdf.to_string(index=False))
print("-"*88)
print(f"PHASE 9 PREFLIGHT: {summary['pass_n']}/{summary['total_n']} PASS")
print("NO PERFORMANCE / ROBUSTNESS RESULT WAS COMPUTED.")
if summary["status"]!="PASS": raise RuntimeError("Phase9 preflight failed")
print("SAFE TO START P9A/P9B/P9C UNDER THE FROZEN LOCK.")
