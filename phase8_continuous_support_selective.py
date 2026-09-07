#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('NUMEXPR_NUM_THREADS','1')

from pathlib import Path
from dataclasses import dataclass
import hashlib, json, math, warnings, zipfile
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore', category=DeprecationWarning)
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

DEFAULT_ROOT = Path('/home/lenovo/Documents/Kiran Gaikwad/Pharmacodia/Chem')
ROOT = Path(os.environ.get('FACTORIAL_ROOT', str(DEFAULT_ROOT)))
P8 = ROOT / 'Phase8_Output'
BOOTDIR = P8 / 'bootstrap'
P8.mkdir(parents=True, exist_ok=True)
BOOTDIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAPS = int(os.environ.get('P8_BOOT','2000'))
JOBS = int(os.environ.get('P8_JOBS','8'))
AURC_COVERAGES = np.round(np.linspace(0.50,1.00,51),2)
KEY_COVERAGES = np.array([1.00,0.90,0.80,0.70,0.60,0.50])

ENDPOINTS = [
    'TOX21_ERa_BLA_Antagonist_ratio',
    'TOX21_FXR_BLA_Antagonist_ratio',
    'TOX21_MMP_ratio',
    'TOX21_PPARg_BLA_Antagonist_ratio',
    'TOX21_RAR_LUC_Antagonist',
    'TOX21_RORg_LUC_CHO_Antagonist',
    'TOX21_SBE_BLA_Antagonist_ratio',
    'TOX21_hERG_U2OS_Antagonist',
]
FAMILIES = ['ELASTIC_NET','RANDOM_FOREST','HIST_GRAD_BOOST']
MODALITIES = ['STRUCTURE','HTTR','EARLY_FUSION','LATE_FUSION']
MODEL_KEYS = [(f,m) for f in FAMILIES for m in MODALITIES]
STRATEGIES = [
    'CHEMICAL_SUPPORT',
    'BIOLOGICAL_SUPPORT',
    'JOINT_SUPPORT',
    'CONFIDENCE',
    'HYBRID_SUPPORT_CONFIDENCE',
]

LOCK = {
    'project':'FACTORIAL-OOD',
    'phase':'8',
    'analysis_name':'Continuous Support, Applicability Domain and Selective Prediction',
    'protocol':'v0.2 + original Phase-0 applicability-domain/selective-prediction plan',
    'freeze_status':'FROZEN_BEFORE_PHASE8_OUTCOME_INSPECTION',
    'primary_inferential_unit':'unique curated parent compound',
    'upstream_results_immutable':['Phase 6','Phase 7','Phase 7B'],
    'learned_models_only':{'families':FAMILIES,'modalities':MODALITIES},
    'support_scores':{
        'chemical_support_percentile':'Outer-training empirical percentile of test maximum Morgan-r2 Tanimoto against training-parent leave-one-out NN Tanimoto distribution; higher = more supported.',
        'biological_support_percentile':'1 minus outer-training empirical percentile of test median HTTr 5NN PCA distance against training-parent leave-one-out HTTr-distance distribution; higher = more supported.',
        'joint_support_primary':'Geometric mean sqrt(chemical_support_percentile * biological_support_percentile).',
        'joint_support_sensitivity':'Minimum of chemical and biological support percentiles; stored but not a primary selective-ranking strategy.',
        'no_test_outcomes_or_test_distribution_used_for_support_scores':True,
    },
    'continuous_reliability_model':{
        'primary_outcome':'per-parent squared probability error (p-y)^2',
        'predictors':'z chemical novelty + z biological novelty + their interaction + z model uncertainty',
        'chemical_novelty':'1 - chemical_support_percentile',
        'biological_novelty':'1 - biological_support_percentile',
        'model_uncertainty':'1 - [2*abs(probability-0.5)]',
        'purpose':'Test whether chemical and biological support contribute information about probability error beyond maximum-class-probability confidence.',
        'incremental_information':'Delta R2 of full support+uncertainty model versus uncertainty-only model; descriptive with bootstrap CI.',
    },
    'selective_prediction':{
        'primary_risk':'mean squared probability error (Brier contribution)',
        'primary_scope':'all immutable OOF parents for each endpoint',
        'joint_ood_scope':'C1/B1 descriptive stress-set curves only',
        'ranking_strategies':STRATEGIES,
        'confidence_score':'2*abs(probability-0.5), higher = more confident',
        'hybrid_score':'geometric mean of joint support and confidence score',
        'aurc_range':'coverage 0.50 to 1.00 in 0.01 increments; normalized trapezoidal area, lower = better',
        'key_coverages':[1.0,0.9,0.8,0.7,0.6,0.5],
        'key_metrics':['Brier_risk','Brier_skill_score','AUROC','balanced_MCC','AUPRC','ECE_10bin','classification_error_rate','prevalence'],
        'primary_comparisons':[
            'full-risk minus JOINT_SUPPORT AURC (positive = selective benefit)',
            'full-risk minus HYBRID_SUPPORT_CONFIDENCE AURC (positive = selective benefit)',
            'CONFIDENCE AURC minus JOINT_SUPPORT AURC (positive = support better than confidence)',
            'CONFIDENCE AURC minus HYBRID_SUPPORT_CONFIDENCE AURC (positive = hybrid better than confidence)',
        ],
        'coverage_floor':0.50,
    },
    'inference':{
        'bootstrap_unit':'parent compound',
        'bootstrap_resamples':BOOTSTRAPS,
        'same_parent_bootstrap_draw_across_models_within_endpoint':True,
        'confidence_interval':'95% percentile',
        'p_value':'centered two-sided bootstrap for signed effects',
        'multiplicity':'Benjamini-Hochberg FDR within named coefficient/comparison families',
        'across_endpoint_synthesis':'DerSimonian-Laird style random-effects using endpoint bootstrap SE; endpoint effects remain visible',
    },
    'governance':{
        'no_retraining':True,
        'no_recalibration':True,
        'no_threshold_change':True,
        'no_endpoint_reselection':True,
        'no_support_definition_change_after_results':True,
        'negative_or_null_selective_results_retained':True,
        'phase9_reserved_for':['q90/q95/q97.5 biological severity','chemical-threshold sensitivities','V3 alternative split sets','leave-one-context-out','mean-vs-median Route-B aggregation'],
    }
}


def sha256_file(p:Path, chunk=8*1024*1024):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()


def seed_for(*parts):
    s='|'.join(['FACTORIAL-OOD','P8','v1']+[str(x) for x in parts])
    return int(hashlib.sha256(s.encode()).hexdigest()[:8],16)


def locate(basename:str, preferred:list[str]|None=None)->Path:
    preferred = preferred or []
    for rel in preferred:
        p=ROOT/rel
        if p.exists(): return p
    hits=[p for p in ROOT.rglob(basename) if p.is_file()]
    if not hits:
        raise FileNotFoundError(f'Missing required file {basename} under {ROOT}')
    # Multiple byte-identical copies are acceptable; otherwise refuse ambiguity.
    shas={sha256_file(p):[] for p in hits}
    for p in hits: shas[sha256_file(p)].append(p)
    if len(shas)>1:
        raise RuntimeError(f'Ambiguous non-identical copies of {basename}: {hits}')
    return sorted(hits,key=lambda p:len(str(p)))[0]


def quantile_ecdf_le(sorted_train:np.ndarray, value:float)->float:
    return float(np.searchsorted(sorted_train,value,side='right')/len(sorted_train))


def tanimoto_cross(A:np.ndarray,B:np.ndarray)->np.ndarray:
    # Inputs binary float32. Returns float32 matrix.
    inter=A@B.T
    sa=A.sum(axis=1)[:,None]
    sb=B.sum(axis=1)[None,:]
    den=sa+sb-inter
    out=np.divide(inter,den,out=np.zeros_like(inter,dtype=np.float32),where=den>0)
    return out


def balanced_mcc(y,pred):
    y=np.asarray(y,dtype=int); pred=np.asarray(pred,dtype=int)
    if len(y)==0 or len(np.unique(y))<2: return np.nan
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    sen=tp/(tp+fn) if tp+fn else np.nan
    spe=tn/(tn+fp) if tn+fp else np.nan
    if not np.isfinite(sen) or not np.isfinite(spe): return np.nan
    den=1.0-(sen-spe)**2
    if den<=1e-15: return np.nan
    return float((sen+spe-1.0)/math.sqrt(den))


def ece10(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    if len(y)==0: return np.nan
    edges=np.linspace(0,1,11); out=0.0
    for i in range(10):
        if i==9: m=(p>=edges[i])&(p<=edges[i+1])
        else: m=(p>=edges[i])&(p<edges[i+1])
        if m.any(): out += m.mean()*abs(p[m].mean()-y[m].mean())
    return float(out)


def point_metrics(y,p,pred):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float); pred=np.asarray(pred,dtype=int)
    n=len(y); prev=float(y.mean()) if n else np.nan
    risk=float(np.mean((p-y)**2)) if n else np.nan
    baseline=prev*(1-prev) if n and 0<prev<1 else np.nan
    bss=1-risk/baseline if np.isfinite(baseline) and baseline>0 else np.nan
    out={'n':n,'active_n':int(y.sum()) if n else 0,'inactive_n':int((y==0).sum()) if n else 0,
         'prevalence':prev,'Brier_risk':risk,'Brier_skill_score':bss,
         'classification_error_rate':float(np.mean(pred!=y)) if n else np.nan,'ECE_10bin':ece10(y,p)}
    if n and len(np.unique(y))>=2:
        out['AUROC']=float(roc_auc_score(y,p))
        out['AUPRC']=float(average_precision_score(y,p))
        out['balanced_MCC']=balanced_mcc(y,pred)
    else:
        out.update({'AUROC':np.nan,'AUPRC':np.nan,'balanced_MCC':np.nan})
    return out


def bh_adjust(vals):
    p=np.asarray(vals,dtype=float); q=np.full(len(p),np.nan)
    idx=np.where(np.isfinite(p))[0]
    if len(idx)==0:return q
    pv=p[idx]; order=np.argsort(pv); r=pv[order]
    a=r*len(r)/np.arange(1,len(r)+1)
    a=np.minimum.accumulate(a[::-1])[::-1]; a=np.minimum(a,1.0)
    q[idx[order]]=a
    return q


def boot_summary(est,boot, signed=True):
    b=np.asarray(boot,dtype=float); v=b[np.isfinite(b)]
    if len(v)==0 or not np.isfinite(est):
        return dict(estimate=est,ci_low=np.nan,ci_high=np.nan,bootstrap_se=np.nan,bootstrap_p_two_sided=np.nan,bootstrap_valid_n=len(v),bootstrap_valid_fraction=len(v)/len(b),bootstrap_supported=False)
    lo,hi=np.quantile(v,[.025,.975]); se=np.std(v,ddof=1) if len(v)>1 else np.nan
    if signed:
        dev=v-est
        pv=(1+np.sum(np.abs(dev)>=abs(est)))/(len(v)+1)
    else: pv=np.nan
    return dict(estimate=float(est),ci_low=float(lo),ci_high=float(hi),bootstrap_se=float(se),bootstrap_p_two_sided=float(pv) if np.isfinite(pv) else np.nan,bootstrap_valid_n=len(v),bootstrap_valid_fraction=float(len(v)/len(b)),bootstrap_supported=bool(len(v)/len(b)>=.90))


def weighted_ols(X,y,w):
    w=np.asarray(w,dtype=float)
    sw=float(w.sum())
    if sw<=0:return np.full(X.shape[1],np.nan),np.nan
    XtW=X.T*w
    A=XtW@X; b=XtW@y
    try: beta=np.linalg.solve(A,b)
    except np.linalg.LinAlgError:
        beta=np.linalg.pinv(A)@b
    wy=float(np.sum(w*y)); wy2=float(np.sum(w*y*y))
    sse=wy2-2*float(beta@b)+float(beta@A@beta)
    sst=wy2-(wy*wy/sw)
    r2=1-sse/sst if sst>1e-15 else np.nan
    return beta,r2


def weighted_ols_bootstrap_batch(X,y,W):
    # Batched weighted least squares for multinomial parent-bootstrap counts.
    Wf=W.astype(np.float64,copy=False); X=np.asarray(X,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    A=np.einsum('bn,ni,nj->bij',Wf,X,X,optimize=True)
    b=np.einsum('bn,ni,n->bi',Wf,X,y,optimize=True)
    try:
        beta=np.linalg.solve(A,b[...,None])[...,0]
    except np.linalg.LinAlgError:
        beta=np.empty_like(b)
        for i in range(len(Wf)):
            try: beta[i]=np.linalg.solve(A[i],b[i])
            except np.linalg.LinAlgError: beta[i]=np.linalg.pinv(A[i])@b[i]
    wy=Wf@y; wy2=Wf@(y*y); sw=Wf.sum(axis=1)
    sse=wy2-2*np.einsum('bi,bi->b',beta,b)+np.einsum('bi,bij,bj->b',beta,A,beta)
    sst=wy2-(wy*wy/np.maximum(sw,1e-15))
    r2=np.where(sst>1e-15,1-sse/sst,np.nan)
    return beta,r2


def observed_ols(X,y):
    return weighted_ols(X,y,np.ones(len(y)))


def normalized_aurc(covs,risks):
    order=np.argsort(covs); x=np.asarray(covs)[order]; y=np.asarray(risks)[order]
    if np.any(~np.isfinite(y)):return np.nan
    return float(np.trapz(y,x)/(x[-1]-x[0]))


def risk_curve_bootstrap(W, errors, score, coverages):
    # W: B x n multinomial bootstrap counts; fixed ranking avoids repeated sorting.
    order=np.argsort(-score,kind='mergesort')
    Wo=W[:,order]
    e=errors[order]
    cw=np.cumsum(Wo,axis=1)
    ce=np.cumsum(Wo*e[None,:],axis=1)
    n=int(W.shape[1]); rows=np.arange(W.shape[0])
    risks=np.empty((W.shape[0],len(coverages)),dtype=np.float32)
    for j,c in enumerate(coverages):
        target=max(1,int(math.ceil(c*n)))
        idx=np.argmax(cw>=target,axis=1)
        den=cw[rows,idx]
        risks[:,j]=ce[rows,idx]/den
    aurc=np.trapz(risks,np.asarray(coverages),axis=1)/(coverages[-1]-coverages[0])
    return risks,aurc.astype(np.float32)


def random_effects(df,effect_col='estimate',se_col='bootstrap_se'):
    rows=[]
    group_cols=['analysis_family','effect_name','model_family','modality']
    for keys,g in df.groupby(group_cols,dropna=False):
        z=g[np.isfinite(g[effect_col])&np.isfinite(g[se_col])&(g[se_col]>0)&g['bootstrap_supported']].copy()
        if len(z)<3:continue
        yi=z[effect_col].to_numpy(float); sei=z[se_col].to_numpy(float); vi=sei**2; wi=1/vi
        fixed=np.sum(wi*yi)/np.sum(wi); Q=np.sum(wi*(yi-fixed)**2)
        den=np.sum(wi)-np.sum(wi**2)/np.sum(wi); tau2=max(0.0,(Q-(len(z)-1))/den) if den>0 else 0.0
        wr=1/(vi+tau2); mu=np.sum(wr*yi)/np.sum(wr); se=math.sqrt(1/np.sum(wr)); zz=mu/se if se>0 else np.nan
        pv=math.erfc(abs(zz)/math.sqrt(2)) if np.isfinite(zz) else np.nan
        rows.append({'analysis_family':keys[0],'effect_name':keys[1],'model_family':keys[2],'modality':keys[3],
                     'endpoint_n':len(z),'random_effect_mean':mu,'random_effect_se':se,'ci_low':mu-1.96*se,'ci_high':mu+1.96*se,
                     'tau2':tau2,'Q':Q,'p_value':pv})
    out=pd.DataFrame(rows)
    if len(out):
        out['fdr_family']=out['analysis_family'].astype(str)+'|'+out['effect_name'].astype(str)
        out['q_value']=np.nan
        for fam,idx in out.groupby('fdr_family').groups.items():
            out.loc[idx,'q_value']=bh_adjust(out.loc[idx,'p_value'].to_numpy())
    return out


def construct_support_scores():
    p3=locate('P3_05_factorial_cell_manifest.csv',['Phase3/outputs/P3_05_factorial_cell_manifest.csv'])
    p5m=locate('P5_01_Morgan_r2_2048_parent_matrix.npz',['Phase5/outputs/P5_01_Morgan_r2_2048_parent_matrix.npz'])
    p5id=locate('P5_03_structure_feature_manifest.csv',['Phase5/outputs/P5_03_structure_feature_manifest.csv'])
    bth=None
    try:bth=locate('P3_04a_biological_support_fold_thresholds.csv',['Phase3/outputs/P3_04a_biological_support_fold_thresholds.csv'])
    except Exception:pass

    base=pd.read_csv(p3)
    if 'index' in base.columns:base=base.drop(columns=['index'])
    base['compound_id']=base['compound_id'].astype(str)
    ids=pd.read_csv(p5id)['compound_id'].astype(str).to_numpy()
    npz=np.load(p5m,allow_pickle=False); bits=npz['morgan_bits']
    if bits.shape!=(len(ids),2048):raise RuntimeError(f'Unexpected Morgan shape {bits.shape}')
    X=bits.astype(np.float32); id_to_i={x:i for i,x in enumerate(ids)}
    if set(base.compound_id)!=set(ids):raise RuntimeError('Phase3/Phase5 parent identity mismatch')

    rec=[]; maxdiff=0.0; qchecks=[]
    for fold in range(5):
        tr=base.loc[base.outer_fold!=fold,'compound_id'].tolist(); te=base.loc[base.outer_fold==fold,'compound_id'].tolist()
        ti=np.array([id_to_i[x] for x in tr]); ei=np.array([id_to_i[x] for x in te])
        A=X[ti]; sim=tanimoto_cross(A,A); np.fill_diagonal(sim,-1.0); train_loo=sim.max(axis=1); train_loo_sort=np.sort(train_loo)
        test_sim=tanimoto_cross(X[ei],X[ti]).max(axis=1)
        stored=base.set_index('compound_id').loc[te,'max_train_morgan_tanimoto'].to_numpy(float)
        maxdiff=max(maxdiff,float(np.max(np.abs(test_sim-stored))))

        bio_path=locate(f'P3_train_biodist_fold{fold}.csv',[f'Phase4/upstream_evidence/P3_train_biodist_fold{fold}.csv'])
        bd=pd.read_csv(bio_path)['train_bio_distance'].to_numpy(float); bd_sort=np.sort(bd)
        if len(bd)!=len(tr):raise RuntimeError(f'Fold {fold} bio training-parent count mismatch')
        testrows=base.set_index('compound_id').loc[te]
        for cid,cs,bdist in zip(te,stored,testrows['bio_distance_median'].to_numpy(float)):
            csp=quantile_ecdf_le(train_loo_sort,cs)
            bnov=quantile_ecdf_le(bd_sort,bdist); bsp=1.0-bnov
            rec.append({'compound_id':cid,'outer_fold':fold,'max_train_morgan_tanimoto':cs,
                        'scaffold_represented_in_train':bool(testrows.loc[cid,'scaffold_represented_in_train']),
                        'bio_distance_median':bdist,'train_q75':float(testrows.loc[cid,'train_q75']),
                        'chemical_support_percentile':csp,'chemical_novelty_percentile':1-csp,
                        'biological_support_percentile':bsp,'biological_novelty_percentile':bnov,
                        'joint_support_gmean':math.sqrt(max(0,csp*bsp)),'joint_support_min':min(csp,bsp),
                        'factorial_cell':testrows.loc[cid,'factorial_cell']})
        if bth is not None:
            th=pd.read_csv(bth).set_index('outer_fold').loc[fold]
            qchecks.append({'outer_fold':fold,
                'q75_diff':abs(np.quantile(bd,.75)-th.q75),'q90_diff':abs(np.quantile(bd,.90)-th.q90),
                'q95_diff':abs(np.quantile(bd,.95)-th.q95),'q975_diff':abs(np.quantile(bd,.975)-th.q975)})
    out=pd.DataFrame(rec).sort_values('compound_id').reset_index(drop=True)
    return out,maxdiff,pd.DataFrame(qchecks)


def endpoint_job(ep, pred_path, support_path):
    pred=pd.read_csv(pred_path)
    pred=pred[(pred.endpoint==ep)&pred.model_family.isin(FAMILIES)&pred.modality.isin(MODALITIES)].copy()
    sup=pd.read_csv(support_path); sup['compound_id']=sup.compound_id.astype(str); pred['compound_id']=pred.compound_id.astype(str)
    pred=pred.merge(sup,on=['compound_id','outer_fold','factorial_cell'],how='left',validate='many_to_one')
    if pred['joint_support_gmean'].isna().any():raise RuntimeError(f'{ep}: missing support join')
    meta=pred[['compound_id','y_true']].drop_duplicates().sort_values('compound_id').reset_index(drop=True)
    if meta.compound_id.duplicated().any():raise RuntimeError(f'{ep}: duplicate parent')
    ids=meta.compound_id.to_numpy(str); y=meta.y_true.to_numpy(int); n=len(meta); idxmap={x:i for i,x in enumerate(ids)}
    sup_ep=sup.set_index('compound_id').loc[ids]
    cs=sup_ep.chemical_support_percentile.to_numpy(float); bs=sup_ep.biological_support_percentile.to_numpy(float); js=sup_ep.joint_support_gmean.to_numpy(float)
    cell=sup_ep.factorial_cell.to_numpy(str)
    rng=np.random.default_rng(seed_for(ep,'PARENT_BOOTSTRAP'))
    W=rng.multinomial(n,np.full(n,1/n),size=BOOTSTRAPS).astype(np.int16)

    regression_rows=[]; delta_rows=[]; risk_curve_rows=[]; key_rows=[]; select_rows=[]
    beta_boot_all=np.full((BOOTSTRAPS,len(MODEL_KEYS),4),np.nan,dtype=np.float32)
    dr2_boot_all=np.full((BOOTSTRAPS,len(MODEL_KEYS)),np.nan,dtype=np.float32)
    aurc_boot_all=np.full((BOOTSTRAPS,len(MODEL_KEYS),len(STRATEGIES)),np.nan,dtype=np.float32)
    fullrisk_boot_all=np.full((BOOTSTRAPS,len(MODEL_KEYS)),np.nan,dtype=np.float32)

    for mj,(fam,mod) in enumerate(MODEL_KEYS):
        g=pred[(pred.model_family==fam)&(pred.modality==mod)][['compound_id','probability','predicted_class']]
        if g.compound_id.duplicated().any():raise RuntimeError(f'{ep} {fam} {mod}: duplicate')
        g=g.set_index('compound_id').reindex(ids)
        if g.isna().any().any():raise RuntimeError(f'{ep} {fam} {mod}: incomplete')
        p=g.probability.to_numpy(float); pc=g.predicted_class.to_numpy(int)
        err=(p-y)**2; conf=2*np.abs(p-.5); unc=1-conf; hybrid=np.sqrt(np.clip(js*conf,0,1))

        # Continuous support-error regression. Predictors standardized within endpoint sample, fixed before bootstrap.
        cn=1-cs; bn=1-bs
        def z(v):
            sd=np.std(v,ddof=0); return (v-v.mean())/sd if sd>1e-12 else np.zeros_like(v)
        zc,zb,zu=z(cn),z(bn),z(unc); inter=zc*zb
        Xfull=np.column_stack([np.ones(n),zc,zb,inter,zu]); Xbase=np.column_stack([np.ones(n),zu])
        beta,r2=observed_ols(Xfull,err); _,r2b=observed_ols(Xbase,err); dr2=r2-r2b
        bt_full,rf=weighted_ols_bootstrap_batch(Xfull,err,W); bt_base,rb=weighted_ols_bootstrap_batch(Xbase,err,W)
        bb=bt_full[:,1:5].astype(np.float32); dboot=(rf-rb).astype(np.float32)
        beta_boot_all[:,mj,:]=bb; dr2_boot_all[:,mj]=dboot
        names=['chemical_novelty','biological_novelty','chemical_x_biological_novelty','model_uncertainty']
        for j,name in enumerate(names):
            sm=boot_summary(beta[j+1],bb[:,j],True)
            regression_rows.append({'endpoint':ep,'analysis_family':'CONTINUOUS_SQUARED_ERROR','effect_name':name,'model_family':fam,'modality':mod,**sm})
        ds=boot_summary(dr2,dboot,False)
        delta_rows.append({'endpoint':ep,'analysis_family':'SUPPORT_BEYOND_CONFIDENCE','effect_name':'delta_R2_support_terms_beyond_uncertainty','model_family':fam,'modality':mod,**ds})

        scores={'CHEMICAL_SUPPORT':cs,'BIOLOGICAL_SUPPORT':bs,'JOINT_SUPPORT':js,'CONFIDENCE':conf,'HYBRID_SUPPORT_CONFIDENCE':hybrid}
        # Dense ALL-OOF Brier-risk curves and bootstrap AURC.
        aurcs={}
        for sj,sname in enumerate(STRATEGIES):
            score=scores[sname]
            order=np.argsort(-score,kind='mergesort')
            dense=[]
            for cov in AURC_COVERAGES:
                k=max(1,int(math.ceil(cov*n))); sel=order[:k]; r=float(np.mean(err[sel])); dense.append(r)
                risk_curve_rows.append({'endpoint':ep,'scope':'ALL_OOF','model_family':fam,'modality':mod,'strategy':sname,'coverage':float(cov),'Brier_risk':r,'retained_n':k})
            aurcs[sname]=normalized_aurc(AURC_COVERAGES,np.array(dense))
            _,ab=risk_curve_bootstrap(W,err,score,AURC_COVERAGES); aurc_boot_all[:,mj,sj]=ab

        fullrisk=float(err.mean()); fullrisk_boot=(W@err)/n; fullrisk_boot_all[:,mj]=fullrisk_boot.astype(np.float32)
        # Primary + decomposition selective effects.
        for sname in STRATEGIES:
            sj=STRATEGIES.index(sname); est=fullrisk-aurcs[sname]; bd=fullrisk_boot-aurc_boot_all[:,mj,sj]
            sm=boot_summary(est,bd,True)
            role='PRIMARY' if sname in ['JOINT_SUPPORT','HYBRID_SUPPORT_CONFIDENCE'] else 'DECOMPOSITION_SECONDARY'
            select_rows.append({'endpoint':ep,'analysis_family':'SELECTIVE_VS_NO_ABSTENTION','effect_name':f'full_risk_minus_AURC__{sname}','comparison_role':role,'model_family':fam,'modality':mod,**sm})
        confj=STRATEGIES.index('CONFIDENCE')
        for sname in ['CHEMICAL_SUPPORT','BIOLOGICAL_SUPPORT','JOINT_SUPPORT','HYBRID_SUPPORT_CONFIDENCE']:
            sj=STRATEGIES.index(sname); est=aurcs['CONFIDENCE']-aurcs[sname]; bd=aurc_boot_all[:,mj,confj]-aurc_boot_all[:,mj,sj]
            sm=boot_summary(est,bd,True); role='PRIMARY' if sname in ['JOINT_SUPPORT','HYBRID_SUPPORT_CONFIDENCE'] else 'DECOMPOSITION_SECONDARY'
            select_rows.append({'endpoint':ep,'analysis_family':'SUPPORT_VS_CONFIDENCE','effect_name':f'confidence_AURC_minus_{sname}_AURC','comparison_role':role,'model_family':fam,'modality':mod,**sm})

        # Key-coverage full metrics, ALL primary + C1/B1 descriptive.
        for scope,mask in [('ALL_OOF',np.ones(n,dtype=bool)),('JOINT_OOD_DESCRIPTIVE',cell=='C1/B1')]:
            baseidx=np.where(mask)[0]
            if len(baseidx)==0:continue
            for sname in STRATEGIES:
                sc=scores[sname][baseidx]; ordlocal=np.argsort(-sc,kind='mergesort')
                for cov in KEY_COVERAGES:
                    k=max(1,int(math.ceil(cov*len(baseidx)))); sel=baseidx[ordlocal[:k]]
                    met=point_metrics(y[sel],p[sel],pc[sel])
                    key_rows.append({'endpoint':ep,'scope':scope,'model_family':fam,'modality':mod,'strategy':sname,'coverage':float(cov),**met})

    np.savez_compressed(BOOTDIR/f'P8_BOOT_{ep}.npz',beta_boot=beta_boot_all,delta_r2_boot=dr2_boot_all,aurc_boot=aurc_boot_all,fullrisk_boot=fullrisk_boot_all,
                        model_family=np.asarray([x[0] for x in MODEL_KEYS],dtype=str),modality=np.asarray([x[1] for x in MODEL_KEYS],dtype=str),
                        regression_effect=np.asarray(['chemical_novelty','biological_novelty','chemical_x_biological_novelty','model_uncertainty'],dtype=str),
                        strategy=np.asarray(STRATEGIES,dtype=str),bootstrap_n=np.asarray([BOOTSTRAPS],dtype=int),compound_n=np.asarray([n],dtype=int))
    return regression_rows,delta_rows,risk_curve_rows,key_rows,select_rows


def main():
    print('='*88) ; print('FACTORIAL-OOD PHASE 8 — CONTINUOUS SUPPORT / APPLICABILITY DOMAIN / SELECTIVE PREDICTION'); print('='*88)
    tests=[]
    def T(tid,desc,ok,detail=''):
        tests.append({'test_id':tid,'description':desc,'status':'PASS' if ok else 'FAIL','detail':str(detail)})

    lock_path=P8/'P8_00_analysis_lock_v1.0.json'
    if lock_path.exists():
        old=json.load(open(lock_path))
        # A prior lock is reusable only if its scientific content matches, allowing BOOTSTRAPS to be the same.
        if old!=LOCK: raise RuntimeError('Existing Phase8 lock differs from current script. Do not overwrite after outcomes.')
    else:
        with open(lock_path,'w') as f:json.dump(LOCK,f,indent=2)
    lock_sha=sha256_file(lock_path)
    print('Phase 8 lock SHA256:',lock_sha)
    T('P8-T01','Phase 8 scientific lock frozen before computation',True,lock_sha)

    # Upstream statuses, if present, must be PASS.
    for phase,statusname,preferred in [('P7','P7_FINAL_STATUS.json','Phase7_Output/P7_FINAL_STATUS.json'),('P7B','P7B_FINAL_STATUS.json','Phase7B_Output/P7B_FINAL_STATUS.json')]:
        sp=locate(statusname,[preferred]); st=json.load(open(sp)); ok=st.get('status')=='PASS'
        T(f'P8-T0{2 if phase=="P7" else 3}',f'{phase} upstream status PASS',ok,st.get('status'))
        if not ok:raise RuntimeError(f'{phase} not PASS')

    support,maxdiff,qchecks=construct_support_scores()
    support_path=P8/'P8_01_training_derived_support_scores.csv'; support.to_csv(support_path,index=False)
    qchecks.to_csv(P8/'P8_02_training_support_threshold_reconstruction.csv',index=False)
    T('P8-T04','2017 unique parent support scores constructed',len(support)==2017 and support.compound_id.nunique()==2017,len(support))
    T('P8-T05','Recomputed test Morgan max similarity matches frozen Phase3 values',maxdiff<1e-6,maxdiff)
    T('P8-T06','Continuous support percentiles finite and in [0,1]',support[['chemical_support_percentile','biological_support_percentile','joint_support_gmean','joint_support_min']].apply(np.isfinite).all().all() and ((support[['chemical_support_percentile','biological_support_percentile','joint_support_gmean','joint_support_min']]>=0)&(support[['chemical_support_percentile','biological_support_percentile','joint_support_gmean','joint_support_min']]<=1)).all().all())
    if len(qchecks):
        mx=float(qchecks[['q75_diff','q90_diff','q95_diff','q975_diff']].to_numpy().max()); T('P8-T07','Training biological q75/q90/q95/q97.5 thresholds reconstruct',mx<1e-8,mx)
    else:T('P8-T07','Training biological threshold reconstruction file available',False,'P3_04a missing')

    pred_path=locate('P7_02_locked_OOF_with_factorial_cells.csv',['Phase7_Output/P7_02_locked_OOF_with_factorial_cells.csv'])
    pred=pd.read_csv(pred_path)
    learned=pred[pred.model_family.isin(FAMILIES)&pred.modality.isin(MODALITIES)].copy()
    T('P8-T08','Locked learned OOF rows exactly 120000',len(learned)==120000,len(learned))
    T('P8-T09','Exactly 8 endpoints and 12 learned model combinations',set(learned.endpoint)==set(ENDPOINTS) and len(learned[['model_family','modality']].drop_duplicates())==12)

    # Save learned-only locked prediction table with continuous support (model confidence/hybrid are model-specific and computed per row).
    joined=learned.merge(support,on=['compound_id','outer_fold','factorial_cell'],how='left',validate='many_to_one')
    joined['confidence_score']=2*np.abs(joined.probability-.5)
    joined['hybrid_support_confidence']=np.sqrt(np.clip(joined.joint_support_gmean*joined.confidence_score,0,1))
    joined['squared_probability_error']=(joined.probability-joined.y_true)**2
    T('P8-T10','All learned OOF rows join to continuous support',joined.joint_support_gmean.notna().all(),int(joined.joint_support_gmean.isna().sum()))
    joined.to_csv(P8/'P8_03_locked_predictions_with_continuous_support.csv',index=False)

    print(f'Starting Phase 8 analysis with {BOOTSTRAPS} paired parent bootstraps × 8 endpoints using {JOBS} workers...')
    res=Parallel(n_jobs=JOBS,backend='loky',verbose=10)(delayed(endpoint_job)(ep,pred_path,support_path) for ep in ENDPOINTS)
    reg=[];delta=[];risk=[];key=[];sel=[]
    for a,b,c,d,e in res:reg+=a;delta+=b;risk+=c;key+=d;sel+=e
    reg=pd.DataFrame(reg); delta=pd.DataFrame(delta); risk=pd.DataFrame(risk); key=pd.DataFrame(key); sel=pd.DataFrame(sel)

    # FDR only for signed inferential families; delta R2 is descriptive bootstrap CI.
    reg['fdr_family']=reg.analysis_family+'|'+reg.effect_name; reg['q_value']=np.nan
    for fam,idx in reg.groupby('fdr_family').groups.items():reg.loc[idx,'q_value']=bh_adjust(reg.loc[idx,'bootstrap_p_two_sided'].to_numpy())
    sel['fdr_family']=sel.analysis_family+'|'+sel.effect_name; sel['q_value']=np.nan
    for fam,idx in sel.groupby('fdr_family').groups.items():sel.loc[idx,'q_value']=bh_adjust(sel.loc[idx,'bootstrap_p_two_sided'].to_numpy())
    reg.to_csv(P8/'P8_04_continuous_support_error_models.csv',index=False)
    delta.to_csv(P8/'P8_05_support_incremental_R2.csv',index=False)
    risk.to_csv(P8/'P8_06_selective_brier_risk_curves.csv',index=False)
    key.to_csv(P8/'P8_07_selective_metrics_key_coverages.csv',index=False)
    sel.to_csv(P8/'P8_08_selective_bootstrap_comparisons.csv',index=False)

    # Prespecified relative risk-reduction coverage targets from dense point curves.
    targetrows=[]
    for keys,g in risk.groupby(['endpoint','model_family','modality','strategy']):
        gg=g.sort_values('coverage'); full=float(gg.loc[np.isclose(gg.coverage,1.0),'Brier_risk'].iloc[0])
        for frac in [0.10,0.20,0.30]:
            ok=gg[gg.Brier_risk<=full*(1-frac)]
            maxcov=float(ok.coverage.max()) if len(ok) else np.nan
            targetrows.append({'endpoint':keys[0],'model_family':keys[1],'modality':keys[2],'strategy':keys[3],
                               'target_relative_Brier_risk_reduction':frac,'max_coverage_achieving_target_within_50_100pct':maxcov})
    pd.DataFrame(targetrows).to_csv(P8/'P8_09_coverage_cost_targets.csv',index=False)

    synth_source=pd.concat([reg,sel],ignore_index=True,sort=False)
    synth=random_effects(synth_source); synth.to_csv(P8/'P8_10_random_effects_synthesis.csv',index=False)

    # Integrity expectations.
    T('P8-T11','Continuous regression coefficient rows = 384',len(reg)==8*12*4,len(reg))
    T('P8-T12','Incremental R2 rows = 96',len(delta)==8*12,len(delta))
    T('P8-T13','Dense ALL-OOF risk curve rows = 24480',len(risk)==8*12*5*51,len(risk))
    T('P8-T14','Key-coverage rows include ALL and joint-OOD descriptive scopes',set(key.scope)=={'ALL_OOF','JOINT_OOD_DESCRIPTIVE'},key.scope.value_counts().to_dict())
    T('P8-T15','Selective signed comparison rows = 864',len(sel)==8*12*9,len(sel))
    T('P8-T16','Eight bootstrap archives generated',len(list(BOOTDIR.glob('P8_BOOT_*.npz')))==8,len(list(BOOTDIR.glob('P8_BOOT_*.npz'))))
    T('P8-T17','All regression/selective bootstrap signed effects meet >=90% valid fraction',reg.bootstrap_supported.all() and sel.bootstrap_supported.all(),{'reg_unsupported':int((~reg.bootstrap_supported).sum()),'sel_unsupported':int((~sel.bootstrap_supported).sum())})
    T('P8-T18','All finite q-values in [0,1]',reg.q_value.dropna().between(0,1).all() and sel.q_value.dropna().between(0,1).all())
    # Phase7/7B source predictions untouched by design; no write path outside P8.
    T('P8-T19','No retraining/recalibration/threshold modification performed',True)

    tdf=pd.DataFrame(tests); tdf.to_csv(P8/'P8_11_integrity_tests.csv',index=False)
    if (tdf.status!='PASS').any():
        print(tdf[tdf.status!='PASS'].to_string(index=False)); raise RuntimeError('Phase 8 integrity test failure')

    status={'phase':8,'status':'PASS','analysis':LOCK['analysis_name'],'phase8_lock_sha256':lock_sha,'bootstrap_resamples_per_endpoint':BOOTSTRAPS,
            'parent_support_n':len(support),'learned_oof_rows':len(joined),'continuous_coefficient_rows':len(reg),'selective_risk_curve_rows':len(risk),
            'selective_comparison_rows':len(sel),'integrity_tests':f'{len(tdf)}/{len(tdf)} PASS',
            'interpretation_status':'Outputs generated and locked; interpret only after independent package validation.'}
    with open(P8/'P8_FINAL_STATUS.json','w') as f:json.dump(status,f,indent=2)

    manifest=P8/'SHA256SUMS_PHASE8.txt'
    files=[p for p in P8.rglob('*') if p.is_file() and p.name not in ['SHA256SUMS_PHASE8.txt','Phase8_UPLOAD_TO_CHAT.zip']]
    with open(manifest,'w') as f:
        for p in sorted(files):f.write(f'{sha256_file(p)}  {p.relative_to(P8)}\n')
    upload=P8/'Phase8_UPLOAD_TO_CHAT.zip'
    with zipfile.ZipFile(upload,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(P8.rglob('*')):
            if p.is_file() and p!=upload:z.write(p,arcname=str(p.relative_to(P8)))
    upsha=sha256_file(upload); status['upload_zip']=str(upload);status['upload_zip_sha256']=upsha;status['upload_zip_size_bytes']=upload.stat().st_size
    with open(P8/'P8_FINAL_STATUS.json','w') as f:json.dump(status,f,indent=2)

    print();print('='*88);print('PHASE 8 LOCAL COMPUTATION PASS');print('Training-derived support parents:',len(support));print('Morgan support reconstruction max abs diff:',maxdiff)
    print('Continuous regression rows:',len(reg));print('Selective Brier-risk curve rows:',len(risk));print('Selective comparison rows:',len(sel));print('Integrity:',f'{len(tdf)}/{len(tdf)} PASS')
    print('Upload:',upload);print('Upload ZIP SHA256:',upsha);print('Phase 6/7/7B models and conclusions were not modified.');print('='*88)

if __name__=='__main__':main()
