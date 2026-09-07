#!/usr/bin/env python3

import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('NUMEXPR_NUM_THREADS','1')

from pathlib import Path
import hashlib, json, math, zipfile, warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

ROOT = Path('/home/lenovo/Documents/Kiran Gaikwad/Pharmacodia/Chem')
P7 = ROOT / 'Phase7_Output'
OUT = ROOT / 'Phase7B_Output'
BOOT = OUT / 'bootstrap'
OUT.mkdir(exist_ok=True)
BOOT.mkdir(exist_ok=True)

B = 2000
JOBS = int(os.environ.get('P7B_JOBS','8'))

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
CELLS = ['C0/B0','C1/B0','C0/B1','C1/B1']
CI = {x:i for i,x in enumerate(CELLS)}
MODEL_KEYS = [(f,m) for f in FAMILIES for m in MODALITIES]

# Prespecified secondary metrics frozen in Protocol v0.2 / Phase 6.
PRESPEC = [
    'AUROC','balanced_accuracy','sensitivity','specificity',
    'calibration_intercept','calibration_slope'
]
# Added sensitivity metrics are explicitly post-Phase-7 explanatory analyses.
ADDED = [
    'balanced_MCC','Brier_skill_score',
    'abs_calibration_intercept','calibration_slope_abs_error'
]
METRICS = PRESPEC + ADDED
MI = {m:i for i,m in enumerate(METRICS)}

# Metrics for which conventional pooled-vs-joint and fusion robustness are meaningful.
COMPARATIVE_METRICS = [
    'AUROC','balanced_accuracy','sensitivity','specificity',
    'balanced_MCC','Brier_skill_score',
    'abs_calibration_intercept','calibration_slope_abs_error'
]

DIRECTION = {
    'AUROC':'higher_better',
    'balanced_accuracy':'higher_better',
    'sensitivity':'higher_better',
    'specificity':'higher_better',
    'calibration_intercept':'ideal_zero_raw_shift',
    'calibration_slope':'ideal_one_raw_shift',
    'balanced_MCC':'higher_better',
    'Brier_skill_score':'higher_better',
    'abs_calibration_intercept':'lower_better',
    'calibration_slope_abs_error':'lower_better',
}

LOCK = {
    'project':'FACTORIAL-OOD',
    'phase':'7B',
    'name':'Prevalence-Aware Secondary and Calibration Completion',
    'created_before_phase7b_results': True,
    'does_not_replace_phase7_coprimary_analysis': True,
    'source_predictions':'Phase7_Output/P7_02_locked_OOF_with_factorial_cells.csv',
    'source_status':'Phase7_Output/P7_FINAL_STATUS.json must be PASS',
    'factorial_cells':CELLS,
    'parent_bootstrap_resamples':B,
    'bootstrap_pairing':'same parent resample used across all 12 learned models within an endpoint',
    'bootstrap_draws':'exact Phase-7 parent bootstrap seed namespace and sampling rule reused',
    'prespecified_secondary_metrics':PRESPEC,
    'added_prevalence_aware_sensitivity_metrics':{
        'balanced_MCC':'MCC calibrated to prevalence 0.5 using sensitivity and specificity; Guesne et al., Journal of Cheminformatics 2024, DOI 10.1186/s13321-024-00837-w',
        'Brier_skill_score':'1 - model Brier / cell-prevalence constant-predictor Brier',
        'abs_calibration_intercept':'absolute deviation from ideal intercept 0',
        'calibration_slope_abs_error':'absolute deviation from ideal slope 1',
    },
    'multiplicity':'Benjamini-Hochberg FDR within analysis_family × metric × contrast',
    'support_rule':'bootstrap metric is confirmatorily interpretable only if >=90% of 2000 resamples are estimable',
    'governance':'No model, endpoint, fold, threshold, modality, support cutoff, or Phase-7 primary conclusion may be changed based on Phase-7B results.',
}

TESTS=[]
def test(tid,desc,ok,detail=''):
    TESTS.append({'test_id':tid,'description':desc,'status':'PASS' if bool(ok) else 'FAIL','detail':str(detail)})

def sha256(path, chunk=8*1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def seed_for(*parts):
    # Deliberately identical to Phase 7 to reconstruct the exact same parent draws.
    txt='|'.join(map(str,['FACTORIAL-OOD','P7','v1']+list(parts)))
    return int(hashlib.sha256(txt.encode()).hexdigest()[:8],16)

def bh(p):
    p=np.asarray(p,float); q=np.full(len(p),np.nan)
    idx=np.where(np.isfinite(p))[0]
    if not len(idx): return q
    pv=p[idx]; order=np.argsort(pv); r=pv[order]; m=len(r)
    a=r*m/np.arange(1,m+1)
    a=np.minimum.accumulate(a[::-1])[::-1]
    a=np.minimum(a,1.0)
    q[idx[order]]=a
    return q

def boot_summary(est, vals):
    vals=np.asarray(vals,float); v=vals[np.isfinite(vals)]
    frac=len(v)/len(vals)
    if not np.isfinite(est) or not len(v):
        return dict(estimate=est,ci_low=np.nan,ci_high=np.nan,bootstrap_se=np.nan,
                    bootstrap_p_two_sided=np.nan,bootstrap_valid_n=len(v),
                    bootstrap_valid_fraction=frac,bootstrap_supported=False)
    lo,hi=np.quantile(v,[.025,.975])
    se=float(np.std(v,ddof=1)) if len(v)>1 else np.nan
    dev=v-est
    p=(1+np.sum(np.abs(dev)>=abs(est)))/(len(v)+1)
    return dict(estimate=float(est),ci_low=float(lo),ci_high=float(hi),bootstrap_se=se,
                bootstrap_p_two_sided=float(min(1,p)),bootstrap_valid_n=len(v),
                bootstrap_valid_fraction=float(frac),bootstrap_supported=bool(frac>=.90))

def balanced_mcc(sen,spe):
    sen=np.asarray(sen,float); spe=np.asarray(spe,float)
    den2=1.0-(sen-spe)**2
    out=np.full(np.broadcast(sen,spe).shape,np.nan,float)
    ok=np.isfinite(sen)&np.isfinite(spe)&(den2>1e-14)
    out[ok]=(sen[ok]+spe[ok]-1.0)/np.sqrt(den2[ok])
    return out

def basic_metrics_multi(y,P,Q):
    """Vectorized threshold/discrimination metrics across models. P/Q: n x J."""
    y=np.asarray(y,int)
    P=np.asarray(P,float); Q=np.asarray(Q,int)
    J=P.shape[1]
    if len(y)==0 or len(np.unique(y))<2:
        return {k:np.full(J,np.nan) for k in ['AUROC','balanced_accuracy','sensitivity','specificity','balanced_MCC','Brier_skill_score']}
    yy=y[:,None]
    tp=np.sum((Q==1)&(yy==1),axis=0).astype(float)
    fn=np.sum((Q==0)&(yy==1),axis=0).astype(float)
    tn=np.sum((Q==0)&(yy==0),axis=0).astype(float)
    fp=np.sum((Q==1)&(yy==0),axis=0).astype(float)
    sen=np.divide(tp,tp+fn,out=np.full(J,np.nan),where=(tp+fn)>0)
    spe=np.divide(tn,tn+fp,out=np.full(J,np.nan),where=(tn+fp)>0)
    ba=(sen+spe)/2.0
    bm=balanced_mcc(sen,spe)
    auc=np.full(J,np.nan)
    for j in range(J):
        try: auc[j]=roc_auc_score(y,P[:,j])
        except Exception: pass
    bs=np.mean((P-yy)**2,axis=0)
    prev=float(np.mean(y)); ref=prev*(1-prev)
    bss=np.full(J,np.nan) if ref<=0 else 1.0-bs/ref
    return dict(AUROC=auc,balanced_accuracy=ba,sensitivity=sen,specificity=spe,balanced_MCC=bm,Brier_skill_score=bss)

def calibration_multi(y,P,max_iter=30,tol=1e-8):
    """Unpenalized logistic recalibration y ~ intercept + slope*logit(p), vectorized over models."""
    y=np.asarray(y,int); P=np.asarray(P,float)
    J=P.shape[1]
    if len(y)==0 or len(np.unique(y))<2:
        return np.full(J,np.nan),np.full(J,np.nan)
    Pc=np.clip(P,1e-6,1-1e-6)
    X=np.log(Pc/(1.0-Pc))
    if np.any(~np.isfinite(X)):
        return np.full(J,np.nan),np.full(J,np.nan)
    a=np.zeros(J,float); b=np.ones(J,float); good=np.ones(J,bool)
    for _ in range(max_iter):
        eta=np.clip(a[None,:]+X*b[None,:],-35,35)
        mu=1.0/(1.0+np.exp(-eta)); w=np.maximum(mu*(1-mu),1e-12)
        r=y[:,None]-mu
        g0=np.sum(r,axis=0); g1=np.sum(r*X,axis=0)
        h00=np.sum(w,axis=0); h01=np.sum(w*X,axis=0); h11=np.sum(w*X*X,axis=0)
        det=h00*h11-h01*h01
        ok=np.isfinite(det)&(np.abs(det)>1e-12)&good
        if not np.any(ok): break
        da=np.zeros(J); db=np.zeros(J)
        da[ok]=(g0[ok]*h11[ok]-g1[ok]*h01[ok])/det[ok]
        db[ok]=(g1[ok]*h00[ok]-g0[ok]*h01[ok])/det[ok]
        a[ok]+=da[ok]; b[ok]+=db[ok]
        bad=(~np.isfinite(a))|(~np.isfinite(b))|(np.abs(a)>50)|(np.abs(b)>50)
        good[bad]=False
        if np.nanmax(np.abs(np.r_[da[ok],db[ok]]))<tol: break
    a[~good]=np.nan; b[~good]=np.nan
    return a,b

def all_metrics_multi(y,P,Q):
    base=basic_metrics_multi(y,P,Q)
    a,b=calibration_multi(y,P)
    base['calibration_intercept']=a
    base['calibration_slope']=b
    base['abs_calibration_intercept']=np.abs(a)
    base['calibration_slope_abs_error']=np.abs(b-1.0)
    return np.column_stack([base[m] for m in METRICS])

def endpoint_job(ep, source):
    df=pd.read_csv(source,usecols=['endpoint','compound_id','y_true','model_family','modality','probability','predicted_class','factorial_cell'])
    df=df[(df.endpoint==ep)&df.model_family.isin(FAMILIES)&df.modality.isin(MODALITIES)].copy()
    meta=df[['compound_id','y_true','factorial_cell']].drop_duplicates().sort_values('compound_id').reset_index(drop=True)
    if meta.compound_id.duplicated().any(): raise RuntimeError(f'{ep}: duplicate compound')
    ids=meta.compound_id.astype(str).to_numpy(); y=meta.y_true.astype(int).to_numpy(); cc=meta.factorial_cell.map(CI).astype(int).to_numpy()
    n=len(meta); J=len(MODEL_KEYS); K=len(METRICS)
    P=np.empty((n,J),np.float32); Q=np.empty((n,J),np.int8)
    for j,(fam,mod) in enumerate(MODEL_KEYS):
        g=df[(df.model_family==fam)&(df.modality==mod)][['compound_id','probability','predicted_class']]
        if g.compound_id.duplicated().any(): raise RuntimeError(f'{ep} {fam} {mod}: duplicate')
        g=g.set_index('compound_id').reindex(ids)
        if g.isna().any().any(): raise RuntimeError(f'{ep} {fam} {mod}: alignment failure')
        P[:,j]=g.probability.to_numpy(np.float32); Q[:,j]=g.predicted_class.to_numpy(np.int8)

    obs=np.full((J,4,K),np.nan,float)
    for c in range(4):
        m=cc==c
        obs[:,c,:]=all_metrics_multi(y[m],P[m,:],Q[m,:])
    pool_obs=all_metrics_multi(y,P,Q)  # J x K

    boot=np.full((B,J,4,K),np.nan,np.float32)
    pool=np.full((B,J,K),np.nan,np.float32)
    rng=np.random.default_rng(seed_for(ep,'PARENT_CLUSTER_BOOTSTRAP'))
    for bi in range(B):
        idx=rng.integers(0,n,size=n)
        yb=y[idx]; Pb=P[idx,:]; Qb=Q[idx,:]; cb=cc[idx]
        pool[bi,:,:]=all_metrics_multi(yb,Pb,Qb)
        for c in range(4):
            m=cb==c
            if np.any(m): boot[bi,:,c,:]=all_metrics_multi(yb[m],Pb[m,:],Qb[m,:])

    np.savez_compressed(BOOT/f'P7B_BOOT_{ep}.npz',cell_boot=boot,pool_boot=pool,
                        model_family=np.asarray([x[0] for x in MODEL_KEYS],str),
                        modality=np.asarray([x[1] for x in MODEL_KEYS],str),cells=np.asarray(CELLS,str),metrics=np.asarray(METRICS,str))

    cell_rows=[]; contrast_rows=[]
    # cell summaries
    for j,(fam,mod) in enumerate(MODEL_KEYS):
        for c,cell in enumerate(CELLS):
            m=cc==c
            row={'endpoint':ep,'model_family':fam,'modality':mod,'factorial_cell':cell,
                 'n':int(m.sum()),'active_n':int(y[m].sum()),'inactive_n':int(np.sum(y[m]==0)),'prevalence':float(np.mean(y[m]))}
            for k,metric in enumerate(METRICS):
                s=boot_summary(obs[j,c,k],boot[:,j,c,k])
                for key,val in s.items(): row[f'{metric}_{key}' if key!='estimate' else metric]=val
            cell_rows.append(row)

    def add(analysis_family,metric,contrast,fam,mod,est,bd,extra=None):
        s=boot_summary(est,bd)
        r={'endpoint':ep,'analysis_family':analysis_family,'metric':metric,'metric_role':'PRESPECIFIED_SECONDARY' if metric in PRESPEC else 'ADDED_SENSITIVITY',
           'metric_direction':DIRECTION[metric],'contrast':contrast,'model_family':fam,'modality':mod,**s}
        if extra: r.update(extra)
        contrast_rows.append(r)

    for j,(fam,mod) in enumerate(MODEL_KEYS):
        for metric in METRICS:
            k=MI[metric]
            r,ch,bi,jo=[obs[j,CI[x],k] for x in CELLS]
            br,bch,bbi,bjo=[boot[:,j,CI[x],k] for x in CELLS]
            for name,est,bd in [
                ('chemical_shift',ch-r,bch-br),('biological_shift',bi-r,bbi-br),
                ('joint_vs_reference',jo-r,bjo-br),('joint_vs_chemical',jo-ch,bjo-bch),
                ('joint_vs_biological',jo-bi,bjo-bbi),('factorial_interaction',jo-ch-bi+r,bjo-bch-bbi+br)]:
                add('SECONDARY_FACTORIAL',metric,name,fam,mod,est,bd)
        # pooled-vs-joint optimism / reliability gap
        for metric in COMPARATIVE_METRICS:
            k=MI[metric]; pooled=pool_obs[j,k]; joint=obs[j,CI['C1/B1'],k]
            add('SECONDARY_OPTIMISM',metric,'pooled_OOF_minus_joint_OOD',fam,mod,pooled-joint,pool[:,j,k]-boot[:,j,CI['C1/B1'],k],
                {'pooled_OOF_metric':pooled,'joint_OOD_metric':joint})

    # secondary fusion robustness
    midx={k:i for i,k in enumerate(MODEL_KEYS)}
    for fam in FAMILIES:
        js=midx[(fam,'STRUCTURE')]
        for fm in ['EARLY_FUSION','LATE_FUSION']:
            jf=midx[(fam,fm)]
            for metric in COMPARATIVE_METRICS:
                k=MI[metric]
                sref=obs[js,CI['C0/B0'],k]; sj=obs[js,CI['C1/B1'],k]
                fref=obs[jf,CI['C0/B0'],k]; fj=obs[jf,CI['C1/B1'],k]
                est=(fj-sj)-(fref-sref)
                bd=(boot[:,jf,CI['C1/B1'],k]-boot[:,js,CI['C1/B1'],k])-(boot[:,jf,CI['C0/B0'],k]-boot[:,js,CI['C0/B0'],k])
                add('SECONDARY_FUSION_ROBUSTNESS',metric,'joint_minus_reference_fusion_advantage',fam,fm,est,bd,
                    {'reference_fusion_advantage':fref-sref,'joint_fusion_advantage':fj-sj})

    return cell_rows,contrast_rows

def random_effects(df):
    rows=[]
    keys=['analysis_family','metric','contrast','model_family','modality']
    for kk,g in df.groupby(keys,dropna=False):
        z=g[g.bootstrap_supported & np.isfinite(g.estimate) & np.isfinite(g.bootstrap_se) & (g.bootstrap_se>0)]
        if len(z)<3: continue
        yi=z.estimate.to_numpy(float); vi=z.bootstrap_se.to_numpy(float)**2; wi=1/vi
        fixed=np.sum(wi*yi)/np.sum(wi); Q=np.sum(wi*(yi-fixed)**2)
        den=np.sum(wi)-np.sum(wi**2)/np.sum(wi); tau=max(0,(Q-(len(z)-1))/den) if den>0 else 0
        wr=1/(vi+tau); mu=np.sum(wr*yi)/np.sum(wr); se=math.sqrt(1/np.sum(wr)); zz=mu/se if se>0 else np.nan
        rows.append({'analysis_family':kk[0],'metric':kk[1],'contrast':kk[2],'model_family':kk[3],'modality':kk[4],
                     'endpoint_n':len(z),'random_effect_mean':mu,'random_effect_se':se,'ci_low':mu-1.96*se,'ci_high':mu+1.96*se,
                     'tau2':tau,'Q':Q,'p_value':2*norm.sf(abs(zz)) if np.isfinite(zz) else np.nan})
    out=pd.DataFrame(rows)
    if len(out):
        out['fdr_family']=out.analysis_family.astype(str)+'|'+out.metric.astype(str)+'|'+out.contrast.astype(str)
        out['q_value']=np.nan
        for fam,idx in out.groupby('fdr_family').groups.items(): out.loc[idx,'q_value']=bh(out.loc[idx,'p_value'])
    return out

def main():
    print('='*80); print('FACTORIAL-OOD PHASE 7B — PREVALENCE-AWARE SECONDARY/CALIBRATION'); print('='*80)
    lock_path=OUT/'P7B_00_analysis_lock_v1.0.json'
    if lock_path.exists():
        existing=json.load(open(lock_path))
        if existing!=LOCK: raise RuntimeError('Existing Phase 7B lock differs from current script. Stop.')
    else:
        json.dump(LOCK,open(lock_path,'w'),indent=2)
    locksha=sha256(lock_path); print('Phase 7B lock SHA256:',locksha)

    stat=P7/'P7_FINAL_STATUS.json'; src=P7/'P7_02_locked_OOF_with_factorial_cells.csv'
    if not stat.exists() or not src.exists(): raise RuntimeError('Missing validated Phase 7 inputs')
    s=json.load(open(stat)); test('P7B-T01','Phase 7 status PASS',s.get('status')=='PASS',s.get('status'))
    test('P7B-T02','Phase 7 source OOF exists',src.exists(),src)
    d=pd.read_csv(src,usecols=['endpoint','compound_id','model_family','modality','factorial_cell'])
    test('P7B-T03','Source OOF has 150,000 rows',len(d)==150000,len(d))
    l=d[d.model_family.isin(FAMILIES)&d.modality.isin(MODALITIES)]
    test('P7B-T04','Learned source has 120,000 rows',len(l)==120000,len(l))
    test('P7B-T05','Exactly 12 learned model combinations',l[['model_family','modality']].drop_duplicates().shape[0]==12,l[['model_family','modality']].drop_duplicates().shape[0])
    test('P7B-T06','Exactly four factorial cells',set(l.factorial_cell.unique())==set(CELLS),sorted(l.factorial_cell.unique()))

    print(f'Starting {B} paired parent bootstraps × 8 endpoints using {JOBS} workers...')
    res=Parallel(n_jobs=JOBS,backend='loky',verbose=10)(delayed(endpoint_job)(ep,src) for ep in ENDPOINTS)
    cells=[]; contrasts=[]
    for a,b in res: cells.extend(a); contrasts.extend(b)
    cells=pd.DataFrame(cells); contrasts=pd.DataFrame(contrasts)
    test('P7B-T07','Secondary cell table has 384 rows',len(cells)==384,len(cells))
    test('P7B-T08','Eight bootstrap archives generated',len(list(BOOT.glob('P7B_BOOT_*.npz')))==8,len(list(BOOT.glob('P7B_BOOT_*.npz'))))
    # Primary secondary metrics should be well supported; calibration may legitimately be unsupported in sparse bootstraps.
    for metric in ['AUROC','balanced_accuracy','sensitivity','specificity','balanced_MCC','Brier_skill_score']:
        col=f'{metric}_bootstrap_supported'
        test('P7B-SUP-'+metric[:8],f'{metric} has >=90% bootstrap support in all 384 cells',bool(cells[col].all()),int((~cells[col]).sum()))

    # FDR on endpoint-level contrasts.
    contrasts['fdr_family']=contrasts.analysis_family.astype(str)+'|'+contrasts.metric.astype(str)+'|'+contrasts.contrast.astype(str)
    contrasts['q_value']=np.nan
    for fam,idx in contrasts.groupby('fdr_family').groups.items():
        p=contrasts.loc[idx,'bootstrap_p_two_sided'].where(contrasts.loc[idx,'bootstrap_supported'],np.nan).to_numpy(float)
        contrasts.loc[idx,'q_value']=bh(p)
    contrasts['ci_excludes_zero']=(contrasts.ci_low>0)|(contrasts.ci_high<0)
    contrasts['fdr_significant_0_05']=contrasts.q_value<=.05
    test('P7B-T09','All finite q-values in [0,1]',contrasts.q_value.dropna().between(0,1).all())

    cells.to_csv(OUT/'P7B_01_secondary_cell_metrics.csv',index=False)
    contrasts.to_csv(OUT/'P7B_02_secondary_factorial_contrasts.csv',index=False)
    contrasts[contrasts.analysis_family=='SECONDARY_FUSION_ROBUSTNESS'].to_csv(OUT/'P7B_03_secondary_fusion_robustness.csv',index=False)
    contrasts[contrasts.analysis_family=='SECONDARY_OPTIMISM'].to_csv(OUT/'P7B_04_secondary_optimism.csv',index=False)
    random_effects(contrasts).to_csv(OUT/'P7B_05_random_effects_synthesis.csv',index=False)

    # Metric-role registry for manuscript transparency.
    pd.DataFrame([{'metric':m,'role':'PRESPECIFIED_SECONDARY' if m in PRESPEC else 'ADDED_SENSITIVITY','direction':DIRECTION[m]} for m in METRICS]).to_csv(OUT/'P7B_06_metric_role_registry.csv',index=False)

    t=pd.DataFrame(TESTS); t.to_csv(OUT/'P7B_07_integrity_tests.csv',index=False)
    if (t.status!='PASS').any():
        print(t[t.status!='PASS'].to_string(index=False)); raise RuntimeError('Phase 7B integrity/support test failed')
    status={'phase':'7B','status':'PASS','analysis_lock_sha256':locksha,'bootstrap_resamples_per_endpoint':B,
            'prespecified_secondary_metrics':PRESPEC,'added_sensitivity_metrics':ADDED,
            'cell_metric_rows':len(cells),'contrast_rows':len(contrasts),'integrity_tests':f'{len(t)}/{len(t)} PASS',
            'governance':'Phase 7 co-primary results remain unchanged.'}
    json.dump(status,open(OUT/'P7B_FINAL_STATUS.json','w'),indent=2)

    mf=OUT/'SHA256SUMS_PHASE7B.txt'
    with open(mf,'w') as f:
        for p in sorted([x for x in OUT.rglob('*') if x.is_file() and x.name not in ['SHA256SUMS_PHASE7B.txt','Phase7B_UPLOAD_TO_CHAT.zip']]):
            f.write(f'{sha256(p)}  {p.relative_to(OUT)}\n')
    zpath=OUT/'Phase7B_UPLOAD_TO_CHAT.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p!=zpath: z.write(p,arcname=str(p.relative_to(OUT)))
    zsha=sha256(zpath); status['upload_zip_sha256']=zsha; status['upload_zip']=str(zpath); status['upload_zip_size_bytes']=zpath.stat().st_size
    json.dump(status,open(OUT/'P7B_FINAL_STATUS.json','w'),indent=2)
    print(); print('='*80); print('PHASE 7B LOCAL COMPUTATION PASS'); print('Cell metric rows:',len(cells)); print('Contrast rows:',len(contrasts)); print('Integrity:',f'{len(t)}/{len(t)} PASS'); print('Upload:',zpath); print('Upload ZIP SHA256:',zsha); print('Phase 7 co-primary conclusions remain unchanged.'); print('='*80)

if __name__=='__main__': main()
