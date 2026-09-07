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

ROOT=Path('/home/lenovo/Documents/Kiran Gaikwad/Pharmacodia/Chem')
OUT=ROOT/'Phase9_Output'/'P9A'
OUT.mkdir(parents=True,exist_ok=True)
B=2000
JOBS=int(os.environ.get('P9A_JOBS','8'))
LOCK_SHA='55f769210fa70d1be54904e1e59c7a639175811018a8373264fe3f3226027a01'
P7_SOURCE_SHA='ec3e1ba92a0f3f15e764d63ae92f987bf85c79650d242da855e76b1c5364e632'
P3_CHEM_SHA='b6133e7cc6119a3929bd23981bf14b5a1055039aa8ac16d6ec2b24a83a61afe7'
P3_BIO_SHA='0366ce7661795b422f1525e69631fd29f991d5999f6752c28f11a353a18f419c'
P7_PRIMARY_CELL_SHA='a7fa2b41d349c0d023a1513bc25659997e08068ac6ee65ccbaab34238dbd496f'
P7B_CELL_SHA='fe2d758017ffa39eac6d082864dfd0318af20859cd67a22b2992b2d5da72dffa'

ENDPOINTS=[
'TOX21_ERa_BLA_Antagonist_ratio','TOX21_FXR_BLA_Antagonist_ratio','TOX21_MMP_ratio',
'TOX21_PPARg_BLA_Antagonist_ratio','TOX21_RAR_LUC_Antagonist','TOX21_RORg_LUC_CHO_Antagonist',
'TOX21_SBE_BLA_Antagonist_ratio','TOX21_hERG_U2OS_Antagonist']
FAMILIES=['ELASTIC_NET','RANDOM_FOREST','HIST_GRAD_BOOST']
MODALITIES=['STRUCTURE','HTTR','EARLY_FUSION','LATE_FUSION']
MODEL_KEYS=[(f,m) for f in FAMILIES for m in MODALITIES]
CELLS=['C0/B0','C1/B0','C0/B1','C1/B1']; CI={c:i for i,c in enumerate(CELLS)}
METRICS=['AUROC','AUPRC','MCC','balanced_MCC','Brier','Brier_skill_score','ECE_10bin','balanced_accuracy','sensitivity','specificity']
MI={m:i for i,m in enumerate(METRICS)}
CHEM_DEFS=['CHEM_PRIMARY_HYBRID_050','CHEM_SIM_ONLY_050','CHEM_SIM_ONLY_040','CHEM_SIM_ONLY_030','CHEM_SCAFFOLD_ONLY']
BIO_DEFS=['BIO_PRIMARY_Q75_MEDIAN','BIO_Q90_MEDIAN','BIO_Q95_MEDIAN_BUFFERED','BIO_Q975_MEDIAN_BUFFERED','BIO_Q95_WORST_CONTEXT_BUFFERED']
DEF_KEYS=[(c,b) for c in CHEM_DEFS for b in BIO_DEFS]
PRIMARY_DEF=('CHEM_PRIMARY_HYBRID_050','BIO_PRIMARY_Q75_MEDIAN')
TESTS=[]

def T(tid,desc,ok,detail=''):
    TESTS.append({'test_id':tid,'description':desc,'status':'PASS' if bool(ok) else 'FAIL','detail':str(detail)})

def sha256(p,chunk=8*1024*1024):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(chunk),b''): h.update(x)
    return h.hexdigest()

def locate(name,preferred=()):
    for x in preferred:
        p=ROOT/x
        if p.exists(): return p
    hits=[p for p in ROOT.rglob(name) if 'Phase9_Output' not in str(p)]
    if not hits: raise FileNotFoundError(name)
    hits=sorted(hits,key=lambda p:(0 if any(str(p).startswith(str(ROOT/d)) for d in ['Phase3','Phase7_Output','Phase7B_Output']) else 1,len(str(p))))
    return hits[0]

def bh(p):
    p=np.asarray(p,float); q=np.full(len(p),np.nan); idx=np.where(np.isfinite(p))[0]
    if not len(idx): return q
    pv=p[idx]; o=np.argsort(pv); r=pv[o]; m=len(r)
    a=np.minimum.accumulate((r*m/np.arange(1,m+1))[::-1])[::-1]; a=np.minimum(a,1.)
    q[idx[o]]=a; return q

def boot_summary(est,vals):
    vals=np.asarray(vals,float); v=vals[np.isfinite(vals)]; frac=len(v)/len(vals)
    if not np.isfinite(est) or not len(v):
        return dict(estimate=est,ci_low=np.nan,ci_high=np.nan,bootstrap_se=np.nan,bootstrap_p_two_sided=np.nan,
                    bootstrap_valid_n=len(v),bootstrap_valid_fraction=frac,bootstrap_supported=False)
    lo,hi=np.quantile(v,[.025,.975]); se=np.std(v,ddof=1) if len(v)>1 else np.nan
    dev=v-est; p=(1+np.sum(np.abs(dev)>=abs(est)))/(len(v)+1)
    return dict(estimate=float(est),ci_low=float(lo),ci_high=float(hi),bootstrap_se=float(se),
                bootstrap_p_two_sided=float(min(1,p)),bootstrap_valid_n=int(len(v)),bootstrap_valid_fraction=float(frac),
                bootstrap_supported=bool(frac>=.90))

def seed_for(ep):
    # Same namespace/order as Phase 7 and Phase 7B for paired sensitivity draws.
    txt='|'.join(map(str,['FACTORIAL-OOD','P7','v1',ep,'PARENT_CLUSTER_BOOTSTRAP']))
    return int(hashlib.sha256(txt.encode()).hexdigest()[:8],16)

def balanced_mcc_from_counts(tp,fn,tn,fp):
    sen=np.divide(tp,tp+fn,out=np.full_like(tp,np.nan,dtype=float),where=(tp+fn)>0)
    spe=np.divide(tn,tn+fp,out=np.full_like(tn,np.nan,dtype=float),where=(tn+fp)>0)
    den2=1-(sen-spe)**2; out=np.full_like(sen,np.nan,dtype=float)
    ok=np.isfinite(sen)&np.isfinite(spe)&(den2>1e-14)
    out[ok]=(sen[ok]+spe[ok]-1)/np.sqrt(den2[ok]); return out,sen,spe

def weighted_rank_metrics(y,p,W):
    """Weighted AUROC/AP for B bootstrap weight rows. Handles tied scores exactly by score groups."""
    y=np.asarray(y,int); p=np.asarray(p,float); W=np.asarray(W,float)
    Bn=W.shape[0]; auc=np.full(Bn,np.nan); ap=np.full(Bn,np.nan)
    if len(y)==0:return auc,ap
    order=np.argsort(-p,kind='mergesort'); ps=p[order]; ys=y[order]; Ws=W[:,order]
    starts=np.r_[0,1+np.where(ps[1:]!=ps[:-1])[0]]
    pos_g=np.add.reduceat(Ws*ys[None,:],starts,axis=1)
    neg_g=np.add.reduceat(Ws*(1-ys)[None,:],starts,axis=1)
    Ptot=pos_g.sum(axis=1); Ntot=neg_g.sum(axis=1); ok=(Ptot>0)&(Ntot>0)
    if np.any(ok):
        cpos=np.cumsum(pos_g,axis=1); call=np.cumsum(pos_g+neg_g,axis=1)
        prec=np.divide(cpos,call,out=np.zeros_like(cpos),where=call>0)
        ap[ok]=np.sum(prec[ok]*pos_g[ok],axis=1)/Ptot[ok]
        cneg=np.cumsum(neg_g,axis=1)
        neg_below=Ntot[:,None]-cneg
        num=np.sum(pos_g*(neg_below+0.5*neg_g),axis=1)
        auc[ok]=num[ok]/(Ptot[ok]*Ntot[ok])
    return auc,ap

def weighted_ece(y,p,W,nbins=10):
    y=np.asarray(y,float); p=np.asarray(p,float); W=np.asarray(W,float); Bn=W.shape[0]
    tot=W.sum(axis=1); out=np.zeros(Bn,float); good=tot>0
    bins=np.minimum((np.clip(p,0,1)*nbins).astype(int),nbins-1)
    for k in range(nbins):
        idx=np.where(bins==k)[0]
        if not len(idx): continue
        wb=W[:,idx]; cnt=wb.sum(axis=1); m=cnt>0
        if not np.any(m):continue
        ys=wb@y[idx]; ps=wb@p[idx]
        obs=np.divide(ys,cnt,out=np.zeros_like(cnt),where=m); pred=np.divide(ps,cnt,out=np.zeros_like(cnt),where=m)
        out[m]+=cnt[m]/tot[m]*np.abs(obs[m]-pred[m])
    out[~good]=np.nan; return out

def weighted_metrics(y,P,Q,W):
    """Return B x J x K metrics for fixed rows and bootstrap integer weights."""
    y=np.asarray(y,int); P=np.asarray(P,float); Q=np.asarray(Q,int); W=np.asarray(W,float)
    Bn,n=W.shape; J=P.shape[1]; K=len(METRICS); out=np.full((Bn,J,K),np.nan,float)
    tot=W.sum(axis=1); y1=y.astype(float); prev=np.divide(W@y1,tot,out=np.full(Bn,np.nan),where=tot>0)
    yy=y[:,None]
    for j in range(J):
        q=Q[:,j]; p=P[:,j]
        tp=W@((y==1)&(q==1)).astype(float); fn=W@((y==1)&(q==0)).astype(float)
        tn=W@((y==0)&(q==0)).astype(float); fp=W@((y==0)&(q==1)).astype(float)
        bm,sen,spe=balanced_mcc_from_counts(tp,fn,tn,fp); ba=(sen+spe)/2
        den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); mcc=np.divide(tp*tn-fp*fn,den,out=np.full(Bn,np.nan),where=den>0)
        err=(p-y)**2; brier=np.divide(W@err,tot,out=np.full(Bn,np.nan),where=tot>0)
        ref=prev*(1-prev); bss=np.divide(brier,ref,out=np.full(Bn,np.nan),where=ref>0); bss=1-bss
        auc,ap=weighted_rank_metrics(y,p,W); ece=weighted_ece(y,p,W,10)
        vals={'AUROC':auc,'AUPRC':ap,'MCC':mcc,'balanced_MCC':bm,'Brier':brier,'Brier_skill_score':bss,
              'ECE_10bin':ece,'balanced_accuracy':ba,'sensitivity':sen,'specificity':spe}
        for k,m in enumerate(METRICS):out[:,j,k]=vals[m]
    return out

def chem_state(df,name):
    if name=='CHEM_PRIMARY_HYBRID_050': return df.chemical_state.astype(str).to_numpy()
    if name.startswith('CHEM_SIM_ONLY_'):
        t={'CHEM_SIM_ONLY_050':.50,'CHEM_SIM_ONLY_040':.40,'CHEM_SIM_ONLY_030':.30}[name]
        return np.where(df.max_train_morgan_tanimoto.to_numpy(float)>=t,'C0','C1')
    if name=='CHEM_SCAFFOLD_ONLY': return np.where(df.scaffold_represented_in_train.astype(bool).to_numpy(),'C0','C1')
    raise KeyError(name)

def bio_state(df,name):
    col={'BIO_PRIMARY_Q75_MEDIAN':'biological_state','BIO_Q90_MEDIAN':'B_state_amended_q90',
         'BIO_Q95_MEDIAN_BUFFERED':'B_state_primary_q95','BIO_Q975_MEDIAN_BUFFERED':'B_state_q975',
         'BIO_Q95_WORST_CONTEXT_BUFFERED':'B_state_worst_q95'}[name]
    return df[col].astype(str).to_numpy()

def build_assignments(meta,support):
    s=support.set_index('compound_id').reindex(meta.compound_id).reset_index()
    if s.isna().any().any(): raise RuntimeError('Support alignment contains missing values')
    out={}
    for cd in CHEM_DEFS:
        cs=chem_state(s,cd)
        for bd in BIO_DEFS:
            bs=bio_state(s,bd); cc=np.full(len(s),-1,np.int8)
            for i,(c,b) in enumerate(zip(cs,bs)):
                if c in ('C0','C1') and b in ('B0','B1'): cc[i]=CI[f'{c}/{b}']
            out[(cd,bd)]=cc
    return out

def endpoint_job(ep,source,support):
    use=['endpoint','compound_id','y_true','model_family','modality','probability','predicted_class']
    d=pd.read_csv(source,usecols=use); d=d[(d.endpoint==ep)&d.model_family.isin(FAMILIES)&d.modality.isin(MODALITIES)].copy()
    meta=d[['compound_id','y_true']].drop_duplicates().sort_values('compound_id').reset_index(drop=True)
    if meta.compound_id.duplicated().any(): raise RuntimeError(f'{ep}: duplicate parent labels')
    ids=meta.compound_id.astype(str).to_numpy(); y=meta.y_true.astype(int).to_numpy(); n=len(meta); J=len(MODEL_KEYS); K=len(METRICS)
    P=np.empty((n,J),np.float32); Q=np.empty((n,J),np.int8)
    for j,(fam,mod) in enumerate(MODEL_KEYS):
        g=d[(d.model_family==fam)&(d.modality==mod)][['compound_id','probability','predicted_class']]
        if g.compound_id.duplicated().any():raise RuntimeError(f'{ep} {fam}/{mod}: duplicate')
        g=g.set_index('compound_id').reindex(ids)
        if g.isna().any().any():raise RuntimeError(f'{ep} {fam}/{mod}: alignment failure')
        P[:,j]=g.probability.to_numpy(np.float32); Q[:,j]=g.predicted_class.to_numpy(np.int8)
    assignments=build_assignments(meta,support)
    # Exact Phase7 parent draws, represented as integer parent multiplicities.
    rng=np.random.default_rng(seed_for(ep)); W=np.empty((B,n),np.uint16)
    for bi in range(B): W[bi]=np.bincount(rng.integers(0,n,size=n),minlength=n)
    allmask=np.ones(n,bool); pool_obs=weighted_metrics(y,P,Q,np.ones((1,n),np.uint8))[0]
    pool_boot=weighted_metrics(y,P,Q,W)
    cell_rows=[]; contrast_rows=[]; comp_rows=[]
    midx={k:i for i,k in enumerate(MODEL_KEYS)}
    for di,(cd,bd) in enumerate(DEF_KEYS):
        cc=assignments[(cd,bd)]
        obs=np.full((J,4,K),np.nan); boot=np.full((B,J,4,K),np.nan,np.float32)
        counts={}
        for c,cell in enumerate(CELLS):
            m=cc==c; counts[cell]=int(m.sum())
            if np.any(m):
                om=weighted_metrics(y[m],P[m,:],Q[m,:],np.ones((1,int(m.sum())),np.uint8))[0]
                bm=weighted_metrics(y[m],P[m,:],Q[m,:],W[:,m])
                obs[:,c,:]=om; boot[:,:,c,:]=bm
            active=int(y[m].sum()); inactive=int(m.sum()-active); prev=float(active/m.sum()) if m.sum() else np.nan
            for j,(fam,mod) in enumerate(MODEL_KEYS):
                row={'endpoint':ep,'chemical_definition':cd,'biological_definition':bd,'definition_id':f'{cd}__{bd}',
                     'model_family':fam,'modality':mod,'factorial_cell':cell,'n':int(m.sum()),'active_n':active,'inactive_n':inactive,'prevalence':prev}
                for k,metric in enumerate(METRICS):row[metric]=obs[j,c,k]
                cell_rows.append(row)
        comp_rows.append({'endpoint':ep,'chemical_definition':cd,'biological_definition':bd,'definition_id':f'{cd}__{bd}',
                          'excluded_buffer_n':int(np.sum(cc<0)),**{f'n_{x.replace("/","_")}':counts[x] for x in CELLS}})
        def add(af,metric,contrast,fam,mod,est,bdraw,extra=None):
            r={'endpoint':ep,'chemical_definition':cd,'biological_definition':bd,'definition_id':f'{cd}__{bd}',
               'analysis_family':af,'metric':metric,'contrast':contrast,'model_family':fam,'modality':mod,**boot_summary(est,bdraw)}
            if extra:r.update(extra)
            contrast_rows.append(r)
        for j,(fam,mod) in enumerate(MODEL_KEYS):
            for k,metric in enumerate(METRICS):
                r,ch,bi,jo=[obs[j,CI[x],k] for x in CELLS]; br,bch,bbi,bjo=[boot[:,j,CI[x],k] for x in CELLS]
                add('P9A_FACTORIAL',metric,'joint_vs_reference',fam,mod,jo-r,bjo-br,
                    {'reference_metric':r,'joint_metric':jo})
                add('P9A_FACTORIAL',metric,'factorial_interaction',fam,mod,jo-ch-bi+r,bjo-bch-bbi+br)
                add('P9A_OPTIMISM',metric,'pooled_OOF_minus_joint_OOD',fam,mod,pool_obs[j,k]-jo,pool_boot[:,j,k]-bjo,
                    {'pooled_OOF_metric':pool_obs[j,k],'joint_metric':jo})
        for fam in FAMILIES:
            js=midx[(fam,'STRUCTURE')]
            for fm in ['EARLY_FUSION','LATE_FUSION']:
                jf=midx[(fam,fm)]
                for k,metric in enumerate(METRICS):
                    sref=obs[js,CI['C0/B0'],k]; sj=obs[js,CI['C1/B1'],k]; fref=obs[jf,CI['C0/B0'],k]; fj=obs[jf,CI['C1/B1'],k]
                    est=(fj-sj)-(fref-sref)
                    bdv=(boot[:,jf,CI['C1/B1'],k]-boot[:,js,CI['C1/B1'],k])-(boot[:,jf,CI['C0/B0'],k]-boot[:,js,CI['C0/B0'],k])
                    add('P9A_FUSION_ROBUSTNESS',metric,'joint_minus_reference_fusion_advantage',fam,fm,est,bdv,
                        {'reference_fusion_advantage':fref-sref,'joint_fusion_advantage':fj-sj})
    return cell_rows,contrast_rows,comp_rows

def random_effects(df):
    keys=['chemical_definition','biological_definition','definition_id','analysis_family','metric','contrast','model_family','modality']
    z=df[df.bootstrap_supported & np.isfinite(df.estimate)&np.isfinite(df.bootstrap_se)&(df.bootstrap_se>0)].copy()
    if not len(z): return pd.DataFrame()
    z['vi']=z.bootstrap_se.to_numpy(float)**2
    z['wi']=1.0/z.vi
    z['wiy']=z.wi*z.estimate
    z['wiy2']=z.wi*z.estimate*z.estimate
    z['wi2']=z.wi*z.wi
    a=z.groupby(keys,sort=False,dropna=False).agg(endpoint_n=('estimate','size'),sum_w=('wi','sum'),sum_wy=('wiy','sum'),sum_wy2=('wiy2','sum'),sum_w2=('wi2','sum')).reset_index()
    a=a[a.endpoint_n>=3].copy()
    if not len(a):return a
    a['fixed']=a.sum_wy/a.sum_w
    a['Q']=a.sum_wy2-(a.sum_wy*a.sum_wy/a.sum_w)
    a['den']=a.sum_w-a.sum_w2/a.sum_w
    a['tau2']=np.where(a.den>0,np.maximum(0,(a.Q-(a.endpoint_n-1))/a.den),0.0)
    z=z.merge(a[keys+['tau2']],on=keys,how='inner',validate='many_to_one')
    z['wr']=1.0/(z.vi+z.tau2); z['wry']=z.wr*z.estimate
    b=z.groupby(keys,sort=False,dropna=False).agg(sum_wr=('wr','sum'),sum_wry=('wry','sum')).reset_index()
    out=a.merge(b,on=keys,validate='one_to_one')
    out['random_effect_mean']=out.sum_wry/out.sum_wr; out['random_effect_se']=np.sqrt(1.0/out.sum_wr)
    out['ci_low']=out.random_effect_mean-1.96*out.random_effect_se; out['ci_high']=out.random_effect_mean+1.96*out.random_effect_se
    zz=out.random_effect_mean/out.random_effect_se; out['p_value']=2*norm.sf(np.abs(zz))
    out=out[keys+['endpoint_n','random_effect_mean','random_effect_se','ci_low','ci_high','tau2','Q','p_value']]
    out['fdr_family']=out.analysis_family.astype(str)+'|'+out.metric.astype(str)+'|'+out.contrast.astype(str); out['q_value']=np.nan
    for fam,idx in out.groupby('fdr_family',sort=False).groups.items():out.loc[idx,'q_value']=bh(out.loc[idx,'p_value'].to_numpy(float))
    return out

def stability_summary(re):
    rows=[]
    group=['analysis_family','metric','contrast','model_family','modality']
    for kk,g in re.groupby(group,dropna=False):
        p=g[(g.chemical_definition==PRIMARY_DEF[0])&(g.biological_definition==PRIMARY_DEF[1])]
        if len(p)!=1: continue
        base=float(p.random_effect_mean.iloc[0]); base_sign=np.sign(base) if np.isfinite(base) and base!=0 else 0
        z=g[np.isfinite(g.random_effect_mean)].copy(); z['sign']=np.sign(z.random_effect_mean)
        match=(z.sign==base_sign) if base_sign!=0 else pd.Series(False,index=z.index)
        opp_sig=(z.sign==-base_sign)&(z.q_value<=.05) if base_sign!=0 else pd.Series(False,index=z.index)
        frac=float(match.mean()) if len(z) else np.nan
        rows.append({'analysis_family':kk[0],'metric':kk[1],'contrast':kk[2],'model_family':kk[3],'modality':kk[4],
                     'primary_definition_effect':base,'primary_direction_sign':int(base_sign),'supported_definition_n':len(z),
                     'direction_match_n':int(match.sum()),'direction_match_fraction':frac,'opposite_fdr_significant_n':int(opp_sig.sum()),
                     'gateD_support_definition_robust':bool(len(z)>0 and frac>=.75 and int(opp_sig.sum())==0)})
    return pd.DataFrame(rows)

def main():
    print('='*88);print('FACTORIAL-OOD PHASE 9A — SUPPORT-DEFINITION / SEVERITY ROBUSTNESS');print('='*88)
    lock=ROOT/'P9_00_robustness_lock_v1.0.json'; pre=ROOT/'Phase9_Output'/'P9_PREFLIGHT_STATUS.json'
    if not lock.exists() or sha256(lock)!=LOCK_SHA:raise RuntimeError('Phase9 frozen lock missing/mismatch')
    if not pre.exists() or json.load(open(pre)).get('status')!='PASS':raise RuntimeError('Phase9 preflight PASS is required')
    T('P9A-T01','Phase9 lock SHA matches',sha256(lock)==LOCK_SHA,sha256(lock));T('P9A-T02','Phase9 preflight PASS',True,'PASS')
    src=locate('P7_02_locked_OOF_with_factorial_cells.csv',['Phase7_Output/P7_02_locked_OOF_with_factorial_cells.csv'])
    chem=locate('P3_03_chemical_support_by_parent.csv',['Phase3/outputs/P3_03_chemical_support_by_parent.csv'])
    bio=locate('P3_04_biological_support_by_parent.csv',['Phase3/outputs/P3_04_biological_support_by_parent.csv'])
    p7cells=locate('P7_03_learned_factorial_cell_metrics.csv',['Phase7_Output/P7_03_learned_factorial_cell_metrics.csv'])
    p7bcells=locate('P7B_01_secondary_cell_metrics.csv',['Phase7B_Output/P7B_01_secondary_cell_metrics.csv'])
    T('P9A-T03','Locked Phase7 OOF SHA matches',sha256(src)==P7_SOURCE_SHA,sha256(src));T('P9A-T04','Chemical support SHA matches',sha256(chem)==P3_CHEM_SHA,sha256(chem));T('P9A-T05','Biological support SHA matches',sha256(bio)==P3_BIO_SHA,sha256(bio))
    T('P9A-T06','Phase7 primary cell table SHA matches',sha256(p7cells)==P7_PRIMARY_CELL_SHA,sha256(p7cells));T('P9A-T07','Phase7B secondary cell table SHA matches',sha256(p7bcells)==P7B_CELL_SHA,sha256(p7bcells))
    c=pd.read_csv(chem); b=pd.read_csv(bio); support=c.merge(b,on=['compound_id','outer_fold'],validate='one_to_one')
    T('P9A-T08','Support table has 2017 unique parents',len(support)==2017 and support.compound_id.nunique()==2017,len(support))
    # Verify all 25 definitions exist before performance.
    dummy=build_assignments(pd.DataFrame({'compound_id':sorted(support.compound_id)}),support)
    T('P9A-T09','Exactly 25 frozen support-definition combinations constructed',len(dummy)==25,len(dummy))
    pc=dummy[PRIMARY_DEF]; cnt={CELLS[i]:int(np.sum(pc==i)) for i in range(4)}
    T('P9A-T10','Primary definition reconstructs 1219/574/135/89 cells',cnt=={'C0/B0':1219,'C1/B0':574,'C0/B1':135,'C1/B1':89},cnt)
    print(f'Starting P9A: {B} paired parent bootstraps × 8 endpoints × 25 frozen support definitions using {JOBS} workers...')
    res=Parallel(n_jobs=JOBS,backend='loky',verbose=10)(delayed(endpoint_job)(ep,src,support) for ep in ENDPOINTS)
    cells=[]; cons=[]; comps=[]
    for a,bx,cx in res:cells.extend(a);cons.extend(bx);comps.extend(cx)
    cells=pd.DataFrame(cells);cons=pd.DataFrame(cons);comps=pd.DataFrame(comps)
    T('P9A-T11','Definition composition table has 200 rows',len(comps)==8*25,len(comps));T('P9A-T12','Cell point-metric table has 9600 rows',len(cells)==8*25*12*4,len(cells));T('P9A-T13','Contrast table has 84000 rows',len(cons)==8*25*420,len(cons))
    # FDR endpoint-level across P9A module x metric x contrast.
    cons['fdr_family']=cons.analysis_family.astype(str)+'|'+cons.metric.astype(str)+'|'+cons.contrast.astype(str);cons['q_value']=np.nan
    for fam,idx in cons.groupby('fdr_family').groups.items():
        p=cons.loc[idx,'bootstrap_p_two_sided'].where(cons.loc[idx,'bootstrap_supported'],np.nan).to_numpy(float);cons.loc[idx,'q_value']=bh(p)
    cons['ci_excludes_zero']=(cons.ci_low>0)|(cons.ci_high<0);cons['fdr_significant_0_05']=cons.q_value<=.05
    T('P9A-T14','Finite endpoint q-values are in [0,1]',cons.q_value.dropna().between(0,1).all())
    # Exact point metric reconstruction for primary definition against Phase7 and Phase7B.
    pri=cells[(cells.chemical_definition==PRIMARY_DEF[0])&(cells.biological_definition==PRIMARY_DEF[1])].copy()
    p7=pd.read_csv(p7cells); m=pri.merge(p7,on=['endpoint','model_family','modality','factorial_cell'],suffixes=('_p9','_p7'),validate='one_to_one')
    diffs=[]
    for met in ['AUPRC','MCC','Brier','ECE_10bin']:diffs.append(np.nanmax(np.abs(m[f'{met}_p9']-m[f'{met}_p7'])))
    T('P9A-T15','Primary-definition AUPRC/MCC/Brier/ECE point metrics reproduce Phase7',max(diffs)<2e-7,max(diffs))
    p7b=pd.read_csv(p7bcells); m2=pri.merge(p7b,on=['endpoint','model_family','modality','factorial_cell'],suffixes=('_p9','_p7b'),validate='one_to_one')
    diffs2=[]
    for met in ['AUROC','balanced_accuracy','sensitivity','specificity','balanced_MCC','Brier_skill_score']:diffs2.append(np.nanmax(np.abs(m2[f'{met}_p9']-m2[f'{met}_p7b'])))
    T('P9A-T16','Primary-definition secondary/prevalence-aware point metrics reproduce Phase7B',max(diffs2)<2e-7,max(diffs2))
    T('P9A-T17','Severe sparse definitions are retained as unsupported, never merged',cons.bootstrap_supported.dtype==bool,int((~cons.bootstrap_supported).sum()))
    re=random_effects(cons);stab=stability_summary(re)
    T('P9A-T18','Random-effects output is non-empty and finite q-values valid',len(re)>0 and re.q_value.dropna().between(0,1).all(),len(re));T('P9A-T19','Gate-D stability table has 420 headline model/metric/contrast rows',len(stab)==420,len(stab))
    comps.to_csv(OUT/'P9A_01_definition_cell_counts.csv',index=False);cells.to_csv(OUT/'P9A_02_cell_metrics.csv',index=False);cons.to_csv(OUT/'P9A_03_contrasts.csv',index=False);re.to_csv(OUT/'P9A_04_random_effects.csv',index=False);stab.to_csv(OUT/'P9A_05_definition_stability.csv',index=False)
    t=pd.DataFrame(TESTS);t.to_csv(OUT/'P9A_06_integrity_tests.csv',index=False)
    if (t.status!='PASS').any():
        print(t[t.status!='PASS'].to_string(index=False));raise RuntimeError('P9A integrity failure')
    status={'phase':'9A','status':'PASS','phase9_lock_sha256':LOCK_SHA,'bootstrap_resamples':B,'definition_combinations':25,
            'performance_source':'immutable Phase7 OOF predictions','models_retrained':False,'integrity_pass_n':int((t.status=='PASS').sum()),'integrity_total_n':len(t)}
    json.dump(status,open(OUT/'P9A_FINAL_STATUS.json','w'),indent=2)
    # Checksum and upload package (summary outputs only; bootstrap is deterministic/reconstructable from frozen seed and source predictions).
    names=['P9A_01_definition_cell_counts.csv','P9A_02_cell_metrics.csv','P9A_03_contrasts.csv','P9A_04_random_effects.csv','P9A_05_definition_stability.csv','P9A_06_integrity_tests.csv','P9A_FINAL_STATUS.json']
    with open(OUT/'SHA256SUMS_P9A.txt','w') as f:
        for n in names:f.write(f'{sha256(OUT/n)}  {n}\n')
    names+=['SHA256SUMS_P9A.txt'];z=OUT/'P9A_UPLOAD_TO_CHAT.zip'
    with zipfile.ZipFile(z,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zz:
        for n in names:zz.write(OUT/n,arcname=n)
    print('\n'+'='*88);print('PHASE 9A LOCAL COMPUTATION PASS');print(f'Definition combinations: 25');print(f'Cell metric rows: {len(cells)}');print(f'Contrast rows: {len(cons)}');print(f'Gate-D stability rows: {len(stab)}');print(f'Integrity: {len(t)}/{len(t)} PASS');print('Upload:',z);print('Upload ZIP SHA256:',sha256(z));print('No retraining or retuning was performed.');print('='*88)
if __name__=='__main__':main()
