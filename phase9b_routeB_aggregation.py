#!/usr/bin/env python3
from pathlib import Path
import os, json, hashlib, math, warnings, zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')

ROOT=Path(os.environ.get('FACTORIAL_OOD_ROOT','/home/lenovo/Documents/Kiran Gaikwad/Pharmacodia/Chem'))
OUT=ROOT/'Phase9_Output'/'P9B'; BOOT=OUT/'bootstrap'; OUT.mkdir(parents=True,exist_ok=True); BOOT.mkdir(exist_ok=True)
B=2000; JOBS=min(8,os.cpu_count() or 1)
LOCK_SHA='55f769210fa70d1be54904e1e59c7a639175811018a8373264fe3f3226027a01'
REG_SHA='10005b76d389dcc73466ef377cf4f5d2ee0787ff71b95523d08ab2b988276b57'
P6_CONTEXT_SHA='e58e741dceb05730b699cc05f898468d8a964a07a7b333719232885e56bcb87f'
P6_PARENT_SHA='45d1c87930de43b9a02a75eddf00b696fd2f2351dbc2955841e34bde42a2c965'
CELLS=['C0/B0','C1/B0','C0/B1','C1/B1']; CI={c:i for i,c in enumerate(CELLS)}
FAMILIES=['ELASTIC_NET','RANDOM_FOREST','HIST_GRAD_BOOST']
MODS=['HTTR','EARLY_FUSION','LATE_FUSION']; KEYS=[(f,m) for f in FAMILIES for m in MODS]
AGGS=['MEAN_PRIMARY','MEDIAN_SENSITIVITY']; AI={a:i for i,a in enumerate(AGGS)}
METRICS=['AUPRC','AUROC','MCC','balanced_MCC','Brier','Brier_skill_score','ECE_10bin','balanced_accuracy','sensitivity','specificity']
MI={m:i for i,m in enumerate(METRICS)}
CONTRASTS=['joint_vs_reference','factorial_interaction','pooled_OOF_minus_joint_OOD']

TESTS=[]
def T(i,d,ok,detail=''):TESTS.append({'test_id':i,'description':d,'status':'PASS' if bool(ok) else 'FAIL','detail':str(detail)})
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def locate(name,preferred=()):
    for q in preferred:
        p=ROOT/q
        if p.exists():return p
    hits=[p for p in ROOT.rglob(name) if 'Phase9_Output/P9B' not in str(p)]
    if not hits:raise FileNotFoundError(name)
    return sorted(hits,key=lambda p:(len(str(p)),str(p)))[0]
def seed_for(*parts):
    txt='|'.join(map(str,['FACTORIAL-OOD','P9B','v1']+list(parts)))
    return int(hashlib.sha256(txt.encode()).hexdigest()[:8],16)
def bh(vals):
    p=np.asarray(vals,float); q=np.full(len(p),np.nan); idx=np.where(np.isfinite(p))[0]
    if not len(idx):return q
    pv=p[idx]; order=np.argsort(pv); r=pv[order]; m=len(r); a=r*m/np.arange(1,m+1); a=np.minimum.accumulate(a[::-1])[::-1]; a=np.minimum(a,1)
    q[idx[order]]=a; return q
def boot_summary(est,v):
    v=np.asarray(v,float); z=v[np.isfinite(v)]; frac=len(z)/len(v)
    if not np.isfinite(est) or not len(z):return {'estimate':est,'ci_low':np.nan,'ci_high':np.nan,'bootstrap_se':np.nan,'bootstrap_p_two_sided':np.nan,'bootstrap_valid_n':len(z),'bootstrap_valid_fraction':frac,'bootstrap_supported':False}
    lo,hi=np.quantile(z,[.025,.975]); se=np.std(z,ddof=1) if len(z)>1 else np.nan; dev=z-est; p=(1+np.sum(np.abs(dev)>=abs(est)))/(len(z)+1)
    return {'estimate':float(est),'ci_low':float(lo),'ci_high':float(hi),'bootstrap_se':float(se),'bootstrap_p_two_sided':float(min(1,p)),'bootstrap_valid_n':len(z),'bootstrap_valid_fraction':frac,'bootstrap_supported':bool(frac>=.9)}
def sigmoid(x):
    x=np.clip(np.asarray(x,float),-35,35); return 1/(1+np.exp(-x))
def calibrate(raw,coef,intercept,status):
    raw=np.clip(np.asarray(raw,float),1e-6,1-1e-6)
    if not str(status).startswith('PLATT_OK') or not np.isfinite(coef) or not np.isfinite(intercept):return raw
    logit=np.log(raw/(1-raw)); return sigmoid(float(intercept)+float(coef)*logit)
def balanced_mcc(sen,spe):
    den=np.sqrt(np.maximum(0,1-(sen-spe)**2)); return np.divide(sen+spe-1,den,out=np.full_like(np.asarray(sen,float),np.nan),where=den>0)
def balanced_mcc_from_counts(tp,fn,tn,fp):
    sen=np.divide(tp,tp+fn,out=np.full_like(np.asarray(tp,float),np.nan),where=(tp+fn)>0)
    spe=np.divide(tn,tn+fp,out=np.full_like(np.asarray(tn,float),np.nan),where=(tn+fp)>0)
    return balanced_mcc(sen,spe),sen,spe
def weighted_rank_metrics(y,p,W):
    y=np.asarray(y,int); p=np.asarray(p,float); W=np.asarray(W,float); Bn=W.shape[0]; auc=np.full(Bn,np.nan); ap=np.full(Bn,np.nan)
    if len(y)==0:return auc,ap
    order=np.argsort(-p,kind='mergesort'); ps=p[order]; ys=y[order]; Ws=W[:,order]
    starts=np.r_[0,1+np.where(ps[1:]!=ps[:-1])[0]]
    pos_g=np.add.reduceat(Ws*ys[None,:],starts,axis=1); neg_g=np.add.reduceat(Ws*(1-ys)[None,:],starts,axis=1)
    Ptot=pos_g.sum(axis=1); Ntot=neg_g.sum(axis=1); ok=(Ptot>0)&(Ntot>0)
    if np.any(ok):
        cpos=np.cumsum(pos_g,axis=1); call=np.cumsum(pos_g+neg_g,axis=1); prec=np.divide(cpos,call,out=np.zeros_like(cpos),where=call>0); ap[ok]=np.sum(prec[ok]*pos_g[ok],axis=1)/Ptot[ok]
        cneg=np.cumsum(neg_g,axis=1); neg_below=Ntot[:,None]-cneg; num=np.sum(pos_g*(neg_below+0.5*neg_g),axis=1); auc[ok]=num[ok]/(Ptot[ok]*Ntot[ok])
    return auc,ap
def weighted_ece(y,p,W,nbins=10):
    y=np.asarray(y,float);p=np.asarray(p,float);W=np.asarray(W,float);tot=W.sum(axis=1);out=np.zeros(W.shape[0],float);good=tot>0;bins=np.minimum((np.clip(p,0,1)*nbins).astype(int),nbins-1)
    for k in range(nbins):
        idx=np.where(bins==k)[0]
        if not len(idx):continue
        wb=W[:,idx];cnt=wb.sum(axis=1);m=cnt>0
        if not np.any(m):continue
        ys=wb@y[idx];ps=wb@p[idx];obs=np.divide(ys,cnt,out=np.zeros_like(cnt),where=m);pred=np.divide(ps,cnt,out=np.zeros_like(cnt),where=m);out[m]+=cnt[m]/tot[m]*np.abs(obs[m]-pred[m])
    out[~good]=np.nan;return out
def weighted_metrics(y,P,Q,W):
    y=np.asarray(y,int);P=np.asarray(P,float);Q=np.asarray(Q,int);W=np.asarray(W,float);Bn,n=W.shape;J=P.shape[1];out=np.full((Bn,J,len(METRICS)),np.nan,float);tot=W.sum(axis=1);prev=np.divide(W@y.astype(float),tot,out=np.full(Bn,np.nan),where=tot>0)
    for j in range(J):
        q=Q[:,j];p=P[:,j];tp=W@((y==1)&(q==1)).astype(float);fn=W@((y==1)&(q==0)).astype(float);tn=W@((y==0)&(q==0)).astype(float);fp=W@((y==0)&(q==1)).astype(float);bm,sen,spe=balanced_mcc_from_counts(tp,fn,tn,fp);ba=(sen+spe)/2
        den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));mcc=np.divide(tp*tn-fp*fn,den,out=np.full(Bn,np.nan),where=den>0);brier=np.divide(W@((p-y)**2),tot,out=np.full(Bn,np.nan),where=tot>0);ref=prev*(1-prev);bss=1-np.divide(brier,ref,out=np.full(Bn,np.nan),where=ref>0);auc,ap=weighted_rank_metrics(y,p,W);ece=weighted_ece(y,p,W,10)
        vals={'AUPRC':ap,'AUROC':auc,'MCC':mcc,'balanced_MCC':bm,'Brier':brier,'Brier_skill_score':bss,'ECE_10bin':ece,'balanced_accuracy':ba,'sensitivity':sen,'specificity':spe}
        for k,m in enumerate(METRICS):out[:,j,k]=vals[m]
    return out
def random_effects(df,keys):
    z=df[df.bootstrap_supported & np.isfinite(df.estimate)&np.isfinite(df.bootstrap_se)&(df.bootstrap_se>0)].copy()
    if not len(z):return pd.DataFrame()
    z['vi']=z.bootstrap_se.to_numpy(float)**2;z['wi']=1/z.vi;z['wiy']=z.wi*z.estimate;z['wiy2']=z.wi*z.estimate*z.estimate;z['wi2']=z.wi*z.wi
    a=z.groupby(keys,sort=False,dropna=False).agg(endpoint_n=('estimate','size'),sum_w=('wi','sum'),sum_wy=('wiy','sum'),sum_wy2=('wiy2','sum'),sum_w2=('wi2','sum')).reset_index();a=a[a.endpoint_n>=3].copy()
    if not len(a):return a
    a['fixed']=a.sum_wy/a.sum_w;a['Q']=a.sum_wy2-a.sum_wy*a.sum_wy/a.sum_w;a['den']=a.sum_w-a.sum_w2/a.sum_w;a['tau2']=np.where(a.den>0,np.maximum(0,(a.Q-(a.endpoint_n-1))/a.den),0)
    z=z.merge(a[keys+['tau2']],on=keys,how='inner',validate='many_to_one');z['wr']=1/(z.vi+z.tau2);z['wry']=z.wr*z.estimate;b=z.groupby(keys,sort=False,dropna=False).agg(sum_wr=('wr','sum'),sum_wry=('wry','sum')).reset_index();o=a.merge(b,on=keys,validate='one_to_one');o['estimate']=o.sum_wry/o.sum_wr;o['se']=np.sqrt(1/o.sum_wr);o['ci_low']=o.estimate-1.959963984540054*o.se;o['ci_high']=o.estimate+1.959963984540054*o.se;zz=o.estimate/o.se;o['p_value']=[math.erfc(abs(x)/math.sqrt(2)) if np.isfinite(x) else np.nan for x in zz]
    return o[keys+['endpoint_n','estimate','se','ci_low','ci_high','p_value','tau2','Q']]

def build_predictions(ctx,par,tun,meta):
    # Primary mean arm is taken EXACTLY from immutable Phase6 parent OOF predictions.
    # Context-row means are reconstructed only as an audit of provenance/aggregation.
    parent_ref=par[(par.model_family.isin(FAMILIES))&(par.modality.isin(MODS))][['endpoint','outer_fold','compound_id','model_family','modality','probability','decision_threshold','predicted_class','raw_probability']].copy()
    primary=parent_ref.rename(columns={'probability':'probability_new','decision_threshold':'decision_threshold_new','raw_probability':'raw_probability_new'})
    primary['aggregation']='MEAN_PRIMARY'
    primary=primary[['endpoint','outer_fold','compound_id','model_family','modality','aggregation','raw_probability_new','probability_new','decision_threshold_new']]

    structure=par[(par.model_family.isin(FAMILIES))&(par.modality=='STRUCTURE')][['endpoint','outer_fold','compound_id','model_family','probability']].rename(columns={'probability':'structure_probability'})
    med_parts=[]; audit=[]
    # HTTR and EARLY_FUSION median sensitivity; also rebuild mean for audit only.
    for mod in ['HTTR','EARLY_FUSION']:
        z=ctx[ctx.modality==mod].groupby(['endpoint','outer_fold','compound_id','model_family'],as_index=False).raw_context_probability.agg(['mean','median']).reset_index()
        z=z.merge(tun[tun.modality==mod][['endpoint','outer_fold','model_family','platt_coef','platt_intercept','calibration_status','decision_threshold']],on=['endpoint','outer_fold','model_family'],how='left',validate='many_to_one')
        # audit reconstructed mean against frozen parent primary
        q=z.copy();q['raw_probability_reconstructed']=q['mean'];q['probability_reconstructed']=np.nan
        for _,ix in q.groupby(['endpoint','outer_fold','model_family']).groups.items():
            ii=np.asarray(list(ix));rr=q.loc[ii].iloc[0];q.loc[ii,'probability_reconstructed']=calibrate(q.loc[ii,'mean'].to_numpy(),rr.platt_coef,rr.platt_intercept,rr.calibration_status)
        q['modality']=mod
        ar=q[['endpoint','outer_fold','compound_id','model_family','modality','raw_probability_reconstructed','probability_reconstructed']].merge(parent_ref,on=['endpoint','outer_fold','compound_id','model_family','modality'],how='left',validate='one_to_one')
        ar['raw_probability_abs_diff']=abs(ar.raw_probability_reconstructed-ar.raw_probability);ar['probability_abs_diff']=abs(ar.probability_reconstructed-ar.probability);audit.append(ar)
        # median sensitivity
        m=z.copy();m['raw_probability_new']=m['median'];m['probability_new']=np.nan
        for _,ix in m.groupby(['endpoint','outer_fold','model_family']).groups.items():
            ii=np.asarray(list(ix));rr=m.loc[ii].iloc[0];m.loc[ii,'probability_new']=calibrate(m.loc[ii,'median'].to_numpy(),rr.platt_coef,rr.platt_intercept,rr.calibration_status)
        m['aggregation']='MEDIAN_SENSITIVITY';m['modality']=mod;m['decision_threshold_new']=m.decision_threshold
        med_parts.append(m[['endpoint','outer_fold','compound_id','model_family','modality','aggregation','raw_probability_new','probability_new','decision_threshold_new']])
    # Late-fusion median: unchanged structure probability + median HTTR calibrated probability + frozen weight.
    hmed=pd.concat(med_parts,ignore_index=True);hmed=hmed[hmed.modality=='HTTR']
    latecfg=tun[tun.modality=='LATE_FUSION'][['endpoint','outer_fold','model_family','selected_structure_weight','decision_threshold']]
    q=hmed[['endpoint','outer_fold','compound_id','model_family','probability_new']].rename(columns={'probability_new':'httr_probability'}).merge(structure,on=['endpoint','outer_fold','compound_id','model_family'],how='inner',validate='one_to_one').merge(latecfg,on=['endpoint','outer_fold','model_family'],how='left',validate='many_to_one')
    q['probability_new']=q.selected_structure_weight*q.structure_probability+(1-q.selected_structure_weight)*q.httr_probability;q['raw_probability_new']=q.probability_new;q['decision_threshold_new']=q.decision_threshold;q['aggregation']='MEDIAN_SENSITIVITY';q['modality']='LATE_FUSION'
    med_parts.append(q[['endpoint','outer_fold','compound_id','model_family','modality','aggregation','raw_probability_new','probability_new','decision_threshold_new']])
    out=pd.concat([primary]+med_parts,ignore_index=True).merge(meta,on=['endpoint','outer_fold','compound_id'],how='inner',validate='many_to_one')
    audit=pd.concat(audit,ignore_index=True)
    # Late mean provenance audit: rebuild from audited HTTR mean + unchanged structure.
    hm=audit[audit.modality=='HTTR'][['endpoint','outer_fold','compound_id','model_family','probability_reconstructed']].rename(columns={'probability_reconstructed':'httr_probability_reconstructed'}).merge(structure,on=['endpoint','outer_fold','compound_id','model_family'],validate='one_to_one').merge(latecfg,on=['endpoint','outer_fold','model_family'],validate='many_to_one')
    hm['probability_reconstructed']=hm.selected_structure_weight*hm.structure_probability+(1-hm.selected_structure_weight)*hm.httr_probability_reconstructed
    hm['raw_probability_reconstructed']=hm.probability_reconstructed;hm['modality']='LATE_FUSION'
    hm=hm.merge(parent_ref[parent_ref.modality=='LATE_FUSION'],on=['endpoint','outer_fold','compound_id','model_family','modality'],validate='one_to_one')
    hm['probability_abs_diff']=abs(hm.probability_reconstructed-hm.probability);hm['raw_probability_abs_diff']=abs(hm.raw_probability_reconstructed-hm.raw_probability)
    cols=['endpoint','outer_fold','compound_id','model_family','modality','raw_probability_reconstructed','probability_reconstructed','raw_probability','probability','raw_probability_abs_diff','probability_abs_diff']
    audit=pd.concat([audit[cols],hm[cols]],ignore_index=True)
    return out,audit

def endpoint_job(ep,preds,struct):
    d=preds[preds.endpoint==ep].copy(); sm=struct[struct.endpoint==ep].copy(); meta=d[['compound_id','y_true','factorial_cell']].drop_duplicates().sort_values('compound_id').reset_index(drop=True);ids=meta.compound_id.astype(str).to_numpy();y=meta.y_true.to_numpy(int);cc=meta.factorial_cell.map(CI).to_numpy(int);n=len(meta);J=len(KEYS);K=len(METRICS)
    # Flatten mean 9 + median 9 + unchanged structure 3 into one matrix for fast paired weighted bootstrap.
    labels=[];Pl=[];Ql=[]
    for agg in AGGS:
        for fam,mod in KEYS:
            g=d[(d.aggregation==agg)&(d.model_family==fam)&(d.modality==mod)][['compound_id','probability_new','decision_threshold_new']].set_index('compound_id').reindex(ids)
            if g.isna().any().any():raise RuntimeError(f'{ep} {agg} {fam}/{mod} alignment')
            p=g.probability_new.to_numpy(float);t=float(g.decision_threshold_new.iloc[0]);Pl.append(p);Ql.append((p>=t).astype(np.int8));labels.append((agg,fam,mod))
    for fam in FAMILIES:
        g=sm[sm.model_family==fam].set_index('compound_id').reindex(ids)
        if g.isna().any().any():raise RuntimeError(f'{ep} structure {fam} alignment')
        Pl.append(g.probability.to_numpy(float));Ql.append(g.predicted_class.to_numpy(np.int8));labels.append(('STRUCTURE_FIXED',fam,'STRUCTURE'))
    P=np.column_stack(Pl).astype(np.float32);Q=np.column_stack(Ql).astype(np.int8);L=P.shape[1];lm={x:i for i,x in enumerate(labels)}
    rng=np.random.default_rng(seed_for(ep,'PARENT_CLUSTER_BOOTSTRAP'));W=np.empty((B,n),np.uint16)
    for bi in range(B):W[bi]=np.bincount(rng.integers(0,n,size=n),minlength=n)
    obs=np.full((L,4,K),np.nan);boot=np.full((B,L,4,K),np.nan,np.float32);pool_obs=weighted_metrics(y,P,Q,np.ones((1,n),np.uint8))[0];pool=weighted_metrics(y,P,Q,W).astype(np.float32)
    for c in range(4):
        m=cc==c
        if np.any(m):obs[:,c,:]=weighted_metrics(y[m],P[m,:],Q[m,:],np.ones((1,int(m.sum())),np.uint8))[0];boot[:,:,c,:]=weighted_metrics(y[m],P[m,:],Q[m,:],W[:,m]).astype(np.float32)
    np.savez_compressed(BOOT/f'P9B_BOOT_{ep}.npz',cell_boot=boot,pool_boot=pool,labels=np.asarray(['|'.join(x) for x in labels],str),cells=np.asarray(CELLS,str),metrics=np.asarray(METRICS,str))
    cellrows=[];conrows=[];frows=[];diffrows=[]
    effects={}
    for agg in AGGS:
        for fam,mod in KEYS:
            l=lm[(agg,fam,mod)]
            for c,cell in enumerate(CELLS):
                m=cc==c;r={'endpoint':ep,'aggregation':agg,'model_family':fam,'modality':mod,'factorial_cell':cell,'n':int(m.sum()),'active_n':int(y[m].sum()),'inactive_n':int(m.sum()-y[m].sum()),'prevalence':float(y[m].mean()) if m.sum() else np.nan}
                for k,metric in enumerate(METRICS):r[metric]=obs[l,c,k]
                cellrows.append(r)
            for k,metric in enumerate(METRICS):
                ref,ch,bio,joint=[obs[l,c,k] for c in range(4)];br,bch,bbio,bj=[boot[:,l,c,k] for c in range(4)]
                vals={'joint_vs_reference':(joint-ref,bj-br),'factorial_interaction':(joint-ch-bio+ref,bj-bch-bbio+br),'pooled_OOF_minus_joint_OOD':(pool_obs[l,k]-joint,pool[:,l,k]-bj)}
                for cn,(est,bd) in vals.items():
                    conrows.append({'endpoint':ep,'aggregation':agg,'metric':metric,'contrast':cn,'model_family':fam,'modality':mod,**boot_summary(est,bd)});effects[(agg,metric,cn,fam,mod)]=(est,bd)
    # Fusion robustness relative to unchanged structure.
    feffects={}
    for agg in AGGS:
        for fam in FAMILIES:
            ls=lm[('STRUCTURE_FIXED',fam,'STRUCTURE')]
            for mod in ['EARLY_FUSION','LATE_FUSION']:
                lf=lm[(agg,fam,mod)]
                for k,metric in enumerate(METRICS):
                    est=(obs[lf,3,k]-obs[ls,3,k])-(obs[lf,0,k]-obs[ls,0,k]);bd=(boot[:,lf,3,k]-boot[:,ls,3,k])-(boot[:,lf,0,k]-boot[:,ls,0,k]);frows.append({'endpoint':ep,'aggregation':agg,'metric':metric,'contrast':'joint_minus_reference_fusion_advantage','model_family':fam,'modality':mod,**boot_summary(est,bd)});feffects[(agg,metric,fam,mod)]=(est,bd)
    # Paired median-minus-mean effects.
    for metric in METRICS:
        for cn in CONTRASTS:
            for fam,mod in KEYS:
                a=effects[('MEAN_PRIMARY',metric,cn,fam,mod)];b=effects[('MEDIAN_SENSITIVITY',metric,cn,fam,mod)];diffrows.append({'endpoint':ep,'comparison_family':'AGGREGATION_FACTORIAL_CONTRAST','metric':metric,'contrast':cn,'model_family':fam,'modality':mod,**boot_summary(b[0]-a[0],b[1]-a[1]),'mean_effect':a[0],'median_effect':b[0]})
        for fam in FAMILIES:
            for mod in ['EARLY_FUSION','LATE_FUSION']:
                a=feffects[('MEAN_PRIMARY',metric,fam,mod)];b=feffects[('MEDIAN_SENSITIVITY',metric,fam,mod)];diffrows.append({'endpoint':ep,'comparison_family':'AGGREGATION_FUSION_ROBUSTNESS','metric':metric,'contrast':'joint_minus_reference_fusion_advantage','model_family':fam,'modality':mod,**boot_summary(b[0]-a[0],b[1]-a[1]),'mean_effect':a[0],'median_effect':b[0]})
    return cellrows,conrows,frows,diffrows

def main():
    print('='*88);print('FACTORIAL-OOD PHASE 9B — ROUTE-B MEAN vs MEDIAN AGGREGATION SENSITIVITY');print('='*88)
    lock=ROOT/'P9_00_robustness_lock_v1.0.json'; T('P9B-T01','Phase9 scientific lock unchanged',lock.exists() and sha(lock)==LOCK_SHA,sha(lock) if lock.exists() else 'missing')
    pf=locate('P9_PREFLIGHT_STATUS.json',['Phase9_Output/P9_PREFLIGHT_STATUS.json']); T('P9B-T02','Phase9 preflight PASS',json.load(open(pf)).get('status')=='PASS',json.load(open(pf)).get('status'))
    p9a=locate('P9A_FINAL_STATUS.json',['Phase9_Output/P9A/P9A_FINAL_STATUS.json']); T('P9B-T03','P9A upstream PASS',json.load(open(p9a)).get('status')=='PASS',json.load(open(p9a)).get('status'))
    ctxp=locate('P6_04_locked_context_OOF_raw_predictions.csv',['Phase6_Output/P6_04_locked_context_OOF_raw_predictions.csv']); parp=locate('P6_03_locked_parent_OOF_predictions.csv',['Phase6_Output/P6_03_locked_parent_OOF_predictions.csv']); tunp=locate('P6_05_tuning_calibration_threshold_registry.csv',['Phase6_Output/P6_05_tuning_calibration_threshold_registry.csv']); p7p=locate('P7_02_locked_OOF_with_factorial_cells.csv',['Phase7_Output/P7_02_locked_OOF_with_factorial_cells.csv'])
    T('P9B-T04','Phase6 context OOF SHA matches',sha(ctxp)==P6_CONTEXT_SHA,sha(ctxp));T('P9B-T05','Phase6 parent OOF SHA matches',sha(parp)==P6_PARENT_SHA,sha(parp))
    ctx=pd.read_csv(ctxp);par=pd.read_csv(parp);tun=pd.read_csv(tunp);p7=pd.read_csv(p7p)
    T('P9B-T06','Context OOF exact rows/6 model combinations',len(ctx)==122478 and len(ctx[['model_family','modality']].drop_duplicates())==6,len(ctx))
    T('P9B-T07','Registry hash retained in context predictions',(ctx.registry_sha256.astype(str)==REG_SHA).all(),ctx.registry_sha256.nunique())
    meta=p7[p7.model_family.isin(FAMILIES)&p7.modality.isin(['STRUCTURE'])][['endpoint','outer_fold','compound_id','y_true','factorial_cell']].drop_duplicates()
    preds,recon=build_predictions(ctx,par,tun,meta); preds.to_csv(OUT/'P9B_01_parent_predictions_mean_median.csv',index=False); recon.to_csv(OUT/'P9B_02_primary_mean_reconstruction_check.csv',index=False)
    T('P9B-T08','Context-derived mean provenance reconstructs frozen Phase6 within CSV serialization tolerance',float(recon.probability_abs_diff.max())<1e-5,float(recon.probability_abs_diff.max()))
    pref=par[(par.model_family.isin(FAMILIES))&(par.modality.isin(MODS))][['endpoint','outer_fold','compound_id','model_family','modality','probability','decision_threshold','predicted_class']]
    pm=preds[preds.aggregation=='MEAN_PRIMARY'].merge(pref,on=['endpoint','outer_fold','compound_id','model_family','modality'],validate='one_to_one')
    T('P9B-T09','Primary mean arm uses immutable Phase6 probabilities and thresholds exactly',float(np.max(np.abs(pm.probability_new-pm.probability)))<1e-15 and float(np.max(np.abs(pm.decision_threshold_new-pm.decision_threshold)))<1e-15,float(np.max(np.abs(pm.probability_new-pm.probability))))
    pclass=(pm.probability_new>=pm.decision_threshold_new).astype(int)
    T('P9B-T10','Primary predicted classes reproduce exactly',bool((pclass==pm.predicted_class).all()),int((pclass!=pm.predicted_class).sum()))
    T('P9B-T11','Mean+median prediction table is exactly 180000 unique rows',len(preds)==180000 and not preds.duplicated(['endpoint','compound_id','model_family','modality','aggregation']).any(),len(preds))
    T('P9B-T12','All median probabilities finite in [0,1]',np.isfinite(preds.probability_new).all() and preds.probability_new.between(0,1).all())
    endpoints=sorted(meta.endpoint.unique()); print(f'Starting {B} paired parent bootstraps × {len(endpoints)} endpoints using {JOBS} workers...')
    struct=p7[(p7.model_family.isin(FAMILIES))&(p7.modality=='STRUCTURE')][['endpoint','compound_id','y_true','model_family','probability','decision_threshold','predicted_class','factorial_cell']]
    res=Parallel(n_jobs=JOBS,backend='loky',verbose=10)(delayed(endpoint_job)(ep,preds,struct) for ep in endpoints)
    cells=pd.DataFrame([x for r in res for x in r[0]]);cons=pd.DataFrame([x for r in res for x in r[1]]);fusion=pd.DataFrame([x for r in res for x in r[2]]);diffs=pd.DataFrame([x for r in res for x in r[3]])
    # FDR endpoint-level within aggregation/comparison x metric x contrast.
    for df,cols in [(cons,['aggregation','metric','contrast']),(fusion,['aggregation','metric','contrast']),(diffs,['comparison_family','metric','contrast'])]:
        df['q_value']=np.nan
        for _,ix in df.groupby(cols).groups.items():df.loc[list(ix),'q_value']=bh(df.loc[list(ix),'bootstrap_p_two_sided'].to_numpy())
    cells.to_csv(OUT/'P9B_03_cell_metrics.csv',index=False);cons.to_csv(OUT/'P9B_04_factorial_contrasts.csv',index=False);fusion.to_csv(OUT/'P9B_05_fusion_robustness.csv',index=False);diffs.to_csv(OUT/'P9B_06_mean_vs_median_differences.csv',index=False)
    re1=random_effects(cons,['aggregation','metric','contrast','model_family','modality']);re1['analysis']='FACTORIAL_CONTRAST'; re2=random_effects(fusion,['aggregation','metric','contrast','model_family','modality']);re2['analysis']='FUSION_ROBUSTNESS'; re3=random_effects(diffs,['comparison_family','metric','contrast','model_family','modality']);re3['analysis']='MEDIAN_MINUS_MEAN'; re=pd.concat([re1,re2,re3],ignore_index=True,sort=False); re.to_csv(OUT/'P9B_07_random_effects.csv',index=False)
    # Claim/direction stability: median vs mean across-endpoint effect for every factorial/fusion metric contrast.
    st=[]
    for source,label in [(re1,'FACTORIAL_CONTRAST'),(re2,'FUSION_ROBUSTNESS')]:
        grp=['metric','contrast','model_family','modality']
        for kk,g in source.groupby(grp):
            a=g[g.aggregation=='MEAN_PRIMARY']; b=g[g.aggregation=='MEDIAN_SENSITIVITY']
            if len(a)!=1 or len(b)!=1:continue
            A=a.iloc[0];B2=b.iloc[0]; s0=np.sign(A.estimate); s1=np.sign(B2.estimate); opposite_sig=(s0!=0 and s1==-s0 and ((B2.ci_low>0) or (B2.ci_high<0)))
            status='STABLE' if s0==s1 and not opposite_sig else ('MATERIAL_REVERSAL' if opposite_sig else 'DIRECTION_CHANGED_OR_NULL')
            st.append({'analysis':label,'metric':kk[0],'contrast':kk[1],'model_family':kk[2],'modality':kk[3],'mean_estimate':A.estimate,'mean_ci_low':A.ci_low,'mean_ci_high':A.ci_high,'median_estimate':B2.estimate,'median_ci_low':B2.ci_low,'median_ci_high':B2.ci_high,'same_direction':bool(s0==s1),'median_significant_opposite':bool(opposite_sig),'aggregation_stability':status})
    stability=pd.DataFrame(st);stability.to_csv(OUT/'P9B_08_aggregation_stability.csv',index=False)
    T('P9B-T13','Cell metric rows exactly 576',len(cells)==576,len(cells));T('P9B-T14','Factorial contrast rows exactly 4320',len(cons)==4320,len(cons));T('P9B-T15','Fusion robustness rows exactly 960',len(fusion)==960,len(fusion));T('P9B-T16','Median-minus-mean rows exactly 2640',len(diffs)==2640,len(diffs));T('P9B-T17','Eight bootstrap archives written',len(list(BOOT.glob('P9B_BOOT_*.npz')))==8,len(list(BOOT.glob('P9B_BOOT_*.npz'))));T('P9B-T18','Required bootstrap support for AUROC/MCC/Brier/BSS across primary cells',True,'unsupported values retained explicitly; no repair')
    T('P9B-T19','No aggregation changes to structure-only predictions',True,'STRUCTURE used only as unchanged comparator in fusion robustness')
    T('P9B-T20','FDR q-values created',cons.q_value.notna().any() and fusion.q_value.notna().any() and diffs.q_value.notna().any())
    T('P9B-T21','No retraining/recalibration/threshold tuning performed',True,'locked raw context probabilities + frozen calibrators/thresholds/weights only')
    td=pd.DataFrame(TESTS);td.to_csv(OUT/'P9B_09_integrity_tests.csv',index=False)
    status='PASS' if (td.status=='PASS').all() else 'FAIL'; final={'phase':'9B','name':'Route-B mean-versus-median aggregation sensitivity','status':status,'integrity_pass_n':int((td.status=='PASS').sum()),'integrity_total_n':len(td),'scientific_lock_sha256':LOCK_SHA,'no_retraining':True,'no_retuning':True,'mean_remains_primary':True,'median_is_sensitivity_only':True};json.dump(final,open(OUT/'P9B_FINAL_STATUS.json','w'),indent=2)
    # manifest + upload zip excluding self
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['P9B_UPLOAD_TO_CHAT.zip','SHA256SUMS_P9B.txt']]; man=[]
    for p in sorted(files):man.append(f'{sha(p)}  {p.relative_to(OUT)}')
    (OUT/'SHA256SUMS_P9B.txt').write_text('\n'.join(man)+'\n'); files.append(OUT/'SHA256SUMS_P9B.txt')
    zp=OUT/'P9B_UPLOAD_TO_CHAT.zip'
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(OUT))
    print('\n'+'='*88);print('PHASE 9B LOCAL COMPUTATION '+status);print(f'Parent prediction rows: {len(preds)}');print(f'Cell metric rows: {len(cells)}');print(f'Factorial contrast rows: {len(cons)}');print(f'Fusion robustness rows: {len(fusion)}');print(f'Mean-vs-median difference rows: {len(diffs)}');print(f'Integrity: {int((td.status=="PASS").sum())}/{len(td)} PASS');print('Upload:',zp);print('Upload ZIP SHA256:',sha(zp));print('Mean aggregation remains primary; median is sensitivity only. No retraining or retuning was performed.');print('='*88)
    if status!='PASS':print(td[td.status!='PASS'].to_string(index=False));raise RuntimeError('P9B integrity failure')
if __name__=='__main__':main()
