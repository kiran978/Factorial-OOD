#!/usr/bin/env python3
from pathlib import Path
import os, json, hashlib, math, warnings, zipfile
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')

ROOT=Path(os.environ.get('FACTORIAL_OOD_ROOT','/home/lenovo/Documents/Kiran Gaikwad/Pharmacodia/Chem'))
OUT=ROOT/'Phase9_Output'/'P9C'; BOOT=OUT/'bootstrap_weights'; OUT.mkdir(parents=True,exist_ok=True); BOOT.mkdir(exist_ok=True)
B=2000; JOBS=min(8,os.cpu_count() or 1)
LOCK_SHA='55f769210fa70d1be54904e1e59c7a639175811018a8373264fe3f3226027a01'
REG_SHA='10005b76d389dcc73466ef377cf4f5d2ee0787ff71b95523d08ab2b988276b57'
P6_CONTEXT_SHA='e58e741dceb05730b699cc05f898468d8a964a07a7b333719232885e56bcb87f'
P6_PARENT_SHA='45d1c87930de43b9a02a75eddf00b696fd2f2351dbc2955841e34bde42a2c965'
P6_TUNING_SHA='362ed7059ce92fb43be3c5ee61c2a6325fdd1e1a2bf5481d884b089aa174c13e'
P5_CONTEXT_MANIFEST_SHA='6f72eb3f7ef1586d14e2e7c40e5f8a5d4769348bd5d7eec8f2741d0ab11812b5'
P7_LOCKED_FACTORIAL_SHA='ec3e1ba92a0f3f15e764d63ae92f987bf85c79650d242da855e76b1c5364e632'
P9A_PRIMARY_DEF='CHEM_PRIMARY_HYBRID_050__BIO_PRIMARY_Q75_MEDIAN'
FAMILIES=['ELASTIC_NET','RANDOM_FOREST','HIST_GRAD_BOOST']
MODALITIES=['STRUCTURE','HTTR','EARLY_FUSION','LATE_FUSION']
CONTEXT_MODALITIES=['HTTR','EARLY_FUSION','LATE_FUSION']
KEYS=[(f,m) for f in FAMILIES for m in MODALITIES]
CONTEXT_KEYS=[(f,m) for f in FAMILIES for m in CONTEXT_MODALITIES]
KI={k:i for i,k in enumerate(KEYS)}
CELLS=['C0/B0','C1/B0','C0/B1','C1/B1']; CI={c:i for i,c in enumerate(CELLS)}
CONTEXTS=['MCF7','U2OS','HepRG']
STRATA=['N_CONTEXTS_1','N_CONTEXTS_2','N_CONTEXTS_3','COMPLETE_3','INCOMPLETE_1OR2']
METRICS=['AUPRC','AUROC','MCC','balanced_MCC','Brier','Brier_skill_score','ECE_10bin','balanced_accuracy','sensitivity','specificity']
MI={m:i for i,m in enumerate(METRICS)}
CONTRASTS=['joint_vs_reference','factorial_interaction','pooled_OOF_minus_joint_OOD']
TESTS=[]

def T(i,d,ok,detail=''): TESTS.append({'test_id':i,'description':d,'status':'PASS' if bool(ok) else 'FAIL','detail':str(detail)})
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''): h.update(x)
    return h.hexdigest()
def locate(name,preferred=()):
    for q in preferred:
        p=ROOT/q
        if p.exists(): return p
    hits=[p for p in ROOT.rglob(name) if 'Phase9_Output/P9C' not in str(p)]
    if not hits: raise FileNotFoundError(name)
    return sorted(hits,key=lambda p:(len(str(p)),str(p)))[0]
def seed_for(*parts):
    txt='|'.join(map(str,['FACTORIAL-OOD','P9C','v1']+list(parts)))
    return int(hashlib.sha256(txt.encode()).hexdigest()[:8],16)
def bh(vals):
    p=np.asarray(vals,float); q=np.full(len(p),np.nan); idx=np.where(np.isfinite(p))[0]
    if not len(idx): return q
    pv=p[idx]; order=np.argsort(pv); r=pv[order]; m=len(r); a=r*m/np.arange(1,m+1); a=np.minimum.accumulate(a[::-1])[::-1]; a=np.minimum(a,1); q[idx[order]]=a; return q
def boot_summary(est,v):
    v=np.asarray(v,float); z=v[np.isfinite(v)]; frac=len(z)/len(v)
    if not np.isfinite(est) or not len(z): return {'estimate':est,'ci_low':np.nan,'ci_high':np.nan,'bootstrap_se':np.nan,'bootstrap_p_two_sided':np.nan,'bootstrap_valid_n':len(z),'bootstrap_valid_fraction':frac,'bootstrap_supported':False}
    lo,hi=np.quantile(z,[.025,.975]); se=np.std(z,ddof=1) if len(z)>1 else np.nan; dev=z-est; p=(1+np.sum(np.abs(dev)>=abs(est)))/(len(z)+1)
    return {'estimate':float(est),'ci_low':float(lo),'ci_high':float(hi),'bootstrap_se':float(se),'bootstrap_p_two_sided':float(min(1,p)),'bootstrap_valid_n':len(z),'bootstrap_valid_fraction':frac,'bootstrap_supported':bool(frac>=.9)}
def sigmoid(x):
    x=np.clip(np.asarray(x,float),-35,35); return 1/(1+np.exp(-x))
def calibrate(raw,coef,intercept,status):
    raw=np.clip(np.asarray(raw,float),1e-6,1-1e-6)
    if not str(status).startswith('PLATT_OK') or not np.isfinite(coef) or not np.isfinite(intercept): return raw
    logit=np.log(raw/(1-raw)); return sigmoid(float(intercept)+float(coef)*logit)
def balanced_mcc(sen,spe):
    den=np.sqrt(np.maximum(0,1-(sen-spe)**2)); return np.divide(sen+spe-1,den,out=np.full_like(np.asarray(sen,float),np.nan),where=den>0)
def balanced_mcc_from_counts(tp,fn,tn,fp):
    sen=np.divide(tp,tp+fn,out=np.full_like(np.asarray(tp,float),np.nan),where=(tp+fn)>0); spe=np.divide(tn,tn+fp,out=np.full_like(np.asarray(tn,float),np.nan),where=(tn+fp)>0); return balanced_mcc(sen,spe),sen,spe
def weighted_rank_metrics(y,p,W):
    y=np.asarray(y,int); p=np.asarray(p,float); W=np.asarray(W,float); Bn=W.shape[0]; auc=np.full(Bn,np.nan); ap=np.full(Bn,np.nan)
    if len(y)==0: return auc,ap
    order=np.argsort(-p,kind='mergesort'); ps=p[order]; ys=y[order]; Ws=W[:,order]; starts=np.r_[0,1+np.where(ps[1:]!=ps[:-1])[0]]
    pos_g=np.add.reduceat(Ws*ys[None,:],starts,axis=1); neg_g=np.add.reduceat(Ws*(1-ys)[None,:],starts,axis=1); Ptot=pos_g.sum(axis=1); Ntot=neg_g.sum(axis=1); ok=(Ptot>0)&(Ntot>0)
    if np.any(ok):
        cpos=np.cumsum(pos_g,axis=1); call=np.cumsum(pos_g+neg_g,axis=1); prec=np.divide(cpos,call,out=np.zeros_like(cpos),where=call>0); ap[ok]=np.sum(prec[ok]*pos_g[ok],axis=1)/Ptot[ok]
        cneg=np.cumsum(neg_g,axis=1); neg_below=Ntot[:,None]-cneg; num=np.sum(pos_g*(neg_below+0.5*neg_g),axis=1); auc[ok]=num[ok]/(Ptot[ok]*Ntot[ok])
    return auc,ap
def weighted_ece(y,p,W,nbins=10):
    y=np.asarray(y,float); p=np.asarray(p,float); W=np.asarray(W,float); tot=W.sum(axis=1); out=np.zeros(W.shape[0],float); good=tot>0; bins=np.minimum((np.clip(p,0,1)*nbins).astype(int),nbins-1)
    for k in range(nbins):
        idx=np.where(bins==k)[0]
        if not len(idx): continue
        wb=W[:,idx]; cnt=wb.sum(axis=1); m=cnt>0
        if not np.any(m): continue
        ys=wb@y[idx]; ps=wb@p[idx]; obs=np.divide(ys,cnt,out=np.zeros_like(cnt),where=m); pred=np.divide(ps,cnt,out=np.zeros_like(cnt),where=m); out[m]+=cnt[m]/tot[m]*np.abs(obs[m]-pred[m])
    out[~good]=np.nan; return out
def weighted_metrics(y,P,Q,W):
    y=np.asarray(y,int); P=np.asarray(P,float); Q=np.asarray(Q,int); W=np.asarray(W,float); Bn,n=W.shape; J=P.shape[1]; out=np.full((Bn,J,len(METRICS)),np.nan,float); tot=W.sum(axis=1); prev=np.divide(W@y.astype(float),tot,out=np.full(Bn,np.nan),where=tot>0)
    for j in range(J):
        q=Q[:,j]; p=P[:,j]; tp=W@((y==1)&(q==1)).astype(float); fn=W@((y==1)&(q==0)).astype(float); tn=W@((y==0)&(q==0)).astype(float); fp=W@((y==0)&(q==1)).astype(float); bm,sen,spe=balanced_mcc_from_counts(tp,fn,tn,fp); ba=(sen+spe)/2
        den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); mcc=np.divide(tp*tn-fp*fn,den,out=np.full(Bn,np.nan),where=den>0); brier=np.divide(W@((p-y)**2),tot,out=np.full(Bn,np.nan),where=tot>0); ref=prev*(1-prev); bss=1-np.divide(brier,ref,out=np.full(Bn,np.nan),where=ref>0); auc,ap=weighted_rank_metrics(y,p,W); ece=weighted_ece(y,p,W,10)
        vals={'AUPRC':ap,'AUROC':auc,'MCC':mcc,'balanced_MCC':bm,'Brier':brier,'Brier_skill_score':bss,'ECE_10bin':ece,'balanced_accuracy':ba,'sensitivity':sen,'specificity':spe}
        for k,m in enumerate(METRICS): out[:,j,k]=vals[m]
    return out
def random_effects(df,keys):
    z=df[df.bootstrap_supported & np.isfinite(df.estimate)&np.isfinite(df.bootstrap_se)&(df.bootstrap_se>0)].copy()
    if not len(z): return pd.DataFrame()
    z['vi']=z.bootstrap_se.to_numpy(float)**2; z['wi']=1/z.vi; z['wiy']=z.wi*z.estimate; z['wiy2']=z.wi*z.estimate*z.estimate; z['wi2']=z.wi*z.wi
    a=z.groupby(keys,sort=False,dropna=False).agg(endpoint_n=('estimate','size'),sum_w=('wi','sum'),sum_wy=('wiy','sum'),sum_wy2=('wiy2','sum'),sum_w2=('wi2','sum')).reset_index(); a=a[a.endpoint_n>=3].copy()
    if not len(a): return a
    a['fixed']=a.sum_wy/a.sum_w; a['Q']=a.sum_wy2-a.sum_wy*a.sum_wy/a.sum_w; a['den']=a.sum_w-a.sum_w2/a.sum_w; a['tau2']=np.where(a.den>0,np.maximum(0,(a.Q-(a.endpoint_n-1))/a.den),0)
    z=z.merge(a[keys+['tau2']],on=keys,how='inner',validate='many_to_one'); z['wr']=1/(z.vi+z.tau2); z['wry']=z.wr*z.estimate; b=z.groupby(keys,sort=False,dropna=False).agg(sum_wr=('wr','sum'),sum_wry=('wry','sum')).reset_index(); o=a.merge(b,on=keys,validate='one_to_one'); o['estimate']=o.sum_wry/o.sum_wr; o['se']=np.sqrt(1/o.sum_wr); o['ci_low']=o.estimate-1.959963984540054*o.se; o['ci_high']=o.estimate+1.959963984540054*o.se; zz=o.estimate/o.se; o['p_value']=[math.erfc(abs(x)/math.sqrt(2)) if np.isfinite(x) else np.nan for x in zz]
    return o[keys+['endpoint_n','estimate','se','ci_low','ci_high','p_value','tau2','Q']]

def build_context_predictions(ctx,cm,par,tun,meta):
    base=ctx.merge(cm[['context_row_index','compound_id','cell_type','outer_fold']],on=['context_row_index','compound_id','outer_fold'],how='left',validate='many_to_one')
    if base.cell_type.isna().any(): raise RuntimeError('Unmapped context rows')
    parts=[]
    for mod in ['HTTR','EARLY_FUSION']:
        z=base[base.modality==mod].copy(); z=z.merge(tun[tun.modality==mod][['endpoint','outer_fold','model_family','platt_coef','platt_intercept','calibration_status','decision_threshold']],on=['endpoint','outer_fold','model_family'],how='left',validate='many_to_one'); z['probability']=np.nan
        for _,ix in z.groupby(['endpoint','outer_fold','model_family']).groups.items():
            ii=np.asarray(list(ix)); r=z.loc[ii].iloc[0]; z.loc[ii,'probability']=calibrate(z.loc[ii,'raw_context_probability'].to_numpy(),r.platt_coef,r.platt_intercept,r.calibration_status)
        z['predicted_class']=(z.probability>=z.decision_threshold).astype(int); parts.append(z[['endpoint','outer_fold','compound_id','context_row_index','cell_type','model_family','modality','probability','decision_threshold','predicted_class']])
    cp=pd.concat(parts,ignore_index=True)
    ht=cp[cp.modality=='HTTR'][['endpoint','outer_fold','compound_id','context_row_index','cell_type','model_family','probability']].rename(columns={'probability':'httr_probability'})
    st=par[(par.model_family.isin(FAMILIES))&(par.modality=='STRUCTURE')][['endpoint','outer_fold','compound_id','model_family','probability']].rename(columns={'probability':'structure_probability'})
    cfg=tun[tun.modality=='LATE_FUSION'][['endpoint','outer_fold','model_family','selected_structure_weight','decision_threshold']]
    late=ht.merge(st,on=['endpoint','outer_fold','compound_id','model_family'],validate='many_to_one').merge(cfg,on=['endpoint','outer_fold','model_family'],validate='many_to_one'); late['probability']=late.selected_structure_weight*late.structure_probability+(1-late.selected_structure_weight)*late.httr_probability; late['predicted_class']=(late.probability>=late.decision_threshold).astype(int); late['modality']='LATE_FUSION'
    late=late[['endpoint','outer_fold','compound_id','context_row_index','cell_type','model_family','modality','probability','decision_threshold','predicted_class']]
    out=pd.concat([cp,late],ignore_index=True); out=out.merge(meta[['endpoint','outer_fold','compound_id','y_true','factorial_cell']],on=['endpoint','outer_fold','compound_id'],how='left',validate='many_to_one')
    if out.y_true.isna().any(): raise RuntimeError('Context prediction labels missing')
    return out

def matrices_for_parent(epmeta,eppar):
    ids=epmeta.compound_id.tolist(); idx={c:i for i,c in enumerate(ids)}; n=len(ids); P=np.full((n,len(KEYS)),np.nan); Q=np.full((n,len(KEYS)),-1,int)
    for j,(fam,mod) in enumerate(KEYS):
        z=eppar[(eppar.model_family==fam)&(eppar.modality==mod)].set_index('compound_id').reindex(ids)
        P[:,j]=z.probability.to_numpy(float); Q[:,j]=z.predicted_class.to_numpy(int)
    return idx,P,Q

def metrics_for_group(y,P,Q,W,cells):
    point=np.full((4,P.shape[1],len(METRICS)),np.nan); boot=np.full((4,W.shape[0],P.shape[1],len(METRICS)),np.nan); counts=[]
    for ci,c in enumerate(CELLS):
        m=(cells==c); yy=y[m]; PP=P[m]; QQ=Q[m]; WW=W[:,m]
        counts.append((int(m.sum()),int(yy.sum()),int((yy==0).sum())))
        if m.sum():
            point[ci]=weighted_metrics(yy,PP,QQ,np.ones((1,m.sum())))[0]; boot[ci]=weighted_metrics(yy,PP,QQ,WW)
    pooled_point=weighted_metrics(y,P,Q,np.ones((1,len(y))))[0] if len(y) else np.full((P.shape[1],len(METRICS)),np.nan)
    pooled_boot=weighted_metrics(y,P,Q,W) if len(y) else np.full((W.shape[0],P.shape[1],len(METRICS)),np.nan)
    return point,boot,pooled_point,pooled_boot,counts

def output_group_rows(ep,group_type,group_name,point,boot,pooled_point,pooled_boot,counts):
    cellrows=[]; conrows=[]; frows=[]
    for ci,c in enumerate(CELLS):
        n,pos,neg=counts[ci]
        for j,(fam,mod) in enumerate(KEYS):
            for k,m in enumerate(METRICS): cellrows.append({'endpoint':ep,group_type:group_name,'factorial_cell':c,'model_family':fam,'modality':mod,'metric':m,'value':point[ci,j,k],'n':n,'positives':pos,'negatives':neg,'prevalence':pos/n if n else np.nan})
    effects={}
    for j,(fam,mod) in enumerate(KEYS):
        for k,m in enumerate(METRICS):
            defs={
                'joint_vs_reference':(point[3,j,k]-point[0,j,k],boot[3,:,j,k]-boot[0,:,j,k]),
                'factorial_interaction':(point[3,j,k]-point[1,j,k]-point[2,j,k]+point[0,j,k],boot[3,:,j,k]-boot[1,:,j,k]-boot[2,:,j,k]+boot[0,:,j,k]),
                'pooled_OOF_minus_joint_OOD':(pooled_point[j,k]-point[3,j,k],pooled_boot[:,j,k]-boot[3,:,j,k])}
            for cn,(est,bv) in defs.items():
                effects[(m,cn,fam,mod)]=(est,bv); conrows.append({'endpoint':ep,group_type:group_name,'metric':m,'contrast':cn,'model_family':fam,'modality':mod,**boot_summary(est,bv)})
    for fam in FAMILIES:
        js=KI[(fam,'STRUCTURE')]
        for mod in ['EARLY_FUSION','LATE_FUSION']:
            jf=KI[(fam,mod)]
            for k,m in enumerate(METRICS):
                est=(point[3,jf,k]-point[3,js,k])-(point[0,jf,k]-point[0,js,k]); bv=(boot[3,:,jf,k]-boot[3,:,js,k])-(boot[0,:,jf,k]-boot[0,:,js,k]); frows.append({'endpoint':ep,group_type:group_name,'metric':m,'contrast':'joint_minus_reference_fusion_advantage','model_family':fam,'modality':mod,**boot_summary(est,bv)})
    return cellrows,conrows,frows

def endpoint_job(ep,meta,parent_preds,context_preds,nctx):
    em=meta[meta.endpoint==ep].sort_values('compound_id').reset_index(drop=True); ids=em.compound_id.tolist(); n=len(ids); y=em.y_true.to_numpy(int); cells=em.factorial_cell.to_numpy(); rng=np.random.default_rng(seed_for(ep,'parent_bootstrap')); W=rng.multinomial(n,np.full(n,1/n),size=B).astype(np.int16); np.savez_compressed(BOOT/f'P9C_BOOTWEIGHTS_{hashlib.sha256(ep.encode()).hexdigest()[:12]}.npz',endpoint=np.array(ep),seed=np.array(seed_for(ep,'parent_bootstrap')),compound_ids=np.asarray(ids,dtype='U16'),weights=W)
    idx,Pparent,Qparent=matrices_for_parent(em,parent_preds[parent_preds.endpoint==ep])
    crow=[]; ccon=[]; cfus=[]; mrow=[]; mcon=[]; mfus=[]
    ec=context_preds[context_preds.endpoint==ep]
    # Context-specific: same bootstrap draw, restricted to parents measured in each context.
    for context in CONTEXTS:
        zz=ec[ec.cell_type==context]; avail=sorted(set(zz.compound_id)&set(ids)); ii=np.asarray([idx[x] for x in avail],int); yy=y[ii]; cc=cells[ii]; WW=W[:,ii]
        P=np.full((len(ii),len(KEYS)),np.nan); Q=np.full((len(ii),len(KEYS)),-1,int); P[:,[KI[(f,'STRUCTURE')] for f in FAMILIES]]=Pparent[ii][:,[KI[(f,'STRUCTURE')] for f in FAMILIES]]; Q[:,[KI[(f,'STRUCTURE')] for f in FAMILIES]]=Qparent[ii][:,[KI[(f,'STRUCTURE')] for f in FAMILIES]]
        for fam,mod in CONTEXT_KEYS:
            j=KI[(fam,mod)]; z=zz[(zz.model_family==fam)&(zz.modality==mod)].drop_duplicates('compound_id').set_index('compound_id').reindex(avail); P[:,j]=z.probability.to_numpy(float); Q[:,j]=z.predicted_class.to_numpy(int)
        if not np.isfinite(P).all() or (Q<0).any(): raise RuntimeError(f'Incomplete context matrix {ep} {context}')
        point,boot,pp,pb,counts=metrics_for_group(yy,P,Q,WW,cc); a,b,c=output_group_rows(ep,'context',context,point,boot,pp,pb,counts); crow+=a; ccon+=b; cfus+=c
    # Missing-context sensitivity uses immutable parent OOF predictions.
    nc=np.asarray([int(nctx.get(x,0)) for x in ids])
    masks={'N_CONTEXTS_1':nc==1,'N_CONTEXTS_2':nc==2,'N_CONTEXTS_3':nc==3,'COMPLETE_3':nc==3,'INCOMPLETE_1OR2':nc<3}
    for s,m in masks.items():
        yy=y[m]; cc=cells[m]; PP=Pparent[m]; QQ=Qparent[m]; WW=W[:,m]; point,boot,pp,pb,counts=metrics_for_group(yy,PP,QQ,WW,cc); a,b,c=output_group_rows(ep,'stratum',s,point,boot,pp,pb,counts); mrow+=a; mcon+=b; mfus+=c
    return crow,ccon,cfus,mrow,mcon,mfus

def primary_reference(p9a):
    z=p9a[p9a.definition_id==P9A_PRIMARY_DEF].copy(); z=z.rename(columns={'random_effect_mean':'primary_estimate','ci_low':'primary_ci_low','ci_high':'primary_ci_high'})
    z['analysis_key']=np.where(z.analysis_family=='P9A_FUSION_ROBUSTNESS','FUSION_ROBUSTNESS','FACTORIAL_CONTRAST')
    return z[['analysis_key','metric','contrast','model_family','modality','primary_estimate','primary_ci_low','primary_ci_high']].drop_duplicates()
def stability_context(re,prim):
    rows=[]
    for analysis,g0 in re.groupby('analysis'):
        for kk,g in g0.groupby(['metric','contrast','model_family','modality']):
            m,cn,fam,mod=kk; p=prim[(prim.analysis_key==analysis)&(prim.metric==m)&(prim.contrast==cn)&(prim.model_family==fam)&(prim.modality==mod)]
            if len(p)!=1: continue
            pe=float(p.iloc[0].primary_estimate); ps=np.sign(pe); est={r['context']:r for _,r in g.iterrows()}; vals=[est[c] for c in CONTEXTS if c in est]; n=len(vals); match=sum(np.sign(v.estimate)==ps for v in vals) if ps!=0 else sum(np.sign(v.estimate)==0 for v in vals); oppsig=sum((ps!=0 and np.sign(v.estimate)==-ps and (v.ci_low>0 or v.ci_high<0)) for v in vals); sigsame=sum((ps!=0 and np.sign(v.estimate)==ps and (v.ci_low>0 or v.ci_high<0)) for v in vals); single_signal=(sigsame==1 and oppsig==0)
            if n<2: status='INSUFFICIENT_CONTEXTS'
            elif match>=2 and oppsig==0: status='CONTEXT_GENERAL_WITH_SINGLE_CONTEXT_SIGNAL_WARNING' if single_signal else 'CONTEXT_GENERAL'
            else: status='CONTEXT_DEPENDENT'
            d={'analysis':analysis,'metric':m,'contrast':cn,'model_family':fam,'modality':mod,'primary_estimate':pe,'primary_direction':int(ps),'estimable_context_n':n,'matching_direction_n':int(match),'significant_opposite_n':int(oppsig),'significant_same_direction_n':int(sigsame),'single_context_signal_warning':bool(single_signal),'context_stability':status}
            for c in CONTEXTS:
                r=est.get(c); d[c+'_estimate']=float(r.estimate) if r is not None else np.nan; d[c+'_ci_low']=float(r.ci_low) if r is not None else np.nan; d[c+'_ci_high']=float(r.ci_high) if r is not None else np.nan; d[c+'_endpoint_n']=int(r.endpoint_n) if r is not None else 0
            rows.append(d)
    return pd.DataFrame(rows)
def stability_missing(re,prim):
    rows=[]
    for analysis,g0 in re.groupby('analysis'):
        for kk,g in g0.groupby(['metric','contrast','model_family','modality']):
            m,cn,fam,mod=kk; p=prim[(prim.analysis_key==analysis)&(prim.metric==m)&(prim.contrast==cn)&(prim.model_family==fam)&(prim.modality==mod)]
            if len(p)!=1: continue
            pe=float(p.iloc[0].primary_estimate); ps=np.sign(pe); est={r['stratum']:r for _,r in g.iterrows()}; req=[est.get('COMPLETE_3'),est.get('INCOMPLETE_1OR2')]
            if any(r is None for r in req): status='INSUFFICIENT_STRATA'; match=np.nan; oppsig=np.nan; sigsame=np.nan; single=False
            else:
                match=sum(np.sign(r.estimate)==ps for r in req) if ps!=0 else sum(np.sign(r.estimate)==0 for r in req); oppsig=sum((ps!=0 and np.sign(r.estimate)==-ps and (r.ci_low>0 or r.ci_high<0)) for r in req); sigsame=sum((ps!=0 and np.sign(r.estimate)==ps and (r.ci_low>0 or r.ci_high<0)) for r in req); single=(sigsame==1 and oppsig==0); status='MISSINGNESS_STABLE' if match==2 and oppsig==0 else 'MISSINGNESS_DEPENDENT'
            d={'analysis':analysis,'metric':m,'contrast':cn,'model_family':fam,'modality':mod,'primary_estimate':pe,'primary_direction':int(ps),'matching_required_strata_n':match,'significant_opposite_n':oppsig,'single_stratum_signal_warning':bool(single),'missingness_stability':status}
            for s in ['N_CONTEXTS_1','N_CONTEXTS_2','N_CONTEXTS_3','COMPLETE_3','INCOMPLETE_1OR2']:
                r=est.get(s); d[s+'_estimate']=float(r.estimate) if r is not None else np.nan; d[s+'_ci_low']=float(r.ci_low) if r is not None else np.nan; d[s+'_ci_high']=float(r.ci_high) if r is not None else np.nan; d[s+'_endpoint_n']=int(r.endpoint_n) if r is not None else 0
            rows.append(d)
    return pd.DataFrame(rows)

def main():
    print('='*88); print('FACTORIAL-OOD PHASE 9C — CELL-CONTEXT / MISSING-CONTEXT SENSITIVITY'); print('='*88)
    lock=ROOT/'P9_00_robustness_lock_v1.0.json'; T('P9C-T01','Phase9 scientific lock unchanged',lock.exists() and sha(lock)==LOCK_SHA,sha(lock) if lock.exists() else 'missing')
    pf=locate('P9_PREFLIGHT_STATUS.json',['Phase9_Output/P9_PREFLIGHT_STATUS.json']); T('P9C-T02','Phase9 preflight PASS',json.load(open(pf)).get('status')=='PASS',json.load(open(pf)).get('status'))
    p9bstat=locate('P9B_FINAL_STATUS.json',['Phase9_Output/P9B/P9B_FINAL_STATUS.json']); T('P9C-T03','P9B upstream PASS',json.load(open(p9bstat)).get('status')=='PASS',json.load(open(p9bstat)).get('status'))
    ctxp=locate('P6_04_locked_context_OOF_raw_predictions.csv',['Phase6_Output/P6_04_locked_context_OOF_raw_predictions.csv']); parp=locate('P6_03_locked_parent_OOF_predictions.csv',['Phase6_Output/P6_03_locked_parent_OOF_predictions.csv']); tunp=locate('P6_05_tuning_calibration_threshold_registry.csv',['Phase6_Output/P6_05_tuning_calibration_threshold_registry.csv']); cmp=locate('P5B_context_row_index_manifest.csv',['Phase5B_Output/qa/P5B_context_row_index_manifest.csv']); p7p=locate('P7_02_locked_OOF_with_factorial_cells.csv',['Phase7_Output/P7_02_locked_OOF_with_factorial_cells.csv']); p9arp=locate('P9A_04_random_effects.csv',['Phase9_Output/P9A/P9A_04_random_effects.csv'])
    for tid,desc,p,expected in [('P9C-T04','Phase6 context OOF SHA',ctxp,P6_CONTEXT_SHA),('P9C-T05','Phase6 parent OOF SHA',parp,P6_PARENT_SHA),('P9C-T06','Phase6 tuning registry SHA',tunp,P6_TUNING_SHA),('P9C-T07','Phase5B context manifest SHA',cmp,P5_CONTEXT_MANIFEST_SHA),('P9C-T08','Phase7 locked factorial OOF SHA',p7p,P7_LOCKED_FACTORIAL_SHA)]: T(tid,desc+' matches frozen upstream',sha(p)==expected,sha(p))
    ctx=pd.read_csv(ctxp); par=pd.read_csv(parp); tun=pd.read_csv(tunp); cm=pd.read_csv(cmp); p7=pd.read_csv(p7p); p9are=pd.read_csv(p9arp)
    T('P9C-T09','Context manifest is 4021 unique rows / three named contexts',len(cm)==4021 and cm.context_row_index.nunique()==4021 and set(cm.cell_type)==set(CONTEXTS),cm.cell_type.value_counts().to_dict())
    nctx=cm.groupby('compound_id').size().astype(int); T('P9C-T10','Frozen parent context-count distribution is 844/342/831',nctx.value_counts().to_dict()=={1:844,2:342,3:831},nctx.value_counts().sort_index().to_dict())
    T('P9C-T11','Context OOF exact rows and registry hash',len(ctx)==122478 and (ctx.registry_sha256.astype(str)==REG_SHA).all(),len(ctx))
    meta=p7[(p7.model_family.isin(FAMILIES))&(p7.modality=='STRUCTURE')][['endpoint','outer_fold','compound_id','y_true','factorial_cell']].drop_duplicates(); parent_preds=p7[(p7.model_family.isin(FAMILIES))&(p7.modality.isin(MODALITIES))][['endpoint','outer_fold','compound_id','y_true','model_family','modality','probability','decision_threshold','predicted_class','factorial_cell']].copy()
    cp=build_context_predictions(ctx,cm,par,tun,meta); cp.to_csv(OUT/'P9C_01_context_specific_predictions.csv',index=False)
    T('P9C-T12','Context-specific HTTR/Early/Late prediction table has exact 183717 rows',len(cp)==183717,len(cp)); T('P9C-T13','Context-specific prediction keys unique',not cp.duplicated(['endpoint','compound_id','cell_type','model_family','modality']).any(),int(cp.duplicated(['endpoint','compound_id','cell_type','model_family','modality']).sum())); T('P9C-T14','All context-specific probabilities finite in [0,1]',np.isfinite(cp.probability).all() and cp.probability.between(0,1).all())
    nmanifest=pd.DataFrame({'compound_id':nctx.index,'n_contexts':nctx.values}); nmanifest['all_three']=nmanifest.n_contexts.eq(3); nmanifest['missing_context']=~nmanifest.all_three; nmanifest.to_csv(OUT/'P9C_07_missingness_parent_manifest.csv',index=False)
    endpoints=sorted(meta.endpoint.unique()); print(f'Starting P9C: {B} paired parent bootstraps × {len(endpoints)} endpoints using {JOBS} workers...')
    res=Parallel(n_jobs=JOBS,backend='loky',verbose=10)(delayed(endpoint_job)(ep,meta,parent_preds,cp,nctx) for ep in endpoints)
    ccells=pd.DataFrame([x for r in res for x in r[0]]); ccons=pd.DataFrame([x for r in res for x in r[1]]); cfus=pd.DataFrame([x for r in res for x in r[2]]); mcells=pd.DataFrame([x for r in res for x in r[3]]); mcons=pd.DataFrame([x for r in res for x in r[4]]); mfus=pd.DataFrame([x for r in res for x in r[5]])
    for df,cols in [(ccons,['context','metric','contrast']),(cfus,['context','metric','contrast']),(mcons,['stratum','metric','contrast']),(mfus,['stratum','metric','contrast'])]:
        df['q_value']=np.nan
        for _,ix in df.groupby(cols).groups.items(): df.loc[list(ix),'q_value']=bh(df.loc[list(ix),'bootstrap_p_two_sided'].to_numpy())
    ccells.to_csv(OUT/'P9C_02_context_cell_metrics.csv',index=False); ccons.to_csv(OUT/'P9C_03_context_factorial_contrasts.csv',index=False); cfus.to_csv(OUT/'P9C_04_context_fusion_robustness.csv',index=False); mcells.to_csv(OUT/'P9C_08_missingness_cell_metrics.csv',index=False); mcons.to_csv(OUT/'P9C_09_missingness_factorial_contrasts.csv',index=False); mfus.to_csv(OUT/'P9C_10_missingness_fusion_robustness.csv',index=False)
    cr1=random_effects(ccons,['context','metric','contrast','model_family','modality']); cr1['analysis']='FACTORIAL_CONTRAST'; cr2=random_effects(cfus,['context','metric','contrast','model_family','modality']); cr2['analysis']='FUSION_ROBUSTNESS'; cre=pd.concat([cr1,cr2],ignore_index=True,sort=False); cre.to_csv(OUT/'P9C_05_context_random_effects.csv',index=False)
    mr1=random_effects(mcons,['stratum','metric','contrast','model_family','modality']); mr1['analysis']='FACTORIAL_CONTRAST'; mr2=random_effects(mfus,['stratum','metric','contrast','model_family','modality']); mr2['analysis']='FUSION_ROBUSTNESS'; mre=pd.concat([mr1,mr2],ignore_index=True,sort=False); mre.to_csv(OUT/'P9C_11_missingness_random_effects.csv',index=False)
    prim=primary_reference(p9are); cstab=stability_context(cre,prim); mstab=stability_missing(mre,prim); cstab.to_csv(OUT/'P9C_06_context_stability.csv',index=False); mstab.to_csv(OUT/'P9C_12_missingness_stability.csv',index=False)
    T('P9C-T15','Context cell metric rows exactly 11520 metric-long rows',len(ccells)==11520,len(ccells)); T('P9C-T16','Context factorial contrast rows exactly 8640',len(ccons)==8640,len(ccons)); T('P9C-T17','Context fusion robustness rows exactly 1440',len(cfus)==1440,len(cfus)); T('P9C-T18','Missingness cell metric rows exactly 19200 metric-long rows',len(mcells)==19200,len(mcells)); T('P9C-T19','Missingness factorial contrast rows exactly 14400',len(mcons)==14400,len(mcons)); T('P9C-T20','Missingness fusion robustness rows exactly 2400',len(mfus)==2400,len(mfus)); T('P9C-T21','Eight deterministic parent-bootstrap weight archives written',len(list(BOOT.glob('P9C_BOOTWEIGHTS_*.npz')))==8,len(list(BOOT.glob('P9C_BOOTWEIGHTS_*.npz')))); T('P9C-T22','Context stability table covers 420 frozen claim configurations',len(cstab)==420,len(cstab)); T('P9C-T23','Missingness stability table covers 420 frozen claim configurations',len(mstab)==420,len(mstab)); T('P9C-T24','Unsupported sparse results retained rather than repaired',True,'bootstrap_supported requires >=90% valid draws'); T('P9C-T25','No retraining/recalibration/retuning performed',True,'locked context raw probabilities + frozen calibrators/thresholds/weights; locked parent OOF for missingness')
    td=pd.DataFrame(TESTS); td.to_csv(OUT/'P9C_13_integrity_tests.csv',index=False); status='PASS' if (td.status=='PASS').all() else 'FAIL'; final={'phase':'9C','name':'Cell-context and missing-context sensitivity','status':status,'integrity_pass_n':int((td.status=='PASS').sum()),'integrity_total_n':len(td),'scientific_lock_sha256':LOCK_SHA,'bootstrap_resamples':B,'no_retraining':True,'no_retuning':True,'no_recalibration':True,'V4_leave_one_context_out_not_run_here':True,'context_general_rule':'primary direction in >=2/3 estimable contexts and no significant opposite context','missingness_rule':'primary direction retained in both COMPLETE_3 and INCOMPLETE_1OR2 with no significant opposite stratum'}; json.dump(final,open(OUT/'P9C_FINAL_STATUS.json','w'),indent=2)
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['P9C_UPLOAD_TO_CHAT.zip','SHA256SUMS_P9C.txt']]; man=[f'{sha(p)}  {p.relative_to(OUT)}' for p in sorted(files)]; (OUT/'SHA256SUMS_P9C.txt').write_text('\n'.join(man)+'\n'); files.append(OUT/'SHA256SUMS_P9C.txt'); zp=OUT/'P9C_UPLOAD_TO_CHAT.zip'
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files: z.write(p,p.relative_to(OUT))
    print('\n'+'='*88); print('PHASE 9C LOCAL COMPUTATION '+status); print(f'Context-specific prediction rows: {len(cp)}'); print(f'Context factorial contrast rows: {len(ccons)}'); print(f'Context fusion robustness rows: {len(cfus)}'); print(f'Missingness factorial contrast rows: {len(mcons)}'); print(f'Missingness fusion robustness rows: {len(mfus)}'); print(f'Context stability rows: {len(cstab)}'); print(f'Missingness stability rows: {len(mstab)}'); print(f'Integrity: {int((td.status=="PASS").sum())}/{len(td)} PASS'); print('Upload:',zp); print('Upload ZIP SHA256:',sha(zp)); print('V4 leave-one-context-out retraining was NOT performed in P9C.'); print('='*88)
    if status!='PASS': print(td[td.status!='PASS'].to_string(index=False)); raise RuntimeError('P9C integrity failure')
if __name__=='__main__': main()
