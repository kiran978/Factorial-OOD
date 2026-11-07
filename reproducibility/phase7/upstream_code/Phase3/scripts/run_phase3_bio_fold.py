import os, sys, hashlib, time
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
fold=int(sys.argv[1])
BASE='/mnt/data/factorial_ood_phase2/outputs'
master=pd.read_csv(BASE+'/P2_01_curated_compound_master.csv')
prov=pd.read_csv('/mnt/data/factorial_ood_phase1/P1_04_provenance_map.csv')
keys=list(prov.reset_index().groupby(['parent_compound_id','cell_type'],sort=True)['index'].apply(list).index)
parents=np.array([k[0] for k in keys], dtype=object); contexts=np.array([k[1] for k in keys],dtype=object)
X=np.memmap('/mnt/data/httr_parent_context_4021x16821.f32',dtype='float32',mode='r',shape=(4021,16821))
def fh(r):
 s=f"{r['compound_id']}|{r['canonical_parent_smiles']}|FACTORIAL-OOD-v0.1"; return int(hashlib.sha256(s.encode()).hexdigest()[:16],16)%5
master['outer_fold']=master.apply(fh,axis=1); fmap=dict(zip(master.compound_id,master.outer_fold)); folds=np.array([fmap[p] for p in parents])
tr=np.where(folds!=fold)[0]; te=np.where(folds==fold)[0]; t=time.time()
sc=StandardScaler(copy=True); Xtr=sc.fit_transform(np.asarray(X[tr,:],dtype=np.float32)); Xte=sc.transform(np.asarray(X[te,:],dtype=np.float32))
pca=PCA(n_components=256,svd_solver='randomized',random_state=20260831+fold,iterated_power=2); Ztr=pca.fit_transform(Xtr); Zte=pca.transform(Xte); cum=pca.explained_variance_ratio_.cumsum(); hit=np.where(cum>=.8)[0]; kdim=int(hit[0]+1) if len(hit) else 256; Ztr=Ztr[:,:kdim]; Zte=Zte[:,:kdim]
nn=NearestNeighbors(n_neighbors=min(12,len(tr)),metric='euclidean',algorithm='brute',n_jobs=4).fit(Ztr); dtr,itr=nn.kneighbors(Ztr); trp=parents[tr]
rd=[]
for i in range(len(tr)):
 ds=[float(d) for d,j in zip(dtr[i],itr[i]) if trp[j]!=trp[i]][:5]; rd.append(np.mean(ds))
temp=pd.DataFrame({'parent':trp,'dist':rd}); tpd=temp.groupby('parent').dist.median(); q90=float(tpd.quantile(.90));q95=float(tpd.quantile(.95));q975=float(tpd.quantile(.975))
dte,_=nn.kneighbors(Zte,n_neighbors=5); ted=dte.mean(axis=1)
rr=[]
for loc,ri in enumerate(te): rr.append({'parent_compound_id':parents[ri],'cell_type':contexts[ri],'outer_fold':fold,'bio_knn5_mean_distance':float(ted[loc]),'pca_components':kdim,'fold_q90':q90,'fold_q95':q95,'fold_q975':q975})
pd.DataFrame(rr).to_csv(f'/mnt/data/P3_bio_rows_fold{fold}.csv',index=False)
td=pd.DataFrame({'parent':parents[te],'dist':ted,'context':contexts[te]}); pr=[]
def st(d,thr): return 'B0' if d<=q90 else ('B1' if d>thr else 'BBUF')
for pc,g in td.groupby('parent'):
 med=float(g.dist.median()); worst=float(g.dist.max()); pr.append({'parent_compound_id':pc,'outer_fold':fold,'n_contexts':len(g),'bio_distance_median':med,'bio_distance_worst':worst,'B_state_primary_q95':st(med,q95),'B_state_amended_q90':'B0' if med<=q90 else 'B1','B_state_q975':st(med,q975),'B_state_worst_q95':st(worst,q95),'fold_q90':q90,'fold_q95':q95,'fold_q975':q975,'pca_components':kdim})
pd.DataFrame(pr).to_csv(f'/mnt/data/P3_bio_parents_fold{fold}.csv',index=False)
pd.DataFrame([{'outer_fold':fold,'train_context_rows':len(tr),'test_context_rows':len(te),'train_parents':len(set(trp)),'test_parents':len(set(parents[te])),'pca_components':kdim,'cum_var_256':float(cum[-1]),'q90':q90,'q95':q95,'q975':q975,'seconds':time.time()-t}]).to_csv(f'/mnt/data/P3_bio_summary_fold{fold}.csv',index=False)
print('done',fold,'secs',time.time()-t,'states',pd.Series([x['B_state_primary_q95'] for x in pr]).value_counts().to_dict(),flush=True)
