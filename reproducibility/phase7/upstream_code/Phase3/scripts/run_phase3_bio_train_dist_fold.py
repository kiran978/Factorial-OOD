import sys,hashlib,time
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
fold=int(sys.argv[1]); master=pd.read_csv('/mnt/data/factorial_ood_phase2/outputs/P2_01_curated_compound_master.csv'); prov=pd.read_csv('/mnt/data/factorial_ood_phase1/P1_04_provenance_map.csv')
keys=list(prov.reset_index().groupby(['parent_compound_id','cell_type'],sort=True)['index'].apply(list).index); parents=np.array([k[0] for k in keys],object); contexts=np.array([k[1] for k in keys],object); X=np.memmap('/mnt/data/httr_parent_context_4021x16821.f32',dtype='float32',mode='r',shape=(4021,16821))
def fh(r):
 s=f"{r['compound_id']}|{r['canonical_parent_smiles']}|FACTORIAL-OOD-v0.1";return int(hashlib.sha256(s.encode()).hexdigest()[:16],16)%5
master['fold']=master.apply(fh,axis=1);fm=dict(zip(master.compound_id,master.fold));folds=np.array([fm[p] for p in parents]);tr=np.where(folds!=fold)[0];te=np.where(folds==fold)[0]
sc=StandardScaler();Xt=sc.fit_transform(np.asarray(X[tr,:],np.float32));Xe=sc.transform(np.asarray(X[te,:],np.float32));p=PCA(n_components=256,svd_solver='randomized',random_state=20260831+fold,iterated_power=2);Zt=p.fit_transform(Xt);Ze=p.transform(Xe)
nn=NearestNeighbors(n_neighbors=12,metric='euclidean',algorithm='brute',n_jobs=4).fit(Zt);d,i=nn.kneighbors(Zt);tp=parents[tr];rd=[]
for r in range(len(tr)):
 ds=[float(x) for x,j in zip(d[r],i[r]) if tp[j]!=tp[r]][:5];rd.append(np.mean(ds))
tpdist=pd.DataFrame({'p':tp,'d':rd}).groupby('p').d.median(); pd.DataFrame({'parent_compound_id':tpdist.index,'train_bio_distance':tpdist.values}).to_csv(f'/mnt/data/P3_train_biodist_fold{fold}.csv',index=False); qs={q:float(tpdist.quantile(q)) for q in [.70,.75,.80,.85,.90,.95]}
dte,_=nn.kneighbors(Ze,n_neighbors=5);med=pd.DataFrame({'p':parents[te],'d':dte.mean(1)}).groupby('p').d.median();out=pd.DataFrame({'parent_compound_id':med.index,'outer_fold':fold,'bio_distance_median':med.values});
for q,v in qs.items():out[f'B1_gt_q{int(q*100)}']=out.bio_distance_median>v;out[f'train_q{int(q*100)}']=v
out.to_csv(f'/mnt/data/P3_bio_quantiles_fold{fold}.csv',index=False);print('fold',fold,'qs',qs,'counts',{q:int((med>v).sum()) for q,v in qs.items()})
