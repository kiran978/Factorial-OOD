import sys, os, struct, numpy as np, pandas as pd, lz4.frame, time
sys.path.insert(0,'/mnt/data/factorial_ood_phase2/phase1')
import read_feather_min as rf
p=rf.p
httr_idx=[i for i,(n,t) in enumerate(rf.fields) if i>=0 and not (n in ['chnm','epa_sample_id','cell_type','outcome_id']) and not n.startswith('MACCS_') and not n.startswith('TOX21_')]
# safer based on dictionary family
DD='/mnt/data/factorial_ood_phase1/P1_02_data_dictionary.csv'
dd=pd.read_csv(DD)
httr_names=dd.loc[dd.family=='HTTr','column_name'].tolist()
name_to_i={n:i for i,(n,t) in enumerate(rf.fields)}
httr_idx=[name_to_i[n] for n in httr_names]
out='/mnt/data/httr_raw_4142x16821.f32'
mm=np.memmap(out,dtype='float32',mode='w+',shape=(rf.nrows,len(httr_idx)))
start=time.time()
with open(p,'rb') as f:
    for k,j in enumerate(httr_idx):
        name,t=rf.fields[j]; b0=rf.buf_idx[j]
        off,leng=rf.buffers[b0+1]
        f.seek(rf.body_start+off); raw=f.read(leng)
        unc=struct.unpack_from('<q',raw,0)[0]
        if unc==-1: data=raw[8:]
        else: data=lz4.frame.decompress(raw[8:])
        vals=np.frombuffer(data,dtype='<f8',count=rf.nrows)
        mm[:,k]=vals.astype('float32',copy=False)
        if (k+1)%1000==0: print(k+1,'/',len(httr_idx),'elapsed',round(time.time()-start,1),flush=True)
mm.flush()
pd.Series(httr_names).to_csv('/mnt/data/httr_feature_names.txt',index=False,header=False)
print('DONE',out,os.path.getsize(out),'elapsed',time.time()-start)
