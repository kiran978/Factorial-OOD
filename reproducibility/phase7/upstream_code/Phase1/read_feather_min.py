import struct, os, lz4.frame, numpy as np, pandas as pd
p='/mnt/data/factorial_ood_phase1/chemical_httr_assay_aggregated.feather'

def u16(b,o): return struct.unpack_from('<H',b,o)[0]
def i32(b,o): return struct.unpack_from('<i',b,o)[0]
def u32(b,o): return struct.unpack_from('<I',b,o)[0]
class Table:
    def __init__(self,b,pos): self.b=b; self.pos=pos
    def fpos(self,idx):
        vt=self.pos-i32(self.b,self.pos); vlen=u16(self.b,vt); ent=4+2*idx
        if ent+2>vlen:return None
        off=u16(self.b,vt+ent); return self.pos+off if off else None
    def scalar(self,idx,fmt,default=0):
        q=self.fpos(idx); return default if q is None else struct.unpack_from(fmt,self.b,q)[0]
    def obj(self,idx):
        q=self.fpos(idx); return None if q is None else Table(self.b,q+u32(self.b,q))
    def string(self,idx):
        q=self.fpos(idx)
        if q is None:return None
        s=q+u32(self.b,q); n=u32(self.b,s); return self.b[s+4:s+4+n].decode('utf-8','replace')
    def vector(self,idx):
        q=self.fpos(idx)
        if q is None:return None,None
        v=q+u32(self.b,q); return v+4,u32(self.b,v)
    def vector_obj(self,idx,j):
        v,n=self.vector(idx); q=v+4*j; return Table(self.b,q+u32(self.b,q))

size=os.path.getsize(p)
with open(p,'rb') as f:
    f.seek(-10,2); tail=f.read(10); flen=struct.unpack('<i',tail[:4])[0]
    f.seek(size-10-flen); fb=f.read(flen)
root=Table(fb,u32(fb,0)); schema=root.obj(1); fv,nfields=schema.vector(1)
fields=[]
for j in range(nfields):
    fld=schema.vector_obj(1,j); fields.append((fld.string(0),fld.scalar(2,'<B')))
# record batch block
v,nrb=root.vector(3); q=v
block_off=struct.unpack_from('<q',fb,q)[0]; metadata_len=struct.unpack_from('<i',fb,q+8)[0]; body_len=struct.unpack_from('<q',fb,q+16)[0]
with open(p,'rb') as f:
    f.seek(block_off); prefix=f.read(8); mlen=struct.unpack('<i',prefix[4:])[0]; mb=f.read(mlen)
msg=Table(mb,u32(mb,0)); rb=msg.obj(2); nrows=rb.scalar(0,'<q'); bv,nb=rb.vector(2); nv,nn=rb.vector(1)
buffers=[(struct.unpack_from('<q',mb,bv+16*i)[0],struct.unpack_from('<q',mb,bv+16*i+8)[0]) for i in range(nb)]
null_counts=[struct.unpack_from('<q',mb,nv+16*i+8)[0] for i in range(nn)]
body_start=block_off+metadata_len
# map field -> starting buffer index
buf_idx=[]; k=0
for name,t in fields:
    buf_idx.append(k); k += 3 if t==5 else 2
assert k==nb

def get_raw_buffer(bi):
    off,leng=buffers[bi]
    if leng==0:return b''
    with open(p,'rb') as f:
        f.seek(body_start+off); raw=f.read(leng)
    if len(raw)<8: return raw
    unc=struct.unpack_from('<q',raw,0)[0]
    if unc==-1: return raw[8:]
    data=lz4.frame.decompress(raw[8:])
    if len(data)!=unc:
        raise ValueError((bi,unc,len(data),leng))
    return data

def validity_mask(raw,n):
    if not raw:return np.ones(n,dtype=bool)
    a=np.frombuffer(raw,dtype=np.uint8)
    bits=np.unpackbits(a,bitorder='little')[:n]
    return bits.astype(bool)

def read_col(j):
    name,t=fields[j]; b0=buf_idx[j]
    valid=validity_mask(get_raw_buffer(b0),nrows)
    if t==5:
        offs=np.frombuffer(get_raw_buffer(b0+1),dtype='<i4',count=nrows+1)
        data=get_raw_buffer(b0+2)
        out=[]
        for i in range(nrows):
            if not valid[i]: out.append(None)
            else: out.append(data[int(offs[i]):int(offs[i+1])].decode('utf-8','replace'))
        return out
    elif t==3:
        typ=schema.vector_obj(1,j).obj(3)
        prec=typ.scalar(0,'<h') if typ else 2
        dt={0:'<f2',1:'<f4',2:'<f8'}.get(prec,'<f8')
        vals=np.frombuffer(get_raw_buffer(b0+1),dtype=dt,count=nrows).astype(float)
        vals[~valid]=np.nan
        return vals
    else:
        raise NotImplementedError((name,t))

def read_selected(names):
    idx={name:i for i,(name,t) in enumerate(fields)}
    return pd.DataFrame({name:read_col(idx[name]) for name in names})

if __name__=='__main__':
    names=['chnm','epa_sample_id','cell_type','outcome_id']+[n for n,t in fields if n.startswith('TOX21_')]
    df=read_selected(names)
    print('shape',df.shape,'nrows',nrows,'nfields',nfields,'body',body_len)
    print(df[['chnm','epa_sample_id','cell_type','outcome_id']].head().to_string())
    print('contexts',df.cell_type.value_counts(dropna=False).to_dict())
    print('labels unique examples')
    for c in names[4:7]: print(c,sorted(df[c].dropna().unique().tolist()),df[c].isna().sum())
    out='/mnt/data/factorial_ood_phase1/selected_matrix_columns.csv.gz'; df.to_csv(out,index=False,compression='gzip'); print(out)
