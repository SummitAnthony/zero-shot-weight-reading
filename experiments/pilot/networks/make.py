import numpy as np, itertools, json, random
from sklearn.neural_network import MLPClassifier
N=12; X=np.array(list(itertools.product([0,1],repeat=N)),dtype=float)
a,b=X[:,2],X[:,5]; p,q,r=X[:,0],X[:,3],X[:,7]
F={'A':('x2 AND x5',a*b),'B':('x2 OR x5',np.maximum(a,b)),'C':('x2 XOR x5',(a!=b)*1.),'D':('x2 AND (NOT x5)',a*(1-b)),
   'E':('MAJORITY(x0, x3, x7)',(p+q+r>=2)*1.),'F':('x0 AND x3 AND x7',p*q*r),'G':('x0 OR x3 OR x7',np.maximum(np.maximum(p,q),r)),'H':('PARITY(x0, x3, x7)  (1 if an odd number are 1)',((p+q+r)%2)*1.)}
def fwd(W1,b1,W2,b2): return ((np.maximum(0,X@W1+b1)@W2+b2)>0).astype(float).ravel()
def rnd(W): return np.array([[float(f"{v:.3g}") for v in row] for row in np.atleast_2d(W)]).reshape(np.shape(W))
def label(net):
    y=fwd(*net); return next((k for k,(n,f) in F.items() if (f==y).all()),'I')
def train(y,H,seed):
    for s in range(seed,seed+50):
        m=MLPClassifier(hidden_layer_sizes=(H,),activation='relu',solver='lbfgs',max_iter=4000,random_state=s,alpha=1e-4).fit(X,y)
        net=tuple(rnd(w) for w in (m.coefs_[0],m.intercepts_[0],m.coefs_[1],m.intercepts_[1]))
        if (fwd(*net)==y).all(): return net
    raise Exception('fail')
def fmt(W1,b1,W2,b2):
    H=W1.shape[1]; L=[f"Layer 1 (input 12 -> hidden {H}, ReLU). Row j = hidden unit j: its 12 input weights for x0..x11, then its bias."]
    L+= [f"h{j}: "+" ".join(f"{v:.3g}" for v in W1[:,j])+f" | bias {b1[j]:.3g}" for j in range(H)]
    L.append(f"Layer 2 (hidden {H} -> 1 output logit; output 1 if logit > 0). Weights for h0..h{H-1} in order, then bias.")
    L.append("out: "+" ".join(f"{v:.3g}" for v in W2[:,0])+f" | bias {b2[0]:.3g}"); return "\n".join(L)
rng=np.random.default_rng(1); stim=[]
for H in [36,214]:
  for i,(k,(n,y)) in enumerate(F.items()):
    W1,b1,W2,b2=train(y,H,1000*H+i)
    for _ in range(100):
        pos=np.where(W2[:,0]>0)[0]; neg=np.where(W2[:,0]<0)[0]; kk=max(1,min(len(pos),len(neg))//2)
        flip=np.concatenate([rng.choice(pos,kk,replace=False),rng.choice(neg,kk,replace=False)])
        W2f=W2.copy(); W2f[flip]*=-1; fl=(W1,b1,W2f,b2)
        if label(fl)=='I': break
    for ver,net in [('orig',(W1,b1,W2,b2)),('flip',fl)]:
        stim.append(dict(H=H,func=k,version=ver,label=label(net),text=fmt(*net),nw=int(14*H+1),
                         # heatmap stats preserved?
                         colnorm=np.round(np.linalg.norm(net[0],axis=1),3).tolist(), outabs=float(np.abs(net[2]).sum())))
# check flip keeps magnitudes
for s0,s1 in zip(stim[0::2],stim[1::2]): assert s0['colnorm']==s1['colnorm'] and abs(s0['outabs']-s1['outabs'])<1e-9
print([(s['H'],s['func'],s['version'],s['label']) for s in stim])
OPTS="\n".join(f"{k}: output = {n}" for k,(n,_) in F.items())+"\nI: none of the above (the network computes some other function)"
key={}
for H in [36,214]:
  for st in range(2):
    items=[s for s in stim if s['H']==H and ((list(F).index(s['func'])+st)%2==0)==(s['version']=='orig')]
    random.Random(H+st).shuffle(items); body=[]
    for n,s in enumerate(items,1):
        key[f"H{H}_set{st}_S{n}"]=dict(func=s['func'],version=s['version'],label=s['label'])
        body.append(f"### Network S{n} ({s['nw']} weights)\n{s['text']}")
    open(f'prompt_H{H}_set{st}.txt','w').write(f"""Each network below is an MLP with 12 binary inputs x0..x11 (each 0 or 1), one ReLU hidden layer, and one output logit (output = 1 if logit > 0, else 0). For EACH network, decide which boolean function it computes over all 4096 inputs. Options (same for every network):
{OPTS}

Networks are independent; any option may appear any number of times, including I. Note that several options share the same inputs, so knowing which inputs matter is not enough.

{chr(10).join(body)}
""")
json.dump(key,open('answer_key.json','w'),indent=1)
