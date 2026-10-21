# Physics metrics vs. the same file with physics modifiers off.
# Per region (bust/glute L/R): centroid offset peak/rms (mm), max single-vertex offset (mm),
# max inward offset along rest normal (dent indicator, mm), L/R vertical correlation.
import bpy,os,json,numpy as np
sc=bpy.context.scene;B=bpy.data.objects['CS_Body'];me=B.data;n=len(me.vertices)
PH=('CS_Jiggle','CS_SoftCopy','CS_ThighContact')
for m in B.modifiers:
    if m.name not in PH and m.type not in('ARMATURE',):m.show_viewport=False   # keep vertex indices 1:1
print('K mods',[(m.name,m.show_viewport) for m in B.modifiers])
gi={g.name:g.index for g in B.vertex_groups}
def grp(name,th=0.5):
    k=gi[name];return np.array([v.index for v in me.vertices for g in v.groups if g.group==k and g.weight>=th])
R={f'{p}_{s}':grp(f'CS_Part_{q}_{s}') for p,q in(('bust','Bust'),('glute','Glute')) for s in('L','R')}
F0,F1=int(os.environ.get('F0','30')),int(os.environ.get('F1','88'))
def run(on):
    for m in B.modifiers:
        if m.name in PH:m.show_viewport=on and m.name in ON
    P=[];N=[]
    for f in range(F0,F1+1):
        sc.frame_set(f);dg=bpy.context.evaluated_depsgraph_get();e=B.evaluated_get(dg).to_mesh()
        a=np.empty(n*3);e.vertices.foreach_get('co',a);P.append(a.reshape(-1,3).copy())
        if not on:b=np.empty(n*3);e.vertices.foreach_get('normal',b);N.append(b.reshape(-1,3).copy())
        B.evaluated_get(dg).to_mesh_clear()
    return np.array(P),(np.array(N) if N else None)
ON=[m.name for m in B.modifiers if m.name in PH and m.show_render]
on,_=run(True);off,nr=run(False)
D=on-off;out={}
adj=[[] for _ in range(n)]
for e in me.edges:
    a,b=e.vertices;adj[a].append(b);adj[b].append(a)
for k,ix in R.items():
    d=D[:,ix];c=d.mean(1);cm=np.linalg.norm(c,axis=1)
    vm=np.linalg.norm(d,axis=2).max();inward=-(d*nr[:,ix]).sum(2)
    lap=np.array([[np.linalg.norm(D[t,i]-D[t,adj[i]].mean(0)) for i in ix] for t in range(len(D))])  # roughness: crease/ripple indicator
    out[k]=dict(cen_peak=round(cm.max()*1000,1),cen_rms=round(np.sqrt((cm**2).mean())*1000,1),
                vtx_peak=round(vm*1000,1),dent_peak=round(inward.max()*1000,1),rough_peak=round(lap.max()*1000,2),rough_p99=round(float(np.percentile(lap,99))*1000,2),cz=c[:,2])
res={}
for p in('bust','glute'):
    a,b=out[f'{p}_L']['cz'],out[f'{p}_R']['cz']
    res[p+'_LRcorr']=round(float(np.corrcoef(a,b)[0,1]),2) if a.std()>1e-6 and b.std()>1e-6 else None
for k,v in out.items():v.pop('cz');res[k]=v
print('K metrics',os.environ.get('TAG'),json.dumps(res,ensure_ascii=False))
open(os.environ['MOUT'],'w').write(json.dumps(res))
