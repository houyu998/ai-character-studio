"""v4 overlay: local body/part isolation, expanded upper buttocks, fairing and point metadata.

Old scripts and v021 are unchanged. Registered only by s1_v4_boot.py.
This prepares root/free/contact groups, NOT a completed dynamic collision simulation.
"""
import os, json
import numpy as np
import bpy
import s1_core as C
import s1_custom as SC
import a_points as A
import v3_parts as PT
import v3_body as VB

_ORIG={}
_MASK={}

def active(obj):
    return bool(obj.get('v4_schema'))

LOCAL_CONTOUR=('waist_side_','hip_outer_','pelvis_side_')
ROOT_POINTS=('bust_root_','glute_center_')
def _local_delta(obj):
    d=np.zeros((len(obj.data.vertices),3))
    for pid,v in A.P.off.items():
        if pid.startswith(LOCAL_CONTOUR+ROOT_POINTS) and pid in A.P.by_id:
            d+=A.weights(obj,pid)[:,None]*np.asarray(v)[None,:]
    return d

def _offsets(obj):
    d=_ORIG['offsets'](obj)
    if d is not None and active(obj) and VB._state(obj).get('v4_connection',0)>0:d-= _local_delta(obj)
    return d

def masks(obj):
    key=(obj.name,len(obj.data.vertices))
    if key in _MASK:return _MASK[key]
    z=C.load_zones(obj);rc=A.rest_co(obj);x,y,h=rc.T
    breast=(z==C.ZID['bust'])
    # Expand the old glute patch into posterior upper pelvis/lower waist; exclude central sacral strip.
    upper=np.isin(z,[C.ZID['hip'],C.ZID['waist']]) & (y>.008) & (h<.872) & (h>.75) & (np.abs(x)>.006) & (np.abs(x)<.19)
    glute=(z==C.ZID['glute'])|upper
    _MASK[key]=dict(bust=breast,glute=glute,upper=upper)
    return _MASK[key]

def geo(obj):
    if not active(obj):return _ORIG['geo'](obj)
    key=('v4',obj.name,len(obj.data.vertices))
    if key in PT._GEO:return PT._GEO[key]
    mm=masks(obj);xr=A.rest_x(obj);ea,eb=C._adjacency(obj).T
    ls,lt,lv=PT._faces(obj);fid=np.repeat(np.arange(len(ls)),lt)
    out={}
    for p in PT.PARTS:
        k,s=p.split('_');region=mm[k]&((xr>1e-4) if s=='l' else (xr< -1e-4))
        cross=region[ea]!=region[eb];ring=np.zeros(len(region),bool)
        ring[ea[cross&region[ea]]]=True;ring[eb[cross&region[eb]]]=True
        inner=region&~ring;fin=np.zeros(len(ls),bool);np.logical_or.at(fin,fid,inner[lv])
        faces=np.flatnonzero(fin);fv=[lv[ls[f]:ls[f]+lt[f]] for f in faces]
        er=region[ea]&region[eb]
        out[p]=dict(region=np.flatnonzero(region),ring=np.flatnonzero(ring),inner=np.flatnonzero(inner),faces=faces,fv=fv,
                    vids=np.unique(np.concatenate(fv)),ea=ea[er],eb=eb[er],inner_mask=inner)
    PT._GEO[key]=out
    return out

def _soft_mask(obj,kind):
    key=('soft',obj.name,kind)
    if key not in _MASK:
        ed=C._adjacency(obj);m=masks(obj)[kind].astype(float)
        # Restrict to the actual part. The boundary smoothly tapers, no delta spills into torso.
        w=SC._smooth(m,ed[:,0],ed[:,1],2)*m
        _MASK[key]=w
    return _MASK[key]

def _clip_fn(key,kind):
    fn=SC._DEFORMERS[key]
    def local(co,ctx,amount):
        d=fn(co,ctx,amount);obj=bpy.data.objects[ctx['obj_name']]
        if active(obj) and VB._state(obj).get('v4_connection',0)>0:
            d*= _soft_mask(obj,kind)[:,None]
        return d
    return local

def _fair(co,ctx,w,iterations):
    """Paired Taubin passes: smooth high-frequency kinks with much less volume shrinkage than Laplacian alone."""
    ea,eb=ctx['ea'],ctx['eb'];x=co.copy()
    nb=np.bincount(np.r_[ea,eb],minlength=len(co)).astype(float)
    for _ in range(iterations):
        for step in (.42,-.44):
            sm=np.zeros_like(x);np.add.at(sm,ea,x[eb]);np.add.at(sm,eb,x[ea])
            lap=sm/np.maximum(nb,1)[:,None]-x
            x+=step*w[:,None]*lap
    return x-co

_VOL_TOPO={}
def patch_volume(co,g):
    """Oriented cap volume, closed at the ordered boundary edges to their mean centre (m^3).
    Engineering relative mass metric, not a medical volume or a hollow-organ model.
    """
    key=tuple(g['faces'].tolist())
    if key not in _VOL_TOPO:
        triangles=[];edge={}
        for f in g['fv']:
            for i in range(1,len(f)-1):triangles.append([f[0],f[i],f[i+1]])
            for a,b in zip(f,np.roll(f,-1)):
                k=(min(a,b),max(a,b));edge.setdefault(k,[]).append((a,b))
        border=np.asarray([v[0] for v in edge.values() if len(v)==1],int)
        _VOL_TOPO[key]=(np.asarray(triangles,int),border,np.unique(border))
    t,e,r=_VOL_TOPO[key];a,b,c=co[t[:,0]],co[t[:,1]],co[t[:,2]]
    center=co[r].mean(0)
    return float((np.einsum('ij,ij->i',a,np.cross(b,c)).sum()+np.einsum('ij,ij->i',co[e[:,1]],np.cross(co[e[:,0]],center)).sum())/6)

def d_shape(co,ctx,amount):
    obj=bpy.data.objects[ctx['obj_name']]
    if not active(obj):return np.zeros_like(co)
    st=VB._state(obj);mm=masks(obj);rc=A.rest_co(obj);zone=ctx['zone']
    H=float(np.ptp(co[ctx['body'],2]));ground=float(co[ctx['body'],2].min());fl=(co[:,2]-ground)/H
    d=np.zeros_like(co)
    # Taper upper lateral breast roots without reducing the middle/front projection.
    ax=np.abs(co[:,0]);sg=np.sign(rc[:,0]);bw=mm['bust'].astype(float)
    top=SC._ss((fl-.73)/.055)*(1-SC._ss((fl-.795)/.025))
    side=SC._ss((ax-.10)/.055)
    d[:,0]-=.012*amount*sg*bw*top*side
    # The lateral root must remain on the front rib cage, rather than wrapping behind the armpit.
    lateral_root=bw*SC._ss((ax-.105)/.065)*SC._ss((co[:,1]+.17)/.10)
    lateral_root=SC._smooth(lateral_root,ctx['ea'],ctx['eb'],6)
    lateral_root*=np.isin(zone,[C.ZID['bust'],C.ZID['ribcage'],C.ZID['neck'],C.ZID['waist']])
    d[:,0]-=.012*amount*sg*lateral_root
    d[:,1]-=.008*amount*lateral_root
    # Back low-frequency width correction, local below shoulder. Surface relief remains in the input coordinates.
    back=SC._ss((co[:,1]+.055)/.06)
    vertical=SC._ss((fl-.65)/.055)*(1-SC._ss((fl-.79)/.035))
    backmask=np.isin(zone,[C.ZID['ribcage'],C.ZID['waist'],C.ZID['neck']]).astype(float)
    d[:,0]-=.012*amount*sg*back*vertical*SC._ss((ax-.055)/.09)*backmask
    x=co+d
    # Preserve attachment boundary while fairing the masses; do not union nearby surfaces.
    w=np.clip(SC._smooth((mm['bust']|mm['glute']).astype(float),ctx['ea'],ctx['eb'],2),0,1)
    w*=np.isin(zone,[C.ZID['bust'],C.ZID['glute'],C.ZID['hip'],C.ZID['ribcage'],C.ZID['waist']])
    d+=amount*_fair(x,ctx,w,14)
    x=co+d
    # Rounded upper buttock: modest posterior fullness above the existing peak, tapering to its new attachment.
    upper=mm['glute']*SC._ss((rc[:,1]-.005)/.035)*SC._ss((np.abs(rc[:,0])-.012)/.05)
    upper*=np.exp(-((fl-.57)/.035)**2)*np.exp(-((ax-.10)/.08)**2)
    d[:,1]+=.010*amount*upper
    # Restore anterior projection lost through fairing, concentrated in the breast core rather than the torso.
    for s in ('l','r'):
        g=geo(obj)['bust_'+s];r=g['region'];inn=g['inner']
        if not len(inn):continue
        loss=float(np.min(x[r,1])-np.min(co[r,1]))
        if loss>0:
            pw=np.zeros(len(co));pw[inn]=1;pw=SC._smooth(pw,ctx['ea'],ctx['eb'],3)*mm['bust']
            apex=r[np.argmin(x[r,1])];pw/=max(pw[apex],1e-9)
            d[:,1]-=loss*amount*pw
    # Prevent left/right glute roots crossing the sagittal plane (rest-side ownership is stable).
    x=co+d;near=mm['glute']&(np.abs(rc[:,0])>.0001)&(np.abs(rc[:,0])<.035)
    desired=np.maximum(sg*x[:,0], .0015*SC._ss(np.abs(rc[:,0])/.008))
    d[near,0]+=amount*(sg[near]*desired[near]-x[near,0])
    # Midline has to stay on the midline, including former tiny asymmetric baked displacements.
    mid=(np.abs(rc[:,0])<1e-6)&(zone!=0)
    d[mid,0]=-co[mid,0]
    # Keep the old cap volume while changing the footprint/roundness. Projection compensation stays in the free core.
    for p,g in geo(obj).items():
        if not p.startswith('bust'):continue
        x=co+d;v0=patch_volume(co,g);v=patch_volume(x,g)
        w,_=PT.band_weight(x,g,.045)
        direction=np.zeros_like(co);direction[:,1]=-w
        slope=patch_volume(x+direction*.001,g)-v
        if abs(slope)>1e-12:
            step=np.clip((v0-v)/slope*.001,-.018,.018)
            d+=direction*step*amount
    # Apply these contour edits AFTER the mass fairing, so a waist edit cannot leak through the smoothing graph into breasts.
    d+=_local_delta(obj)
    for kind,prefix in [('bust','bust_root_'),('glute','glute_center_')]:
        v=np.array([st.get('v4_'+kind+'_root_'+a,0.) for a in 'xyz'])*.01
        if np.max(np.abs(v))<1e-10:continue
        for s in 'lr':
            pid=prefix+s
            if pid in A.P.by_id:
                vec=v.copy();vec[0]*=1 if s=='l' else -1
                d+=A.weights(obj,pid)[:,None]*vec[None,:]
    return d

CONTROL=dict(id='v4_connection',label='v4 根部与胸臀曲面修正',group='v3·身体结构',kind='custom',lo=0.,hi=1.,default=0.,
             have='新增·v4',stage='A3',note='局部连接、上臀覆盖和曲面修正；不代表动态碰撞已实现')
ROOT_CONTROLS=[dict(id='v4_'+kind+'_root_'+ax,label=lab+'根部 '+{'x':'左右','y':'前后','z':'上下'}[ax]+'（cm）',
    group='v3·零件安装',kind='custom',lo=-3.,hi=3.,default=0.,have='新增·v4',stage='A3',note='移动根部、零件及有限邻近连接区，默认左右同步')
    for kind,lab in [('bust','乳'),('glute','臀')] for ax in 'xyz']
HEIGHT_CONTROL=dict(id='v4_height_cm',label='胖瘦重算身高校正（cm）',group='v3·身体结构',kind='custom',lo=-5.,hi=5.,default=0.,
    have='新增·v4',stage='A3',note='自动记录胖瘦/肌肉宏重算造成的身高漂移；改变骨架长度或身高时不重新锁定')
def d_height(co,ctx,amount):
    d=np.zeros_like(co);body=ctx['body'];bottom=float(co[body,2].min());H=float(np.ptp(co[body,2]))
    d[:,2]=(co[:,2]-bottom)/max(H,1e-9)*amount*.01
    return d
HEIGHT_DESIGN=('height','proportions','head_scale_v','neck_len','torso_len','thigh_len','shin_len','foot_h')
def _apply(obj,state):
    old=C.get_state(obj)
    same_structure=all(abs(float(state.get(k,old[k]))-old[k])<1e-8 for k in HEIGHT_DESIGN)
    mass_changed=any(abs(float(state.get(k,old[k]))-old[k])>1e-8 for k in ('weight','muscle'))
    keep=active(obj) and same_structure and mass_changed
    H0=float(np.ptp(C.raw_coords(obj)[C.load_zones(obj)!=0,2])) if keep else None
    full=_ORIG['apply'](obj,state)
    if keep:
        H1=float(np.ptp(C.raw_coords(obj)[C.load_zones(obj)!=0,2]));drift=H0-H1
        if abs(drift)>.000001:
            full['v4_height_cm']=float(np.clip(full.get('v4_height_cm',0)+drift*100,-5,5))
            full=_ORIG['apply'](obj,full)
    return full

def physics_groups(obj):
    """Preparation only: root pins, movable mass and prospective contact skin. No simulation modifiers are added."""
    co=C.raw_coords(obj);report={}
    for p,po in PT.part_objects().items():
        if po is None:continue
        g=geo(obj)[p];ids=np.empty(len(po.data.vertices),np.int32);po.data.attributes[PT.VID_ATTR].data.foreach_get('value',ids)
        free,_=PT.band_weight(co,g,PT.KINDS[p.split('_')[0]]['band'])
        root=1-free;valid=np.isin(ids,g['region']);root[~np.isin(np.arange(len(co)),g['region'])]=0
        k,s=p.split('_');r=g['region'];pos=co[r].mean(0)
        if k=='bust':contact=(co[:,2]<pos[2])|(np.abs(co[:,0])<abs(pos[0])*.65)
        else:contact=(np.abs(co[:,0])<abs(pos[0])*.4)|(co[:,2]<pos[2]-.045)
        vals={'Root_Pin':root[ids]*valid,'Free_Surface':free[ids]*valid,'Contact_Surface':contact[ids]*free[ids]*valid}
        for name,weights in vals.items():
            vg=po.vertex_groups.get(name) or po.vertex_groups.new(name=name);vg.remove(list(range(len(ids))))
            for j,v in enumerate(weights):
                if v>1e-5:vg.add([j],float(v),'REPLACE')
        po['v4_connection_policy']='仅根部接口连续；自由面/接触面独立保留。权重为未来软组织控制准备，无动态模拟。'
        report[p]={name:int((v>1e-5).sum()) for name,v in vals.items()}
    obj['v4_physics_preparation']=json.dumps(report)
    return report

def upgrade_points(obj):
    import b_points as BP, r_points as RP
    defs={p['id']:p for p in A.definitions()+BP.definitions()+RP.definitions()+PT.mount_specs(obj)}
    reg=json.loads(obj['cs_points'])
    co=C.raw_coords(obj);zone=C.load_zones(obj);rc=A.rest_co(obj);mv=np.asarray(obj['cs_mvmap'],int)
    # Original green waist anchors sat just below the breast. Locate the true narrow lateral waist band on this candidate.
    skin=zone!=0;ground=float(co[skin,2].min());H=float(np.ptp(co[skin,2]))
    candidate=np.flatnonzero((zone==C.ZID['waist'])&(co[:,0]>.02)&(co[:,2]>ground+.615*H)&(co[:,2]<ground+.65*H))
    band=np.arange(ground+.615*H,ground+.65*H,.006)
    widths=[float(co[candidate[np.abs(co[candidate,2]-h)<.01],0].max()) if np.any(np.abs(co[candidate,2]-h)<.01) else np.inf for h in band]
    waist_z=float(band[np.argmin(widths)]) if len(band) else ground+.635*H
    c=candidate[np.abs(co[candidate,2]-waist_z)<.012]
    ids=c[np.argsort(-co[c,0])[:3]]
    for p in reg:
        if p['id'] in defs:
            for key in ('label','hint','group'):
                p[key]=defs[p['id']][key]
        # Local silhouette handles must not drive circumference targets that deform other body parts.
        if p['id'].startswith(LOCAL_CONTOUR):
            p['owners']={};p['local_axes']='XY';p['hint']+='；v4局部轮廓控制，不带动胸/肩背'
        if p['id'].startswith('waist_side_') and len(ids) and not obj.get('v4_anchors_calibrated'):
            vv=ids if p['id'].endswith('_l') else mv[ids]
            p['verts']=vv.tolist();p['rest']=rc[vv].mean(0).tolist()
            p['source']='v4：腰区实际最窄截面外侧，避开乳下连接'
        if p['id'].startswith(ROOT_POINTS):
            kind='bust' if p['id'].startswith('bust') else 'glute'
            p['owners']={ax.upper():['v4_'+kind+'_root_'+ax] for ax in 'xyz'};p['local_axes']='XYZ';p.pop('part',None)
            p['hint']='根部位置控制：移动根部、零件和有限接口，默认镜像；自由表面形状用红点，整体安装点另有朝向控制'
        if p.get('red') or p.get('white'):
            p['stage']='B'
        p['v4_role']='胸臀造型' if p.get('red') else '体表塑形' if p.get('white') else '结构/安装'
    obj['v4_anchors_calibrated']=1
    obj['cs_points']=json.dumps(reg,ensure_ascii=False);A.load(obj);A.P.wcache.clear()
    return {k:sum(bool(p.get(k)) for p in reg) for k in ('red','white')}

def _weights(obj,pid):
    w=_ORIG['weights'](obj,pid).copy()
    if active(obj) and pid.startswith(('w_',)+LOCAL_CONTOUR+ROOT_POINTS):
        ed=C._adjacency(obj);smooth=SC._smooth(w,ed[:,0],ed[:,1],2)
        smooth[w<1e-5]=0
        seeds=A.P.by_id[pid]['verts'];norm=max(float(smooth[seeds].min()),1e-5)
        w=np.clip(smooth/norm,0,1);w[seeds]=1
    if active(obj) and pid.startswith(LOCAL_CONTOUR):
        z=C.load_zones(obj);p=A.P.by_id[pid];rc=A.rest_co(obj);h=float(np.mean(rc[p['verts'],2]))
        fall=np.clip(1-np.abs(rc[:,2]-h)/.125,0,1);fall=SC._ss(fall)
        w*=fall;w[np.isin(z,[C.ZID['bust'],C.ZID['ribcage'],C.ZID['neck'],C.ZID['head'],C.ZID['uarm'],C.ZID['larm'],C.ZID['hand']])]=0
        w[p['verts']]=1
    return w

def _commit_generated(full,label="参考图生成草模"):
    """The reference route rebuilds the whole state from default_state(), where v4_connection is 0: one reference fit silently switched the
    v4 root/surface layer off. Keep only that on/off layer as the model had it. Other v3/v4 settings (back narrowing, part installs, root
    offsets) are NOT carried over: they are tuned to the old body, and carrying them would move the new body away from the reference.
    The pre-fit state is the last history entry (the fit rewrites the live state while it searches)."""
    import s1_panel as PN
    obj=PN.body()
    if active(obj):
        S=PN.S
        cur=S.hist[S.hi]['state'] if 0<=S.hi<len(S.hist) else C.get_state(obj)
        if 'v4_connection' in cur:
            full=dict(full);full['v4_connection']=cur['v4_connection']
    return _ORIG['commit_generated'](full,label)

def register():
    if _ORIG:return
    _ORIG.update(geo=PT.geo,weights=A.weights,offsets=A.offsets_displacement,apply=C.apply_state)
    PT.geo=geo;A.weights=_weights;A.offsets_displacement=_offsets
    import a_ui as _UI
    _ORIG['commit_generated']=_UI.commit_generated;_UI.commit_generated=_commit_generated
    C.apply_state=_apply
    A.RESID_TOL=.0002
    for key in ('bust_round','bust_upper','bust_sag','bust_yaw','bust_pitch','bust_lowfull')+tuple(SC.RED_BUST):
        if key in SC._DEFORMERS:SC._DEFORMERS[key]=_clip_fn(key,'bust')
    # Restrict glute styling to its enlarged part, while keeping the dedicated thigh-transition controls.
    for key in ('r_glute_shelf','r_glute_lowfull','r_glute_hollow'):
        if key in SC._DEFORMERS:SC._DEFORMERS[key]=_clip_fn(key,'glute')
    for ctrl in [CONTROL]+ROOT_CONTROLS+[HEIGHT_CONTROL]:
        SC.CUSTOM.append(ctrl);SC.CUSTOM_IDS.append(ctrl['id']);C.CONTROLS.append(ctrl);C.CTRL[ctrl['id']]=ctrl
        if ctrl is not CONTROL:SC._DEFORMERS[ctrl['id']]=lambda co,ctx,amount:np.zeros_like(co)
    SC._DEFORMERS['v4_connection']=d_shape
    SC._DEFORMERS['v4_height_cm']=d_height
    post=list(SC._POST);i=post.index('v3_parts_on');post[i:i]=['v4_connection','v4_height_cm'];SC._POST=tuple(post)
    base=A.Drag
    class SafeDrag(base):
        def __init__(self,obj,pid):
            super().__init__(obj,pid)
            if active(obj) and not self.sp['record_only'] and self.h0 is None:
                import a_health as H
                self.h0=H.check(obj)
        def _update(self,target):
            state,msg=super()._update(target)
            if self.sp.get('red') and self.mir_on:
                # The old inverse used only the initial Jacobian; remeasure locally when a guard/fairing makes the response nonlinear.
                for _ in range(2):
                    current=A.pos(self.obj,self.pid);resid=(np.asarray(target)-current)*A._axes_mask(self.sp)
                    if np.linalg.norm(resid)<.0002:break
                    for axis,ids in self.sp['owners'].items():
                        i='XYZ'.index(axis)
                        if abs(resid[i])<.0002:continue
                        cid=ids[0];ctrl=C.CTRL[cid];v=state[cid]
                        eps=.04 if v+.04<=ctrl['hi'] else -.04
                        probe=dict(state);probe[cid]=v+eps;C.apply_state(self.obj,probe)
                        sensitivity=(A.pos(self.obj,self.pid)[i]-current[i])/eps
                        C.apply_state(self.obj,state)
                        if abs(sensitivity)>.0002:
                            state[cid]=float(np.clip(v+resid[i]/sensitivity,ctrl['lo'],ctrl['hi']))
                    C.apply_state(self.obj,state)
            if not self.sp['record_only'] and self.sp['local_axes']:
                for _ in range(2):
                    resid=(np.asarray(target)-A.pos(self.obj,self.pid))*A._axes_mask(self.sp,'local_axes')
                    if np.linalg.norm(resid)<.0002:break
                    o,m=self._offset_for(np.asarray(A.P.off.get(self.pid,[0.,0.,0.]))+resid)
                    A.P.off[self.pid]=o.tolist()
                    if self.mir_on and self.sp['mirror']:A.P.off[self.sp['mirror']]=A.mirror_vec(o)
                    A.save_props(self.obj);C.apply_state(self.obj,state);msg+=m
                if self._kj:self._compensate(state,msg)
            return state,msg
    A.Drag=SafeDrag
    # Older directional caps were measured on a different geometry and often contained a zero that made a point immobile.
    # Use conservative candidate caps on v4 and KEEP the runtime geometry guard; the new test ledger records achieved limits.
    A.WHITE_CAPS={pid:{key:max(float(v),.004) for key,v in caps.items()} for pid,caps in A.WHITE_CAPS.items()}
    import v3_ui as VU
    VU.STRUCT_KEYS=tuple(VU.STRUCT_KEYS)+tuple(c['id'] for c in [CONTROL]+ROOT_CONTROLS+[HEIGHT_CONTROL])
