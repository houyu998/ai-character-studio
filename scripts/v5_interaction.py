"""Point interaction upgrade, opt-in per saved model. No change to the default body shape.
Blender APIs stay on the main thread. Preview is cheap; exact geometry and health guard run on confirmation.
"""
import json, time, hashlib
from collections import OrderedDict
import numpy as np
import bpy, gpu, blf
from mathutils import Vector
from bpy_extras import view3d_utils as VU
from gpu_extras.batch import batch_for_shader
import a_points as A, a_ui as UI, s1_core as C, s1_panel as PN
import v4_body as V4, s1_custom as SC

_ORIG={}; _WEIGHTS={}; _JAC=OrderedDict(); MOTION=None

class CS_PT_v5_quick(bpy.types.Panel):
    bl_label='v5 点编辑（G / X / Y / Z）'
    bl_idname='CS_PT_v5_quick';bl_space_type='VIEW_3D';bl_region_type='UI';bl_category=PN.TAB;bl_order=-200
    @classmethod
    def poll(cls,context):return active(PN.body())
    def draw(self,context):
        import v4_ui
        v4_ui.CS_PT_v4.draw(self,context)

def active(obj): return bool(obj and obj.get('v5_interaction'))

def upgrade_points(obj):
    """Stable new lateral hip anchors; old IDs/offsets remain available."""
    reg=json.loads(obj['cs_points']); ids={p['id'] for p in reg}
    if 'w_hip_side_l' not in ids:
        co=C.raw_coords(obj); z=C.load_zones(obj); rc=A.rest_co(obj)
        H=float(np.ptp(co[z!=0,2])); lo=float(co[z!=0,2].min())
        # The user's marks are the upper lateral pelvis between waist and buttock maximum.
        candidates=np.flatnonzero(np.isin(z,[C.ZID['hip'],C.ZID['glute']]) & (co[:,0]>.06) &
                                  (co[:,2]>lo+.555*H) & (co[:,2]<lo+.59*H))
        if not len(candidates): raise RuntimeError('没有找到胯侧表面，不能创建空白点')
        chosen=candidates[np.argsort(-co[candidates,0])[:3]]
        for side,label,vv in [('L','左',chosen),('R','右',A.P.mvmap[chosen])]:
            pid='w_hip_side_'+side.lower()
            p=A._spec(pid,label+'胯侧（腰臀过渡）','白·腰腹',side,
                '局部调节胯侧与腰臀过渡；不改髋关节、肩背和胸；默认左右镜像',verts=vv.tolist(),
                main=True,parent='pelvis_center',axes='XYZ',local_axes='XYZ',owners={},
                region=['hip','glute','waist'],radius=8,kind='surface',source='v5：上胯外侧表面固定顶点，角色自身左右')
            p.update(white=True,stage='B',mirror='w_hip_side_'+('r' if side=='L' else 'l'),rest=rc[vv].mean(0).tolist())
            reg.append(p)
    for p in reg:
        if p.get('red'):
            p.setdefault('v5_previous_owners',p['owners']);p.setdefault('v5_previous_axes',p['axes'])
            p.update(owners={},axes='XYZ',local_axes='XYZ',radius=8)
            p['hint']='局部XYZ曲面塑形；根部保持，默认镜像。原形态参数仍在胸臀滑块；到范围/新增问题时提示实际量。'
            kind='bust' if p['group']=='红·胸' else 'glute'
            g=V4.geo(obj)[kind+'_'+p['side'].lower()]
            if not set(p['verts']).issubset(set(g['inner'].tolist())):
                oldpos=co[np.asarray(p['verts'],int)].mean(0) if 'co' in locals() else C.raw_coords(obj)[p['verts']].mean(0)
                vv=g['inner'][np.argsort(np.linalg.norm(C.raw_coords(obj)[g['inner']]-oldpos,axis=1))[:3]]
                p['verts']=vv.tolist();p['rest']=A.rest_co(obj)[vv].mean(0).tolist()
                p['source']='v5：移到对应零件自由表面，避开固定根部；稳定顶点编号'
        if p['id'].startswith('w_hip_side_'): p['v4_role']='胯侧体表塑形'
    selected=A.P.selected
    obj['cs_points']=json.dumps(reg,ensure_ascii=False);A.load(obj);A.P.selected=selected if selected in A.P.by_id else None
    _WEIGHTS.clear();_JAC.clear()
    return len(reg)

def _weights(obj,pid):
    if not active(obj):return _ORIG['weights'](obj,pid)
    sp=A.P.by_id[pid]
    key=(obj.as_pointer(),pid,tuple(sp['verts']),tuple(sp['region']),sp['radius'])
    if key in _WEIGHTS:return _WEIGHTS[key]
    w=_ORIG['weights'](obj,pid).copy()
    if sp.get('red'):
        kind='bust' if sp['group']=='红·胸' else 'glute'
        g=V4.geo(obj)[kind+'_'+sp['side'].lower()]
        allowed=np.zeros(len(w),bool);allowed[g['region']]=True
        # Keep anatomical root interfaces fixed. The unconnected free surface can be adjusted.
        allowed[g['ring']]=False
        ed=C._adjacency(obj);w=SC._smooth(w*allowed,ed[:,0],ed[:,1],3)*allowed
        seeds=np.asarray(sp['verts'],int)
        norm=float(w[seeds].min())
        if norm<1e-6:raise ValueError('红点锚点落在根部固定区，需要重新定位：'+pid)
        w=np.clip(w/norm,0,1);w[seeds]=1
    if pid.startswith('w_hip_side_'):
        z=C.load_zones(obj);w[np.isin(z,[C.ZID['bust'],C.ZID['ribcage'],C.ZID['neck'],C.ZID['head'],C.ZID['uarm'],C.ZID['thigh'],C.ZID['calf'],C.ZID['foot']])]=0
    w.setflags(write=False);_WEIGHTS[key]=w
    return w

def _fair_fast(co,ctx,w,iterations):
    """Same Taubin equations, NumPy bincount replaces slow repeated scatter-add."""
    ea,eb=ctx['ea'],ctx['eb'];n=len(co);x=co.copy()
    degree=np.bincount(np.r_[ea,eb],minlength=n).astype(float)
    den=np.maximum(degree,1)[:,None]
    for _ in range(iterations):
        for step in (.42,-.44):
            sm=np.column_stack([np.bincount(ea,weights=x[eb,j],minlength=n)+np.bincount(eb,weights=x[ea,j],minlength=n) for j in range(3)])
            x+=step*w[:,None]*(sm/den-x)
    return x-co

def register(obj):
    if _ORIG:return
    _ORIG.update(weights=A.weights,fair=V4._fair,local_delta=V4._local_delta,finish=UI.finish_drag)
    bpy.utils.register_class(CS_PT_v5_quick)
    A.weights=_weights;V4._fair=_fair_fast
    def local_delta(obj):
        d=_ORIG['local_delta'](obj)
        if active(obj):
            for pid,v in A.P.off.items():
                if pid.startswith(('r_','w_hip_side_')) and pid in A.P.by_id:
                    d+=A.weights(obj,pid)[:,None]*np.asarray(v)[None,:]
        return d
    V4._local_delta=local_delta
    # Red local displacement bounds are explicit; health guard can further reduce them.
    A.MAX_OFF_BY['r_']=.025;A.MAX_OFF_BY['w_hip_side_']=.03
    base=A.Drag
    class CachedDrag(base):
        def _update(self,target):
            if active(self.obj) and (self.sp.get('red') or self.pid.startswith('w_hip_side_')) and not self.sp['owners']:
                target=np.asarray(target,float)
                if not np.isfinite(target).all():self.cancel();return dict(self.st0),['无效距离，已恢复']
                delta=(target-self.p0)*A._axes_mask(self.sp)
                offset,msg=self._offset_for(np.asarray(self.off0.get(self.pid,[0,0,0]))+delta)
                msg=['局部偏移已到上限 %.1f cm'%(A.max_off(self.pid)*100) if m.startswith('【局部偏移已到上限') else m for m in msg]
                if self.sp.get('red'):
                    dx=A.weights(self.obj,self.pid)*(offset[0]-self.off0.get(self.pid,[0,0,0])[0])
                    if self.mir_on and self.sp['mirror']:
                        dx+=A.weights(self.obj,self.sp['mirror'])*(-offset[0]-self.off0.get(self.sp['mirror'],[0,0,0])[0])
                    kind='bust' if self.sp['group']=='红·胸' else 'glute';fraction=1.0
                    for side,sign in [('l',1),('r',-1)]:
                        ids=V4.geo(self.obj)[kind+'_'+side]['inner'];change=dx[ids]*sign;initial=self.X0[ids,0]*sign
                        bad=(change< -1e-9)&(initial>1e-4)
                        if bad.any():fraction=min(fraction,float(np.clip((initial[bad]-1e-4)/(-change[bad]),0,1).min()))
                    if fraction<1:
                        old=np.asarray(self.off0.get(self.pid,[0,0,0]));offset=old+(offset-old)*fraction
                        msg.append('中线保护：已缩小移动量，避免左右零件相互穿过')
                A.P.off={k:list(v) for k,v in self.off0.items()};A.P.off[self.pid]=offset.tolist()
                if self.mir_on and self.sp['mirror']:A.P.off[self.sp['mirror']]=A.mirror_vec(offset)
                A.save_props(self.obj);C.apply_state(self.obj,self.st0)
                actual=A.pos(self.obj,self.pid)-self.p0
                if np.linalg.norm(actual-delta)>.0005:msg.append('未完全到位：请看实际移动量与保护提示')
                return dict(C.get_state(self.obj)),msg
            return super()._update(target)
        def _jacobian(self):
            if not active(self.obj):return super()._jacobian()
            key=(self.obj.as_pointer(),self.pid,self.mir_on,
                 hashlib.sha256(self.X0.tobytes()).digest(),json.dumps(self.st0,sort_keys=True),
                 json.dumps(self.sp,sort_keys=True),json.dumps(self.off0,sort_keys=True))
            if key in _JAC:
                d=_JAC[key];self.groups=d['groups'];self.dX=d['dX'];self.weak=list(d['weak']);self.h_keep=d['h_keep']
                _JAC.move_to_end(key);return
            super()._jacobian()
            _JAC[key]=dict(groups=self.groups,dX=self.dX,weak=list(self.weak),h_keep=getattr(self,'h_keep',0))
            while len(_JAC)>12:_JAC.popitem(last=False)
    A.Drag=CachedDrag
    def finish_drag(obj,drag,label_extra=''):
        if not active(obj):return _ORIG['finish'](obj,drag,label_extra)
        # Drag.update already applied the exact guarded shape. Record it without applying it a second time.
        st=dict(C.get_state(obj));A.save_props(obj);s=PN.S
        del s.hist[s.hi+1:]
        s.hist.append(dict(label='拖点：'+drag.sp['label']+label_extra,state=st,t=time.time(),co=None,pts=A.snapshot()))
        s.hi=len(s.hist)-1;s.after=st;s.after_pts=A.snapshot()
        PN._health_after_commit();PN.sync_props(bpy.context.scene,st)
    UI.finish_drag=finish_drag

def numeric_event(buffer,event):
    """Pure parsing, numbers are centimetres. Decimal/minus/backspace supported."""
    if event.type=='BACK_SPACE':return buffer[:-1],True
    ch=getattr(event,'ascii','') or ''
    if len(ch)==1 and ch in '0123456789.-':
        trial=buffer+ch
        if len(trial)>12 or trial.count('.')>1 or trial.count('-')>1 or ('-' in trial and not trial.startswith('-')):return buffer,True
        return trial,True
    return buffer,False

def mouse_target(p0,start,mouse,region,rv3d,axis=None,precision=False):
    p0=np.asarray(p0,float);d=np.asarray(mouse,float)-np.asarray(start,float)
    if precision:d*=.1
    if axis:
        i='XYZ'.index(axis);u=np.zeros(3);u[i]=.1
        s0=VU.location_3d_to_region_2d(region,rv3d,Vector(p0));s1=VU.location_3d_to_region_2d(region,rv3d,Vector(p0+u))
        v=np.asarray(s1)-np.asarray(s0) if s0 is not None and s1 is not None else np.zeros(2)
        if np.dot(v,v)<4:return p0.copy(),'该轴朝向视线；请输入数字，或切换视图'
        return p0+u*float(np.dot(d,v)/np.dot(v,v)),''
    a=VU.region_2d_to_location_3d(region,rv3d,start,Vector(p0))
    b=VU.region_2d_to_location_3d(region,rv3d,np.asarray(start)+d,Vector(p0))
    return p0+np.asarray(b-a),''

def _init_move(tool,context,event,pid,key_mode,start=None):
    global MOTION
    obj=PN.body()
    tool.drag=A.Drag(obj,pid);tool._smooth_off(obj)
    tool.key_mode=key_mode;tool.axis=None;tool.number='';tool.last=0;tool.cost=0
    tool.start_mouse=start or (event.mouse_x-tool.win.x,event.mouse_y-tool.win.y)
    tool.target=tool.drag.p0.copy();tool.msgs=[];tool.pending_click=None
    unit='mm' if tool.drag.sp.get('face') else 'cm'
    MOTION=dict(axis=None,label=tool.drag.sp['label'],pid=pid,request_cm=[0,0,0],unit=unit,hint='G移动；X/Y/Z选轴；数字'+unit+'；Enter确认，Esc取消')
    UI._tag_redraw()

def _target(tool,context,event):
    if tool.number not in ('','-','.','-.'):
        if not tool.axis:return tool.drag.p0.copy(),'输入距离前请按 X、Y 或 Z'
        q=tool.drag.p0.copy();value=float(tool.number)
        if not np.isfinite(value):return q,'无效距离'
        q['XYZ'.index(tool.axis)]+=value*.01
        return q,''
    mouse=(event.mouse_x-tool.win.x,event.mouse_y-tool.win.y)
    return mouse_target(tool.drag.p0,tool.start_mouse,mouse,tool.win,tool.area.spaces.active.region_3d,tool.axis,event.shift)

def _preview(tool,context,event):
    global MOTION
    q,hint=_target(tool,context,event);tool.target=q
    t=time.perf_counter();tool.msgs=tool.drag.preview(q);tool.cost=time.perf_counter()-t;tool.last=time.perf_counter()
    scale=1000 if tool.drag.sp.get('face') else 100
    MOTION.update(axis=tool.axis,request_cm=((q-tool.drag.p0)*scale).tolist(),hint=hint or ('距离 '+tool.number+' '+MOTION['unit'] if tool.number else 'Enter/左键确认；Esc/右键取消'))
    UI._tag_redraw()

def _finish(tool,context,cancel=False):
    global MOTION
    obj=PN.body();d=tool.drag
    if cancel:
        d.cancel();PN.log('已取消移动：'+d.sp['label']);tool.msgs=['已取消，模型恢复']
    else:
        if np.linalg.norm(tool.target-d.p0)<1e-8:
            d.cancel();tool.msgs=['未移动，不新增历史']
        else:
            _,tool.msgs=d.update(tool.target)
            unit='mm' if d.sp.get('face') else 'cm'
            actual=(A.pos(obj,d.pid)-d.p0)*(1000 if d.sp.get('face') else 100)
            UI.finish_drag(obj,d)
            tool.msgs=list(tool.msgs)+[('实际移动 X %.2f / Y %.2f / Z %.2f '+unit)%tuple(actual)]
            PN.log('移动 '+d.sp['label']+'：'+'；'.join(tool.msgs))
    tool._smooth_back(obj);tool.drag=None;tool.pending_click=None
    UI.UI.flash=(d.sp['label'],tool.msgs,time.time());MOTION=None;UI._tag_redraw()

def modal(tool,context,event):
    """Select without evaluating the body. G and axes only intercept while the point tool is enabled."""
    global MOTION
    obj=PN.body()
    if not UI.S.tool_on:
        if tool.drag:_finish(tool,context,True)
        return {'FINISHED'}
    if tool.drag is None:
        if not tool._inside(event):return {'PASS_THROUGH'}
        if event.type=='ESC' and event.value=='PRESS':UI.S.tool_on=False;UI._tag_redraw();return {'FINISHED'}
        if event.type=='LEFTMOUSE' and event.value=='PRESS' and not UI.S.showing_before:
            mouse=(event.mouse_x-tool.win.x,event.mouse_y-tool.win.y)
            pid=UI.pick(obj,tool.win,tool.area.spaces.active.region_3d,mouse)
            if pid:
                A.P.selected=pid;tool.pending_click=(pid,mouse);UI._tag_redraw();return {'RUNNING_MODAL'}
            tool.pending_click=None;return {'PASS_THROUGH'}
        if event.type=='LEFTMOUSE' and event.value=='RELEASE':
            tool.pending_click=None;return {'PASS_THROUGH'}
        if event.type=='MOUSEMOVE' and getattr(tool,'pending_click',None):
            pid,start=tool.pending_click;mouse=(event.mouse_x-tool.win.x,event.mouse_y-tool.win.y)
            if np.linalg.norm(np.asarray(mouse)-start)>6:
                _init_move(tool,context,event,pid,False,start);_preview(tool,context,event)
            return {'RUNNING_MODAL'}
        if event.type=='G' and event.value=='PRESS':
            if not A.P.selected:PN.log('请先单击选中一个点（变黄并显示名称）');return {'RUNNING_MODAL'}
            if UI.S.showing_before:PN.log('请先切回现在，再移动点');return {'RUNNING_MODAL'}
            _init_move(tool,context,event,A.P.selected,True);return {'RUNNING_MODAL'}
        if event.type=='Z' and event.value=='PRESS' and event.ctrl:
            if event.shift:bpy.ops.cs.redo()
            else:bpy.ops.cs.undo()
            UI._tag_redraw();return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}
    if event.type in ('ESC','RIGHTMOUSE') and event.value=='PRESS':
        _finish(tool,context,True);return {'RUNNING_MODAL'}
    if event.type in ('X','Y','Z') and event.value=='PRESS':
        if event.type not in tool.drag.sp['axes']:
            MOTION['hint']='此点不能沿 '+event.type+' 移动：'+tool.drag.sp['hint'];UI._tag_redraw();return {'RUNNING_MODAL'}
        tool.axis=None if tool.axis==event.type else event.type
        _preview(tool,context,event);return {'RUNNING_MODAL'}
    if event.value=='PRESS':
        tool.number,handled=numeric_event(tool.number,event)
        if handled:_preview(tool,context,event);return {'RUNNING_MODAL'}
    if event.type=='MOUSEMOVE':
        if time.perf_counter()-tool.last>max(.025,tool.cost*1.2):_preview(tool,context,event)
        return {'RUNNING_MODAL'}
    confirm=(event.type in ('RET','NUMPAD_ENTER') and event.value=='PRESS') or (tool.key_mode and event.type=='LEFTMOUSE' and event.value=='PRESS') or (not tool.key_mode and event.type=='LEFTMOUSE' and event.value=='RELEASE')
    if confirm:
        if tool.number in ('-','.','-.') or (tool.number and not tool.axis):
            MOTION['hint']='请选择方向并输入完整数字';UI._tag_redraw();return {'RUNNING_MODAL'}
        _preview(tool,context,event);_finish(tool,context);return {'RUNNING_MODAL'}
    # Do not send active point movement events to Blender's object transform operator.
    return {'RUNNING_MODAL'}

def draw_axes(obj):
    if not A.P.selected or bpy.context.region_data is None:return
    sp=A.P.by_id.get(A.P.selected)
    if not sp:return
    p=A.pos(obj,sp['id'],C.raw_coords_cached(obj));sh=gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.depth_test_set('NONE');gpu.state.line_width_set(2)
    for i,col in enumerate(((1,.15,.12,1),(.15,1,.25,1),(.15,.4,1,1))):
        ax='XYZ'[i]
        if ax not in sp['axes']:continue
        if MOTION and MOTION['axis'] and MOTION['axis']!=ax:continue
        u=np.zeros(3);u[i]=.012 if sp.get('face') else .055
        batch=batch_for_shader(sh,'LINES',{'pos':[tuple(p-u),tuple(p+u)]});sh.bind();sh.uniform_float('color',col);batch.draw(sh)
    gpu.state.line_width_set(1)

def draw_status(obj):
    ctx=bpy.context
    if ctx.space_data is None or ctx.space_data.type!='VIEW_3D' or not A.P.selected:return
    sp=A.P.by_id.get(A.P.selected)
    if not sp:return
    unit='毫米' if sp.get('face') else '厘米'
    lines=['已选：'+sp['label']+'  ['+sp['id']+']','可动轴 '+sp['axes']+'；单击选点 → G → X/Y/Z → 数字（'+unit+'）']
    if MOTION:
        lines+=['移动方向 '+(MOTION['axis'] or '视图平面')+('；目标 X %.2f / Y %.2f / Z %.2f '+MOTION.get('unit','cm'))%tuple(MOTION['request_cm']),MOTION['hint']]
    # Bottom-left panel avoids Blender's native view labels and the legacy top-left warning text.
    width=min(730,ctx.region.width-45);top=40+24*len(lines)
    sh=gpu.shader.from_builtin('UNIFORM_COLOR');gpu.state.blend_set('ALPHA')
    pts=[(20,25),(20+width,25),(20+width,top+20),(20,25),(20+width,top+20),(20,top+20)]
    batch=batch_for_shader(sh,'TRIS',{'pos':pts});sh.bind();sh.uniform_float('color',(.025,.025,.035,.9));batch.draw(sh)
    gpu.state.blend_set('NONE')
    y=top;blf.size(0,15);blf.color(0,1,1,.25,1)
    for ln in lines:blf.position(0,35,y,0);blf.draw(0,ln);y-=23
