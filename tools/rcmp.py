# Same-camera comparison renders: one version file, several views, frames F0..F1.
# env: OUTD, TAG (column name), PHYS on|off, VIEWS (comma list), F0, F1, TZ, SPAN, RX, RY
import bpy,os,math,time
from mathutils import Vector,Euler
OUTD=os.environ['OUTD'];TAG=os.environ['TAG'];os.makedirs(OUTD,exist_ok=True)
sc=bpy.context.scene;B=bpy.data.objects['CS_Body']
for o in sc.objects:
    if o.name.startswith(('WGT-','CS_D_mk','CS_Ref')) or o.type=='LIGHT':o.hide_render=True
if os.environ.get('PHYS','on')=='off':
    for m in B.modifiers:
        if m.name in('CS_Jiggle','CS_SoftCopy','CS_ThighContact'):m.show_render=m.show_viewport=False
print('K mods',[(m.name,m.show_render) for m in B.modifiers])
sc.render.engine='BLENDER_EEVEE_NEXT';sc.render.resolution_x=int(os.environ.get('RX','360'));sc.render.resolution_y=int(os.environ.get('RY','450'))
sc.render.resolution_percentage=100;sc.eevee.taa_render_samples=int(os.environ.get('SMP','16'));sc.view_settings.view_transform='AgX'
w=bpy.data.worlds.new('w');w.use_nodes=True;bg=[n for n in w.node_tree.nodes if n.type=='BACKGROUND'][0];bg.inputs[0].default_value=(0.35,0.35,0.37,1);bg.inputs[1].default_value=0.6;sc.world=w
for rot,en in (((50,0,30),3.0),((60,0,-140),1.5),((70,0,180),1.2)):
    l=bpy.data.lights.new('s','SUN');l.energy=en;lo=bpy.data.objects.new('s',l);lo.rotation_euler=Euler([math.radians(x) for x in rot]);sc.collection.objects.link(lo)
cam=bpy.data.cameras.new('c');cam.type='ORTHO';co=bpy.data.objects.new('c',cam);sc.collection.objects.link(co);sc.camera=co
VIEWS={'front':(0,-1,0),'f34':(0.8,-0.6,0.05),'side':(1,0,0),'b34':(0.8,0.6,0.05),'back':(0,1,0)}
T=Vector((0,0,float(os.environ.get('TZ','1.12'))));cam.ortho_scale=float(os.environ.get('SPAN','0.95'))
F0,F1=int(os.environ.get('F0','30')),int(os.environ.get('F1','88'))
t0=time.time()
for f in range(F0,F1+1):
    sc.frame_set(f)  # frame outer loop: cloth/geo-node state evaluated once per frame
    for v in os.environ.get('VIEWS','front,f34,side,b34').split(','):
        d=Vector(VIEWS[v]).normalized();co.location=T+d*3;co.rotation_euler=(T-co.location).to_track_quat('-Z','Y').to_euler()
        sc.render.filepath=f'{OUTD}/{v}_{TAG}_{f:04d}.png';bpy.ops.render.render(write_still=True)
print('K done',TAG,round(time.time()-t0,1),'s')
