# Assemble comparison grid frames: rows = views, cols = TAG list. env COLS="tag:label,...", OUT dir
import os,sys
from PIL import Image,ImageDraw,ImageFont
D=os.environ.get('FR','work/c/v105cmp/fr');OUT=os.environ['OUT'];os.makedirs(OUT,exist_ok=True)
cols=[c.split(':') for c in os.environ['COLS'].split(',')];views=['front','f34','side','b34']
fn=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22);W,H,TH=360,480,60
for f in range(30,89):
    g=Image.new('RGB',(W*len(cols),TH+H*4),(30,30,32));d=ImageDraw.Draw(g)
    for i,(t,l) in enumerate(cols):
        d.text((i*W+10,16),l,fill=(235,235,235),font=fn)
        for j,v in enumerate(views):
            im=Image.open(f'{D}/{v}_{t}_{f:04d}.png').convert('RGB');g.paste(im,(i*W,TH+j*H))
    g.save(f'{OUT}/g_{f:04d}.png')
