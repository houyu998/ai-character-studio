# Verification kit

The scripts are in `../scripts/`. Copy them into the project's `work/c/<stage>/` folder or call them in place. They read environment variables.

## 1. Same-camera renders: `scripts/rcmp.py`
```bash
env OUTD="<ABSOLUTE>/fr" TAG=v105 PHYS=on VIEWS=front,f34,side,b34 F0=30 F1=88 TZ=1.0 SPAN=1.15 RX=360 RY=480 \
  "$B" -b versions/s1_draft_v105.blend --python <skill>/scripts/rcmp.py
```
- It uses an orthographic camera at a fixed distance, three suns, AgX and EEVEE at 16 samples. The frame loop is the outer loop, so physics is evaluated once per frame.
- `PHYS=off` turns off `CS_Jiggle`, `CS_SoftCopy` and `CS_ThighContact`. Use it for the "no physics" column.
- Render every column you compare with the **same** TZ, SPAN, RX and RY.

## 2. Grid video: `scripts/grid.py` + ffmpeg
```bash
OUT=work/c/x/grid COLS="off:无物理,v104:V104,v105:V105" FR=<frames dir> python <skill>/scripts/grid.py
ffmpeg -framerate 30 -start_number 30 -i work/c/x/grid/g_%04d.png -c:v libx264 -crf 18 -pix_fmt yuv420p out.mp4
```
Also make a half-speed version (`-framerate 15`) and one still frame.

## 3. Physics metrics: `scripts/metrics.py`
- It evaluates the body with physics on and off and takes the difference D, region by region (`CS_Part_*`).
- Outputs:
  - `cen_rms`: sway amplitude.
  - `vtx_peak`: the largest local offset.
  - `rough_p99`: crease or ripple indicator; the baseline is about 2.5–4 mm.
  - `LRcorr`: left/right sync. 1.0 means the two sides move together; aim for below 0.5.
- `MOUT=<json>` is required.

## 4. Static mesh checks (write per need)
- **Flipped faces:** the dot product of each face normal with the rest normal is below 0.
- **Spikes:** Laplacian outliers.
- **Intersections:** BVH overlap between the head and body, the bust and arms, and the two thighs.
- **Seams:** colour and normal jump across welded rings (render a close-up).
- **Mark problems:** add purple marker objects in the hidden collection `CS_D_AnomalyMarks`.

## 5. Your own visual check before delivering
Read 2–3 stills yourself (frames at the extremes of the motion). Look for dents, creases, intersections, black helpers and lighting artifacts. Then send them to the user with a short checklist of what to look at.
