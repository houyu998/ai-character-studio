# Blender character pipeline: reference images → three-colour points → animated 3D character

You (the AI) write and run Blender Python headless. The user is the **art director**: they judge looks and approve each stage. Never ask them to read logs or check vertices.

## Hard rules (always)
1. **Never overwrite.**
   - Every change is saved as a new `versions/s1_draft_vNNN.blend`.
   - Scripts must `assert not dest.exists()` before saving.
   - User hand edits (their files like `*_1.blend`, the shape key `cs_user_edit`) are never touched. Build on top of them.
2. **One stage at a time.** Each stage delivers a new version, same-camera before/after images or video, and a short Chinese note. Label every delivery `待确认` and stop for the user's visual approval before the next stage.
3. **Look at grey first, materials second.** Hair, clothes and materials hide structural problems.
4. **Measure, don't guess.** Check intersections, flips, creases and left/right sync with numbers (see `docs/checks.md`). Let the user judge beauty.
5. **Record everything.**
   - Append technical notes to the project's appendix file.
   - Keep deliverables in one review folder.
   - Report to the user briefly, in Chinese, in plain words.
6. **Save tokens.**
   - Don't re-read whole files or print big arrays.
   - Grep for `^K` lines only (scripts print key results with a `K ` prefix).
   - Fix the most likely cause first.
   - Render only the affected views.

## Environment
- Blender **4.4** is production: `C:/Program Files/Blender Foundation/Blender 4.4/blender.exe`.
  - Add-ons needed: **MPFB** (`bl_ext.user_default.mpfb`), **KeenTools FaceBuilder**, and Rigify (built in).
  - Blender 5.2 + MCP exists only for experiments on *copies*.
- Run scripts headless, in the background for anything over 2 minutes:
  ```bash
  B="C:/Program Files/Blender Foundation/Blender 4.4/blender.exe"
  "$B" -b versions/s1_draft_vNNN.blend --python work/c/vNNN/step.py -- --save 2>&1 | grep -E "^K|Error|Trace"
  ```
- **Use absolute output paths.** Blender resolves relative paths against C:/, not the shell's working directory.
- With about 32 GB of RAM, run at most 2–3 Blender processes in parallel.
- Project root: `<project root>`. For a new character, copy the structure: `versions/ work/c/ reviews/ textures/ reference/ mocap/`.

## The pipeline (details and script entry points: `docs/pipeline.md`)

| # | Stage | Main tool | Gate (user approves) |
|---|---|---|---|
| A | Body draft from reference, text or preset | MPFB human + `scripts/s1_*.py`, green points | overall proportions |
| B | Body sculpt + separate bust/glute parts | white points (body surface), red points (bust/glute), sliders | local shapes, part joins |
| C | Head and face | **FaceBuilder** fit from photos (the user pins in the GUI), joined to the body; three-colour face points for fine tuning only | likeness |
| D | Skin, makeup, eyes | user photo texture → body UV, makeup layer, eye scale | close-up look |
| E | Topology and welding | bust/glute ring weld 1:1, midline, joint edge loops | seams invisible, bends clean |
| F | Rig and motion | Rigify via MPFB `add_builtin_rig` + `generate_rigify_rig`, Mixamo FBX retarget, in-place loop | deformation in motion |
| G | Physics | jiggle bones (`CS_Jiggle`) + voxel-cage cloth soft body (`CS_Soft`) + thigh contact geometry nodes | motion feel, left/right independent |
| H | Hair, clothing, portfolio render | Hair Curves, cloth | each separately |
| I | Expressions | shape keys (ARKit-style), *to do* | expressions don't break the face |

**Lesson:**
- Three-colour points are great for body structure but could not reach facial likeness (v031–v064, many reworks).
- Photo fitting (FaceBuilder) solved likeness in one step.
- So for faces: fit from photos first, then use points for small adjustments.

## Three-colour point system (spec: `docs/points.md`)
- **Green** = structure and mounting. Shoulder, waist, hip, knee and elbow positions; where the bust and glute parts sit; the eye/nose/mouth/ear layout.
- **White** = surface sculpt. Shoulders, arms, back, belly, legs, hands and feet; on the face, the skull, forehead, cheekbone and jaw outline.
- **Red** = local shape. Bust and glute (upper, lower, inner, outer, orientation); on the face, eyelids, lip edges, nose wings and ear rims.
- **Purple** = problem markers only, never draggable.
- All points: mirrored by default with the midline locked; smooth limited falloff anchored on anatomy; undo, cancel and save-reopen must work.

## How to run a stage (template)
1. Find the newest version and any user hand-edit files (`ls -t versions | head`). Read the appendix section for the last stage.
2. Write `work/c/vNNN/<step>.py`:
   - Load the newest file and apply the change.
   - Print `K` lines with key numbers.
   - Save only with `--save` to the new version.
3. Verify:
   - Run `scripts/rcmp.py` (same-camera renders, before and after).
   - For physics, also run `scripts/metrics.py`.
   - Combine with `scripts/grid.py` and ffmpeg into one mp4 or png.
   - Inspect the stills yourself before showing the user.
4. Deliver: `reviews/<folder>/NN_<what>_待确认.(png|mp4)`, then append the appendix note and send a short Chinese report with a checklist of what to look at.

## Known traps (full list: `docs/traps.md`)
- Fixed render lights create fake artifacts. Example: an "open back of head" turned out to be lighting.
- In Blender 4.4, when you change an action you must also set `animation_data.action_slot`.
- Physics helpers (`CS_Soft` collider and cages) show up **black** in the viewport if they get unhidden. They are invisible in renders and must not be deleted.
- Cloth rebakes are not exactly reproducible, so compare variants within one batch.
- The Armature modifier normalises weights. Add a root filler weight (1 − Σ), or small jiggle weights get pulled to full strength.
- A boot script may re-hook the skin chain; respect version flags (see `docs/traps.md`).
