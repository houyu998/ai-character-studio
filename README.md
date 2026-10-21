**English** | [中文](README.zh-CN.md)

# AI Character Studio

A person judges by feel; a machine needs exact numbers. This project builds a way for the two to talk:
coloured points and red lines translate "make this a bit bigger" into position, range, direction and amount.

![overview](docs/images/00_overview.jpg)

Goal: let an AI coding agent (Claude Code, Codex, ...) build a usable, rigged 3D human character in Blender,
The person provides references, makes the aesthetic calls and does the manual photo fit where needed;
the AI calls the add-ons, makes the edits and runs the checks.

- The AI does not generate meshes from scratch. It writes Blender Python that drives existing add-ons
  (MPFB parametric human, Rigify, optional FaceBuilder, Blender hair/cloth/physics).
- Vague requests like "a bit bigger" are replaced by coloured points and red lines that fix
  position, range and direction of an edit (see `docs/points.md`).
- Every change is saved as a new version; every delivery is a same-camera before/after image.

Rules for agents: `AGENTS.md`. Pipeline: `docs/pipeline.md`.

## Requirements

- Blender 4.4 (tested; other versions untested)
- MPFB 2 (install separately; GPL, its own licence applies)
- Rigify (bundled with Blender)
- Optional: KeenTools FaceBuilder (commercial add-on) for photo-fitted faces
- Optional: a hair editor add-on, installed by yourself (not redistributed here)
- Motion: download Mixamo clips yourself, see `mocap/README.md`

## Quick start: empty scene to first body

Tested on Windows with Blender 4.4 and MPFB 2. Install and enable MPFB first.

```bash
blender -b --factory-startup --python quickstart/first_body.py -- --size M --out my_character
```

- `--size`: XS, S, M, L, XL, 2XL ... (`presets/body_sizes.json`, height 160 cm).
- Output: `my_character/versions/s1_draft_v001.blend` plus grey front and side renders in `my_character/renders/`.
- It never overwrites: it stops if those files already exist.

From there, open the file in Blender and continue with stage A/B (`docs/pipeline.md`), or let your agent read `AGENTS.md`.

![quick start result](docs/images/qs_first_body.jpg)

**What this repository is today:** stage A (body draft) runs from this repository and is tested. Stage B modules are included.
Stages C–H are documented as a recipe from the showcase character; their one-off builder scripts are not included.

## Showcase

All renders are from one character built with this pipeline (reference photos were AI-generated).

**Face (FaceBuilder photo fit, done by hand in the GUI; the smile is static, from the photos)**

| | |
|---|---|
| ![](docs/images/01_face_front.jpg) | ![](docs/images/02_face_side.jpg) |

**Full body with hair and clothing**

| | | |
|---|---|---|
| ![](docs/images/03_full_front.jpg) | ![](docs/images/04_full_side.jpg) | ![](docs/images/05_full_back.jpg) |

**Topology** (99.3 % quads, 26278 of 26450 faces)

| | | |
|---|---|---|
| ![](docs/images/09_topology_full.jpg) | ![](docs/images/10_topology_face_front.jpg) | ![](docs/images/11_topology_face_34.jpg) |

**Head model: clay and textured**

| | | | |
|---|---|---|---|
| ![](docs/images/12_head_clay_front.jpg) | ![](docs/images/13_head_clay_34.jpg) | ![](docs/images/14_head_textured_front.jpg) | ![](docs/images/15_head_textured_34.jpg) |

**Hair (Blender Hair Curves) and clothing (cloth simulation + surface deform)**

| | | | |
|---|---|---|---|
| ![](docs/images/16_hair_34.jpg) | ![](docs/images/17_hair_back.jpg) | ![](docs/images/18_clothing_front.jpg) | ![](docs/images/19_clothing_34.jpg) |

**Keep editing after the character is finished**: change the body, the clothes follow.

![](docs/images/08_edit_propagation_before_after.jpg)

Videos: [cloth under gravity](docs/images/06_cloth_gravity.mp4), [edit propagation demo](docs/images/07_edit_propagation_demo.mp4).

## Status

- Showcase character done: body, face (photo fit), skin, topology and welding, rig and motion, physics, hair, clothing.
- The smile is static (it comes with the photo fit). Controllable expression animation (blink, visemes, emotions) is not done.
- Runnable from this repository: stage A (tested). Stages C–H: recipe only.

## License

MIT, Copyright (c) 2026 拖鞋哥998. Third-party add-ons (MPFB, FaceBuilder, hair editors) keep their own licences and are not included.
