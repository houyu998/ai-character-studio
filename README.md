# AI Character Studio

A person judges by feel; a machine needs exact numbers. This project builds a way for the two to talk:
coloured points and red lines translate "make this a bit bigger" into position, range, direction and amount.

![overview](docs/images/00_overview.jpg)

Goal: let an AI coding agent (Claude Code, Codex, ...) build a usable, rigged 3D human character in Blender,
while a person acts only as the art director: they check and judge renders, and point at what to change.

- The AI does not generate meshes from scratch. It writes Blender Python that drives existing add-ons
  (MPFB parametric human, Rigify, optional FaceBuilder, Blender hair/cloth/physics).
- Vague requests like "a bit bigger" are replaced by coloured points and red lines that fix
  position, range and direction of an edit (see `docs/points.md`).
- Every change is saved as a new version; every delivery is a same-camera before/after image.

Rules for agents: `AGENTS.md`. Pipeline: `docs/pipeline.md`.

## Requirements

- Blender 4.4
- MPFB 2 (install separately; GPL, its own licence applies)
- Rigify (bundled with Blender)
- Optional: KeenTools FaceBuilder (commercial add-on) for photo-fitted faces
- Optional: a hair editor add-on, installed by yourself (not redistributed here)
- Motion: download Mixamo clips yourself, see `mocap/README.md`

## Showcase

All renders are from one character built with this pipeline (reference photos were AI-generated).

**Face (FaceBuilder photo fit, basic smile from the fit)**

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

- Done: body, face (photo fit), skin, topology and welding, rig and motion, physics, hair, clothing, basic expression (smile).
- To do: complex expressions (blink, visemes, emotions).

## License

MIT, Copyright (c) 2026 拖鞋哥998. Third-party add-ons (MPFB, FaceBuilder, hair editors) keep their own licences and are not included.
