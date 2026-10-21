[English](README.md) | **中文**

# AI Character Studio

人靠感觉判断，机器需要精确的数字。这个项目在两者之间建立一套双方都能理解的对话方式：
三色点和红线把"这里再大一点"翻译成具体的位置、范围、方向和改动量。

![总览](docs/images/00_overview.jpg)

目标：让 AI 编程助手（Claude Code、Codex 等）在 Blender 里做出一个可用、可动画的 3D 人物，
人负责提供参考、做审美确认，以及必要时的人工照片拟合；AI 负责调用插件、修改和检查。

- AI 不从零生成网格，而是写 Blender Python 脚本，调用现有插件
  （MPFB 参数化人体、Rigify、可选的 FaceBuilder、Blender 自带的头发/布料/物理）。
- "大一点"这类模糊说法，换成彩色点和红线，确定改动的位置、范围和方向（见 `docs/points.md`）。
- 每次修改都另存为新版本；每次交付都是同机位的前后对比图。

AI 助手的规则：`AGENTS.md`。流水线：`docs/pipeline.md`。

## 环境要求

- Blender 4.4（已测试；其他版本未测试）
- MPFB 2（需另行安装；GPL 协议，按其自身许可使用）
- Rigify（Blender 自带）
- 可选：KeenTools FaceBuilder（商业插件），用于照片拟合脸部
- 可选：头发编辑插件，请自行安装（本仓库不分发）
- 动作：请自行下载 Mixamo 动作，见 `mocap/README.md`

## 快速开始：从空场景到第一个人体

已在 Windows + Blender 4.4 + MPFB 2 上测试。请先安装并启用 MPFB。

```bash
blender -b --factory-startup --python quickstart/first_body.py -- --size M --out my_character
```

- `--size`：XS、S、M、L、XL、2XL ……（`presets/body_sizes.json`，身高 160 cm）。
- 输出：`my_character/versions/s1_draft_v001.blend`，以及 `my_character/renders/` 里的正面、侧面灰模图。
- 不会覆盖：如果这些文件已经存在，脚本会停止。

之后在 Blender 里打开这个文件，继续阶段 A/B（`docs/pipeline.md`），或者让你的 AI 助手读 `AGENTS.md`。

![快速开始结果](docs/images/qs_first_body.jpg)

**仓库目前的状态：** 阶段 A（体型草模）可以直接从本仓库运行，已测试；阶段 B 的模块已包含。
阶段 C–H 以展示角色的制作过程作为做法说明，其中的一次性构建脚本没有包含。

## 效果展示

以下渲染图来自用这套流水线做出的同一个角色（参考照片为 AI 生成）。

**脸部（FaceBuilder 照片拟合，在界面里人工完成；微笑是静态的，随照片带出）**

| | |
|---|---|
| ![](docs/images/01_face_front.jpg) | ![](docs/images/02_face_side.jpg) |

**全身（含头发和服装）**

| | | |
|---|---|---|
| ![](docs/images/03_full_front.jpg) | ![](docs/images/04_full_side.jpg) | ![](docs/images/05_full_back.jpg) |

**拓扑网格**（四边面 99.3%，26450 个面中 26278 个）

| | | |
|---|---|---|
| ![](docs/images/09_topology_full.jpg) | ![](docs/images/10_topology_face_front.jpg) | ![](docs/images/11_topology_face_34.jpg) |

**头模：灰模和贴图**

| | | | |
|---|---|---|---|
| ![](docs/images/12_head_clay_front.jpg) | ![](docs/images/13_head_clay_34.jpg) | ![](docs/images/14_head_textured_front.jpg) | ![](docs/images/15_head_textured_34.jpg) |

**头发（Blender Hair Curves 发丝）和服装（布料模拟 + 表面形变跟随）**

| | | | |
|---|---|---|---|
| ![](docs/images/16_hair_34.jpg) | ![](docs/images/17_hair_back.jpg) | ![](docs/images/18_clothing_front.jpg) | ![](docs/images/19_clothing_34.jpg) |

**角色做完后还能继续改**：改体型，衣服自动跟随。

![](docs/images/08_edit_propagation_before_after.jpg)

视频：[布料重力垂落](docs/images/06_cloth_gravity.mp4)，[持续修改演示](docs/images/07_edit_propagation_demo.mp4)。

## 进度

- 展示角色已完成：体型、脸部（照片拟合）、皮肤、拓扑与焊接、骨骼与动作、物理、头发、服装。
- 微笑是静态的（随照片拟合带出）。可控制的表情动画（眨眼、口型、喜怒等）还没做。
- 可从本仓库直接运行：阶段 A（已测试）。阶段 C–H：只有做法说明。

## 许可

MIT，Copyright (c) 2026 拖鞋哥998。第三方插件（MPFB、FaceBuilder、头发编辑插件）按各自许可，不包含在本仓库中。
