# Licensing & Credits

Open Mocap is built on open-source and research software. This page summarizes the licensing in plain language and credits the projects it relies on. The authoritative texts are the `LICENSE` and `NOTICE.txt` files and the `licenses/` folder.

---

## Open Mocap's license

Open Mocap is licensed under the **GNU General Public License, version 3 or later (GPL-3.0-or-later)**, like Blender itself. Its source is at [github.com/sakthiveltofficial/open-mocap](https://github.com/sakthiveltofficial/open-mocap).

Open Mocap is a modified version of **[BlendCap](https://github.com/Arcomade/BlendCap) 1.0.5 by Arcomade** (© 2026 Arcomade, GPL-3.0-or-later). The BlendCap copyright notices are kept, and the changes are listed in `NOTICE.txt`. Open Mocap is an independent project and is not affiliated with or endorsed by Arcomade.

---

## Can I use my captured animations commercially?

**Yes.** The animation you create is yours to use, including in commercial work. Two notes:

- You are responsible for having the rights to the footage you capture from.
- With the **Pro** engine, the body model is under Meta's SAM License, which permits commercial use royalty-free but prohibits certain uses of the model itself (military and weapons applications and the like; full terms in `licenses/SAM_LICENSE.txt`). The **Fast** engine uses Google's MediaPipe models under Apache-2.0.

---

## Third-party components

Model weights are **not** included in this repository. The installer downloads them from their official sources (see [Installation](03-installation.md#model-sources)), and you accept each model's license when you download it.

| Component | Role | License |
|---|---|---|
| **MediaPipe** (Google) | Fast engine body and hand tracking; face capture | Apache-2.0 |
| **Meta SAM 3D Body** | Pro engine body and hand model | Meta SAM License |
| **Meta DINOv3** | Pro engine vision backbone (architecture code bundled in `models/dinov3_hub/`) | Meta DINOv3 License |
| **Ultralytics YOLO11** | Pro engine person detection | AGPL-3.0 |
| **Fast-SAM-3D-Body** | Pro engine accelerated inference (bundled in `models/fast-sam-3d-body/`) | MIT wrapper over Meta-SAM-licensed code |
| **MoGe** (Microsoft) | Pro engine camera field of view | MIT (with an Apache-2.0 subcomponent) |
| **ICT-FaceKit** (USC ICT) | Face expression basis used by face refinement | MIT (derived data asset) |
| **PyTorch** / **ONNX Runtime** | Pro engine runtimes | BSD-3-Clause / MIT |
| **uv** (Astral) | Installer's Python manager | MIT / Apache-2.0 |

### A note on the AGPL
The Pro engine loads the AGPL-3.0 YOLO11 detector. GPL v3 section 13 allows combining a GPL v3 work with an AGPL v3 component; the AGPL's terms (including its network-use clause) continue to apply to the YOLO portion. This is one more reason the source stays public.

---

## Credits

- **BlendCap** by **Arcomade**: the add-on Open Mocap is built from, including the Pro engine integration, the post-processing pipeline, face capture, retargeting and the UI.
- The research teams and maintainers behind **MediaPipe**, **SAM 3D Body**, **DINOv3**, **YOLO11**, **Fast-SAM-3D-Body**, **MoGe**, **ICT-FaceKit**, **PyTorch**, **ONNX Runtime** and **uv**.
- **Blender** and its community.
