# Licensing & Credits

OpenMocap is built on excellent open-source and research software. This page summarizes the licensing in plain language and credits the projects OpenMocap relies on. The full, authoritative license texts ship **with the product**: the `LICENSE` and `NOTICE.txt` files and the `licenses/` folder in your download.

---

## OpenMocap's license

OpenMocap is licensed under the **GNU General Public License, version 3 or later (GPL-3.0-or-later)**. © 2026 Arcomade.

In practical terms: OpenMocap is free/open-source software, like Blender itself, and its source code is publicly available at **[@@@](https://@@@)**. The paid marketplace version includes a one-click installer, and bundled model access that makes setup smooth and user-friendly, as well as priority support and updates.

---
## Can I use my captured animations commercially?

**Yes.** The animation you create with OpenMocap is yours to use, including in commercial work. The underlying body model is provided under Meta's SAM License, which permits commercial use royalty-free. Two common-sense notes: you're responsible for having the rights to the footage you capture from, and the SAM License prohibits certain uses of the model itself (military/weapons applications and the like, the full terms are in `licenses/SAM_LICENSE.txt`).

---

## Third-party components

OpenMocap bundles, depends on, or redistributes the following. The complete inventory is in `licenses/THIRD_PARTY.txt`; the license-significant ones are:

| Component | Role in OpenMocap | License |
|---|---|---|
| **Meta SAM 3D Body** | Body + hand capture model | Meta SAM License |
| **Meta DINOv3** | Vision backbone used by the body model | Meta DINOv3 License |
| **Ultralytics YOLO11** | Person detection | AGPL-3.0 |
| **Fast-SAM-3D-Body** | Accelerated inference library | MIT wrapper over Meta-SAM-licensed code |
| **MoGe** (Microsoft) | Camera field-of-view estimation | MIT (with an Apache-2.0 subcomponent) |
| **MediaPipe** (Google) | Facial landmark tracking | Apache-2.0 |
| **ICT-FaceKit** (USC ICT) | Face expression basis used by face refinement | MIT (derived data asset) |
| **PyTorch** | Machine-learning runtime | BSD-3-Clause |
| **ONNX Runtime** | Cross-platform model runtime | MIT |

A copy of each required license travels with OpenMocap, as those licenses require.

### A note on the open-source license
Because OpenMocap includes the AGPL-3.0-licensed YOLO11 detector as a core component, the whole add-on is distributed under GPL v3 (which is compatible with, and bridges to, the AGPL component). This is why OpenMocap's source is, and must remain, publicly available.

---

## Credits

OpenMocap stands on the work of the research teams and open-source maintainers behind **SAM 3D Body**, **DINOv3**, **YOLO11**, **Fast-SAM-3D-Body**, **MoGe**, **MediaPipe**, **ICT-FaceKit**, **PyTorch**, and **ONNX Runtime**, and on **Blender** itself and its community. Thank you to all of them.

---

## Questions

Licensing questions about using OpenMocap or its output? Get in touch through the store you bought it from, see [Troubleshooting & FAQ ▸ Getting help](12-troubleshooting-and-faq.md#getting-help).
