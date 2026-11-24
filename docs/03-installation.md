# Installation

Installing Open Mocap has two parts: the Blender add-on itself (a small zip), and the capture engines (a private Python environment plus the AI models), which the built-in installer sets up for you.

---

## Step 1: Install the add-on

1. Download the add-on: on [GitHub](https://github.com/sakthiveltofficial/open-mocap), click **Code ▸ Download ZIP**, or download a release zip.
2. In Blender, open **Edit ▸ Preferences ▸ Add-ons**, open the menu in the top-right corner and choose **Install from Disk**, then select the zip.
3. Enable **OpenMocap** in the list if it isn't enabled automatically.

Once enabled, you'll find it in the **3D Viewport sidebar** (press **N**) under the **OpenMocap** tab, and in the **Output** tab of the Properties editor.

---

## Step 2: Install the capture engines

1. Open **Edit ▸ Preferences ▸ Add-ons**, find **OpenMocap** and expand it. (The **Open Preferences** button in the **Setup** section of the sidebar panel takes you there.)
2. Click **Install Dependencies**. An install guide appears inside the preferences panel.
3. Choose what to install:
   - **Fast (MediaPipe)**: about 150 MB, any CPU, no account needed.
   - **Pro (SAM 3D Body)**: about 11 GB, NVIDIA GPU recommended.
   - **Both**: install everything and switch per capture.
4. **Pro only:** Meta's SAM 3D Body model is gated on Hugging Face.
   1. Sign in (or sign up for free) at [huggingface.co](https://huggingface.co).
   2. Open [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3) and accept the license.
   3. Create a **read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   4. Paste it into the **Hugging Face token** field. It is only passed to the installer for this download and is never saved.
   5. On the next page, read and accept Meta's SAM License.
5. Click **Install**. A console window opens and:
   - fetches [uv](https://docs.astral.sh/uv/), a small standalone Python manager, if it isn't on your system,
   - builds a private **Python 3.12** environment at `venv/` inside the add-on folder (Blender's own Python is not touched),
   - installs the packages for the engine(s) you chose,
   - downloads the models into `checkpoints/`.
6. When the console says **Done**, restart Blender. The preferences show which engines are ready.

You can add the other engine later: click **Add Engine / Reinstall** and pick it. Existing packages and models are reused.

### How long does it take?
Mostly it depends on your connection. The Fast engine is a couple of minutes. The Pro engine downloads about 11 GB. If a download is interrupted, run the installer again; finished files are skipped.

---

## Installing from a terminal

Everything the wizard does is in `installer/`. From the add-on folder:

```bash
# Linux / macOS
bash installer/install.sh --profile fast        # fast, pro or both
HF_TOKEN=hf_xxx bash installer/install.sh --profile pro
```

```bat
:: Windows (Command Prompt or double-click install.bat)
set HF_TOKEN=hf_xxx
installer\install.bat --profile both
```

Other options: `--fresh` rebuilds the environment from scratch, `--skip-onnx` skips the optional ONNX exports on non-NVIDIA machines. The installer logs to `installer/install.log` and records the result in `installer/install_marker.json`.

If you already have [uv](https://docs.astral.sh/uv/) or any Python 3.8+, you can also run `python installer/install.py --profile fast` directly.

### Where things go

| Path (inside the add-on folder) | What |
|---|---|
| `venv/` | Private Python 3.12 environment used for capture |
| `checkpoints/mediapipe/` | MediaPipe pose, hand and face models (Fast engine and face capture) |
| `checkpoints/sam-3d-body-dinov3/` | SAM 3D Body checkpoint and MHR skeleton (Pro) |
| `checkpoints/yolo/`, `checkpoints/moge-2-vits-normal/` | Person detector and field-of-view model (Pro) |
| `checkpoints/onnx/` | Optional ONNX exports for the non-NVIDIA Pro path |

### Model sources

| Model | Source |
|---|---|
| MediaPipe Pose / Hand / Face landmarkers | `storage.googleapis.com/mediapipe-models` (Google) |
| SAM 3D Body | [huggingface.co/facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3) (gated) |
| YOLO11m-pose | [github.com/ultralytics/assets releases](https://github.com/ultralytics/assets/releases) |
| MoGe-2 ViT-S | [huggingface.co/Ruicheng/moge-2-vits-normal](https://huggingface.co/Ruicheng/moge-2-vits-normal) |
| ONNX exports (optional) | [huggingface.co/Blendcap/blendcap-onnx-exports](https://huggingface.co/Blendcap/blendcap-onnx-exports), published by Arcomade |

---

## Platform notes

### Windows
Nothing needs to be installed beforehand. For the Pro engine, install [Git for Windows](https://git-scm.com/download/win) first (needed for the MoGe package). If PyTorch fails to import with `WinError 126`, install the [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist) (x64).

### Linux
The installer opens a visible terminal. On minimal systems MediaPipe may need the EGL/GLES libraries (`sudo apt install libegl1 libgles2` on Debian/Ubuntu). Both standard and **Flatpak** Blender are supported.

> **Flatpak Blender, one-time permission.** Flatpak runs Blender in a sandbox, so Open Mocap needs permission to run its installer on the host. If it's missing, the install guide shows the exact command to run, with a **Copy** button. Run it once and restart Blender. Under Flatpak the Hugging Face token can't be passed through; run `hf auth login` on the host first, or install from a terminal.

### macOS
Run `bash installer/install.sh --profile fast` from Terminal in the add-on folder (Blender can't launch it for you because of Gatekeeper).

---

## Reinstalling and uninstalling

Both live in the OpenMocap section of **Add-on Preferences**:

- **Add Engine / Reinstall**: re-runs the setup. Use it to add the other engine, after a GPU change, or if an install was interrupted.
- **Uninstall**: removes the environment and downloaded models to free disk space, after you confirm. This doesn't remove the add-on itself.
- **Open Install Folder** shows where everything lives on disk.

---

## Verifying the install

1. Load a short test video (see [Quick Start](04-quick-start.md)).
2. Pick an engine and run **Preview**: it should find the person across the clip within seconds.

If Preview finds your subject, capture is working. Otherwise see [Troubleshooting & FAQ](12-troubleshooting-and-faq.md).
