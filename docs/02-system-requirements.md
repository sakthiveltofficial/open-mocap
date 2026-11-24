# System Requirements

Open Mocap has two capture engines with very different needs. Pick the one that fits your machine; you can install both and switch per take.

---

## At a glance

|                      | Fast engine (MediaPipe)                     | Pro engine (SAM 3D Body)                                        |
| -------------------- | ------------------------------------------- | --------------------------------------------------------------- |
| **Blender**          | 4.2 or newer                                | 4.2 or newer                                                    |
| **Operating system** | Windows 10/11, Linux, macOS (64-bit)        | Windows 10/11, Linux. macOS untested (PyTorch MPS)              |
| **Processor / GPU**  | Any modern CPU. No GPU needed.              | NVIDIA RTX 20 / GTX 16 series or newer, 6 GB+ VRAM recommended  |
| **Disk space**       | About 1 GB free                             | About 32 GB free                                                |
| **Download**         | About 150 MB                                | About 11 GB                                                     |
| **Extras**           | None                                        | `git`, and a free Hugging Face account (Meta's model license)   |
| **Typical speed**    | 10 to 30 frames per second on a laptop CPU  | About 3 frames per second on a modern NVIDIA card               |

An internet connection is needed **once**, for the install. Capture runs fully offline afterward, and your footage never leaves your computer.

---

## Blender

Open Mocap is a Blender **extension** and needs **Blender 4.2 or newer**.

## Operating systems

- **Windows 10 / 11**: both engines.
- **Linux**: both engines, including **Flatpak** Blender. The installer opens a terminal so you can follow the setup.
- **macOS**: the Fast engine works on Intel and Apple Silicon Macs. Blender can't launch the installer for you (Gatekeeper), so run `bash installer/install.sh` from Terminal once. The Pro engine installs PyTorch with Apple's MPS backend but has not been tested on Macs.

---

## Pro engine hardware

The Pro engine runs a large 3D body model, so the GPU matters.

### NVIDIA: recommended
- **Architecture:** RTX 20-series / GTX 16-series (Turing) or newer, including RTX 30, 40 and 50 series.
- **VRAM:** 6 GB or more recommended. Less may run out of memory on long clips.
- **Older cards:** GTX 10-series and older are not supported by the CUDA 12.8 build of PyTorch.

### AMD / Intel GPUs: experimental
- **Windows:** the installer sets up ONNX Runtime with DirectML. This path comes from the original BlendCap code and has not been hardware-validated.
- **Linux:** falls back to the CPU path.

### CPU only: works, but slow
The Pro engine can run on the CPU (ONNX Runtime). Expect several seconds per frame. On a machine without an NVIDIA GPU the **Fast engine is usually the better choice**.

---

## Fast engine hardware

The Fast engine runs Google MediaPipe on the CPU. Any 64-bit processor from the last several years works. A 1080p clip typically captures at 10 to 30 frames per second on a laptop, with hands included.
