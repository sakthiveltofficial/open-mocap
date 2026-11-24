#!/usr/bin/env python3
# Open Mocap dependency installer.
#
# Builds the add-on's private Python environment at <addon>/venv and
# downloads the AI models into <addon>/checkpoints. Two engines:
#
#   fast  Google MediaPipe (pose + hands + face). Small (~150 MB total),
#         runs on any CPU. No account needed.
#   pro   Meta SAM 3D Body + YOLO11 + MoGe-2 (the full 3D body recovery
#         pipeline). Several GB; an NVIDIA GPU is strongly recommended.
#         SAM 3D Body is GATED on Hugging Face: accept Meta's license at
#         https://huggingface.co/facebook/sam-3d-body-dinov3 and provide a
#         read token (HF_TOKEN env var, --hf-token, or the add-on prefs).
#   both  Everything.
#
# Usually started by install.sh / install.bat (which fetch `uv` and run
# this script on a managed Python 3.12), or from Blender's preferences.
# Stdlib only, so it also runs on any Python >= 3.8:
#
#   python installer/install.py --profile fast
#
# Part of Open Mocap, GPL-3.0-or-later.

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VENV = ROOT / "venv"
CKPT = ROOT / "checkpoints"
MARKER = HERE / "install_marker.json"
PROFILE_FILE = HERE / "requested_profile.txt"
PY_VERSION = "3.12"

FAST_PACKAGES = [
    "mediapipe==0.10.32",
    "opencv-python==4.13.0.92",
    "numpy==2.4.3",
    "scipy==1.17.1",
]

# Everything the Pro (SAM 3D Body) pipeline imports, minus torch and
# onnxruntime, which depend on the GPU and are added per backend.
PRO_PACKAGES = [
    "onnx==1.21.0",
    "huggingface_hub==1.7.1",
    "safetensors==0.7.0",
    "timm==1.0.25",
    "einops==0.8.2",
    "ultralytics==8.4.53",
    "hydra-core==1.3.2",
    "omegaconf==2.3.0",
    "iopath==0.1.10",
    "fvcore==0.1.5.post20221221",
    "yacs==0.1.8",
    "pycocotools==2.0.11",
    "trimesh==4.11.3",
    "roma==1.5.6",
    "loguru==0.7.3",
    "portalocker==3.2.0",
    "pytorch-lightning==2.6.1",
    "torchmetrics==1.9.0",
    "webdataset==1.0.2",
    "requests==2.34.2",
    "moge @ git+https://github.com/microsoft/MoGe.git@07444410f1e33f402353b99d6ccd26bd31e469e8",
]
TORCH = ["torch==2.7.1", "torchvision==0.22.1"]
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"

MP_BASE = "https://storage.googleapis.com/mediapipe-models/"
MEDIAPIPE_MODELS = {
    "pose_landmarker_heavy.task": MP_BASE + "pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "pose_landmarker_full.task": MP_BASE + "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_lite.task": MP_BASE + "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "hand_landmarker.task": MP_BASE + "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "face_landmarker.task": MP_BASE + "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "blaze_face_short_range.tflite": MP_BASE + "face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
}
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt"
SAM_REPO = "facebook/sam-3d-body-dinov3"
MOGE_REPO = "Ruicheng/moge-2-vits-normal"
# Public ONNX exports of the SAM backbone / YOLO / MoGe encoder, published
# by Arcomade for the BlendCap project this add-on is based on. Only the
# non-NVIDIA (ONNX Runtime) path of the Pro engine reads them.
ONNX_REPO = "Blendcap/blendcap-onnx-exports"
ONNX_FILES = ["backbone_dinov3_fp16.onnx", "backbone_dinov3_fp16.onnx_data",
              "yolo11m_pose.onnx", "moge_s_encoder_fp16.onnx"]


# ---------------------------------------------------------------- output

def say(msg=""):
    print(msg, flush=True)


def step(msg):
    say("")
    say("=" * 64)
    say("  " + msg)
    say("=" * 64)


def fail(msg, code=1):
    say("")
    say("[ERROR] " + msg)
    write_marker({"status": "failed", "error": msg})
    sys.exit(code)


def run(cmd, env=None, check=True):
    say("  $ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], env=env)
    if check and r.returncode != 0:
        fail("Command failed (exit %d): %s" % (r.returncode, cmd[0]))
    return r.returncode


# ---------------------------------------------------------------- environment

def venv_python():
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def find_uv():
    cand = os.environ.get("OPENMOCAP_UV")
    if cand and Path(cand).is_file():
        return cand
    local = HERE / ".uv" / ("uv.exe" if os.name == "nt" else "uv")
    if local.is_file():
        return str(local)
    w = shutil.which("uv")
    if w:
        return w
    # Last resort: pip-install uv next to this script with whatever Python
    # is running us (Blender's bundled Python works).
    step("Fetching uv (Python environment manager)")
    target = HERE / ".uv_pip"
    run([sys.executable, "-m", "pip", "install", "--quiet", "--target", str(target), "uv"])
    for p in target.rglob("uv.exe" if os.name == "nt" else "uv"):
        if p.is_file():
            if os.name != "nt":
                p.chmod(0o755)
            return str(p)
    fail("Could not find or install uv. Install it from https://docs.astral.sh/uv/ and retry.")


def detect_gpu():
    """Returns (gpu_type, description, backend) where backend is one of
    cuda / directml / mps / cpu."""
    sysname = platform.system()
    if sysname == "Darwin":
        return "apple", "Apple Silicon / macOS (MPS)", "mps"
    smi = shutil.which("nvidia-smi")
    if not smi and os.name == "nt":
        p = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        smi = str(p) if p.is_file() else None
    if smi:
        try:
            out = subprocess.run([smi, "--query-gpu=name", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=20)
            name = out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else ""
            if name:
                return "nvidia", "NVIDIA (CUDA) - " + name, "cuda"
        except Exception:
            pass
    if sysname == "Windows":
        return "amd-intel-win", "AMD/Intel (DirectML)", "directml"
    return "cpu-linux", "CPU (no NVIDIA GPU found)", "cpu"


def read_marker():
    try:
        return json.loads(MARKER.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_marker(update):
    data = read_marker()
    data.update(update)
    data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    MARKER.write_text(json.dumps(data, indent=2), encoding="utf-8")


def pip_install(uv, packages, extra_args=()):
    run([uv, "pip", "install", "--python", venv_python(), *extra_args, *packages])


def venv_query(code):
    r = subprocess.run([str(venv_python()), "-c", code], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


# ---------------------------------------------------------------- downloads

def download(url, dest, label=None):
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 0:
        say("  [ok] %s (already downloaded)" % dest.name)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    label = label or dest.name
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "open-mocap-installer"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                got, last = 0, 0.0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if time.time() - last > 1.0:
                        last = time.time()
                        pct = (" %3d%%" % (got * 100 // total)) if total else ""
                        sys.stdout.write("\r  %s%s  %.1f MB" % (label, pct, got / 1e6))
                        sys.stdout.flush()
            os.replace(tmp, dest)
            sys.stdout.write("\r  [ok] %s  %.1f MB\n" % (label, dest.stat().st_size / 1e6))
            return
        except Exception as e:
            say("\n  retry %d for %s (%s)" % (attempt + 1, label, e))
            time.sleep(2 ** attempt)
    fail("Download failed: " + url)


def hf_download(repo, dest_dir, files=None, token=None, gated=False):
    """Download from Hugging Face with the venv's huggingface_hub."""
    code = (
        "import sys, os\n"
        "from huggingface_hub import snapshot_download\n"
        "p = snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2],"
        " allow_patterns=(sys.argv[3].split(',') if sys.argv[3] else None),"
        " token=(os.environ.get('HF_TOKEN') or None))\n"
        "print(p)\n"
    )
    env = dict(os.environ)
    if token:
        env["HF_TOKEN"] = token
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    rc = subprocess.run([str(venv_python()), "-c", code, repo, str(dest_dir),
                         ",".join(files or [])], env=env).returncode
    if rc != 0:
        if gated:
            fail("Could not download %s. It is gated by Meta: sign in at "
                 "https://huggingface.co/%s, accept the license, create a read "
                 "token at https://huggingface.co/settings/tokens and rerun with "
                 "HF_TOKEN set (or paste it in the add-on preferences)." % (repo, repo))
        fail("Could not download %s from Hugging Face." % repo)


# ---------------------------------------------------------------- profiles

def install_fast(uv):
    step("Fast engine: installing MediaPipe")
    pip_install(uv, FAST_PACKAGES)
    step("Fast engine: downloading MediaPipe models (~60 MB)")
    for name, url in MEDIAPIPE_MODELS.items():
        download(url, CKPT / "mediapipe" / name)


def install_pro(uv, backend, token, skip_onnx=False):
    step("Pro engine: installing PyTorch (%s)" % backend)
    if backend == "cuda":
        pip_install(uv, [t + "+cu128" for t in TORCH], ["--index-url", CUDA_INDEX,
                                                         "--extra-index-url", "https://pypi.org/simple",
                                                         "--index-strategy", "unsafe-best-match"])
        ort = ["onnxruntime-gpu==1.26.0"]
    elif backend == "mps":
        pip_install(uv, TORCH)
        ort = ["onnxruntime==1.26.0"]
    else:
        pip_install(uv, TORCH, ["--index-url", CPU_INDEX,
                                "--extra-index-url", "https://pypi.org/simple",
                                "--index-strategy", "unsafe-best-match"])
        ort = ["onnxruntime-directml==1.24.4"] if backend == "directml" else ["onnxruntime==1.26.0"]

    step("Pro engine: installing SAM 3D Body dependencies")
    if not shutil.which("git"):
        say("  [!] git not found; it is needed for the MoGe package. Install git and rerun.")
    pip_install(uv, FAST_PACKAGES + PRO_PACKAGES + ort)

    step("Pro engine: downloading models")
    download(YOLO_URL, CKPT / "yolo" / "yolo11m-pose.pt", "YOLO11m-pose")
    say("  MoGe-2 (focal length estimator)")
    hf_download(MOGE_REPO, CKPT / "moge-2-vits-normal", ["model.pt"])
    say("  SAM 3D Body (gated, needs your Hugging Face token)")
    hf_download(SAM_REPO, CKPT / "sam-3d-body-dinov3",
                ["model.ckpt", "model_config.yaml", "assets/*"], token=token, gated=True)
    if backend in ("directml", "cpu") and not skip_onnx:
        ensure_onnx(required=False)


def ensure_onnx(required=False, only_backbone=False):
    files = ONNX_FILES[:2] if only_backbone else ONNX_FILES
    dest = CKPT / "onnx"
    if all((dest / f).is_file() for f in files):
        return True
    say("  ONNX exports for the ONNX Runtime path (~1.8 GB)")
    code = (
        "import sys\nfrom huggingface_hub import hf_hub_download\n"
        "for f in sys.argv[3:]:\n"
        "    hf_hub_download(sys.argv[1], f, local_dir=sys.argv[2])\n"
    )
    rc = subprocess.run([str(venv_python()), "-c", code, ONNX_REPO, str(dest), *files]).returncode
    if rc != 0:
        msg = ("ONNX exports could not be downloaded; the Pro engine will use "
               "the slower PyTorch backbone instead.")
        if required:
            fail(msg)
        say("  [!] " + msg)
        return False
    return True


# ---------------------------------------------------------------- main

def resolve_profile(cli):
    p = (cli or os.environ.get("OPENMOCAP_INSTALL_PROFILE") or "").strip().lower()
    if not p and PROFILE_FILE.is_file():
        p = PROFILE_FILE.read_text(encoding="utf-8").strip().lower()
    if not p and sys.stdin and sys.stdin.isatty():
        say("Which capture engine do you want to install?")
        say("  1) Fast  - MediaPipe, any CPU, ~150 MB  (recommended to start)")
        say("  2) Pro   - SAM 3D Body, NVIDIA GPU recommended, several GB")
        say("  3) Both")
        a = input("Choose 1, 2 or 3 [1]: ").strip()
        p = {"2": "pro", "3": "both"}.get(a, "fast")
    p = p or "fast"
    if p not in ("fast", "pro", "both"):
        fail("Unknown profile %r (use fast, pro or both)" % p)
    return p


def main():
    ap = argparse.ArgumentParser(description="Open Mocap dependency installer")
    ap.add_argument("--profile", choices=["fast", "pro", "both"])
    ap.add_argument("--hf-token", default="", help="Hugging Face read token (Pro)")
    ap.add_argument("--fresh", action="store_true", help="Delete and rebuild the venv")
    ap.add_argument("--skip-onnx", action="store_true")
    ap.add_argument("--ensure-backbone-onnx", action="store_true",
                    help="Only fetch the ONNX backbone export (used by the add-on)")
    ap.add_argument("--project-root", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.ensure_backbone_onnx:
        ensure_onnx(required=False, only_backbone=True)
        return 0

    profile = resolve_profile(args.profile)
    token = args.hf_token or os.environ.get("HF_TOKEN", "")
    gpu_type, gpu_desc, backend = detect_gpu()

    say("Open Mocap installer")
    say("  Add-on folder : %s" % ROOT)
    say("  Engine(s)     : %s" % profile)
    say("  Hardware      : %s" % gpu_desc)
    if profile in ("pro", "both") and backend != "cuda":
        say("  [!] No NVIDIA GPU detected. The Pro engine will work but is slow;")
        say("      the Fast engine is the better choice on this machine.")
    if profile in ("pro", "both") and not token:
        say("  [!] No Hugging Face token given. SAM 3D Body is gated, so its")
        say("      download will fail unless you are already logged in (hf auth login).")

    write_marker({"status": "installing", "profile": profile})

    uv = find_uv()
    step("Creating Python %s environment" % PY_VERSION)
    if args.fresh and VENV.exists():
        shutil.rmtree(VENV, ignore_errors=True)
    if not venv_python().is_file():
        run([uv, "venv", str(VENV), "--python", PY_VERSION, "--seed"])
    else:
        say("  [ok] venv already exists, reusing it")

    if profile in ("fast", "both"):
        install_fast(uv)
    if profile in ("pro", "both"):
        install_pro(uv, backend, token, skip_onnx=args.skip_onnx)
    else:
        # Face capture needs these MediaPipe files in every profile.
        pass

    torch_v = venv_query("import torch; print(torch.__version__)")
    ort_v = venv_query("import onnxruntime as o; print(o.__version__)")
    mp_v = venv_query("import mediapipe as m; print(m.__version__)")
    old = read_marker()
    engines = set(old.get("engines", [])) if old.get("status") == "success" else set()
    engines |= {"fast"} if profile in ("fast", "both") else set()
    engines |= {"pro"} if profile in ("pro", "both") else set()
    write_marker({
        "status": "success",
        "profile": profile,
        "engines": sorted(engines),
        "gpu_type": gpu_type,
        "gpu_description": gpu_desc,
        "extra": {"cuda": "cuda", "directml": "directml", "mps": "mps"}.get(backend, "mps"),
        "torch_version": torch_v,
        "ort_version": ort_v,
        "mediapipe_version": mp_v,
        "python": PY_VERSION,
    })
    step("Done. Restart Blender (or reopen the Open Mocap panel) to start capturing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
