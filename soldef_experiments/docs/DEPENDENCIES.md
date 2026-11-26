# Environment and external assets

Run commands from `soldef_experiments/` (first `cd soldef_experiments` from the repository root). This release contains source, configurations and split IDs; it does not contain datasets, pretrained weights, third-party checkouts, CUDA binaries or virtual environments. Nothing in the setup commands starts an experiment.

## Base environment

The reference environment is Linux x86_64, Python 3.11, PyTorch 2.5.1 with CUDA 12.1, torchvision 0.20.1 and timm 1.0.24. CPU can run unit checks and the CNN/SegFormer adapters; SegMAN and MambaVision need their native CUDA operators for the intended forward pass. A working NVIDIA driver is required for GPU execution. The release was checked without a usable GPU driver; installing these dependencies is not evidence that full-size GPU preflight passed.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

ConvNeXt-Base uses timm's pretrained encoder initialization (`convnext_base`, ImageNet-1k). Its download is managed by timm/Hugging Face, independently of the comparison-weight fetcher. The training runtime records the encoder pretrained configuration and initialization provenance; it does not independently SHA256-pin the downloaded ConvNeXt checkpoint. Preparing the cache requires network access on a fresh machine, and reproductions should retain the resolved initialization file and its hash separately. Do not silently substitute a random initialization or a Tiny backbone.

## Modern comparison dependencies and author code

Read [THIRD_PARTY.md](THIRD_PARTY.md) before fetching source. The fetcher downloads directly from each author's repository at the pinned commit, then verifies the recorded source and license SHA256 values. Existing unknown or modified checkouts are rejected; they are never reset or overwritten.

```bash
python -m pip install -r requirements-modern.txt
python scripts/fetch_sources.py --list
python scripts/fetch_sources.py
python scripts/fetch_sources.py --check
```

`requirements-modern.txt` adds the observed compatible versions: mmcv 1.7.2 (Python package, not `mmcv-full`), mmengine 0.10.7, einops 0.8.0, fvcore, numpy 2.2.6 and build helpers. Do not combine the upstream repositories' requirements files: they prescribe conflicting Torch, timm and MMCV versions. The adapter imports only the required author modules; it does not require an installed modern MMSeg/MMDetection stack.

The native wheel URLs in `configs/assets.lock.json` are specifically **CPython 3.11 / Linux x86_64 / Torch 2.5 / CUDA 12 / C++ ABI false**. Check these properties before installation:

```bash
python -c "import platform,sys,torch; print(sys.version,platform.machine(),torch.__version__,torch.version.cuda,torch._C._GLIBCXX_USE_CXX11_ABI)"
python scripts/fetch_weights.py --only natten mamba_ssm
python -m pip install --no-deps 'modern_study/assets/natten-0.17.5+torch250cu121-cp311-cp311-linux_x86_64.whl' 'modern_study/assets/mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
```

For another platform or ABI, build these operators from their pinned upstream releases and document a separate environment; do not install the incompatible wheels. The compatibility adapter restores renamed NATTEN exports, without replacing the attention or selective-scan algorithms.

## SegMAN selective-scan extension

Install a complete [CUDA 12.1 Toolkit](https://developer.nvidia.com/cuda-12-1-0-download-archive), including `nvcc`, CUDART, cuBLAS and cuSPARSE development headers. Set `CUDA_HOME` to that installation. The small `cuda_nvcc`, `cuda_cudart` and `cuda_cccl` archives recorded in the asset lock are historical build inputs, **not a complete toolkit**; downloading just those three archives fails at `cusparse.h` unless the other development headers are supplied.

The study uses `selective_scan_cuda_oflex` from SegMAN. Build only the required `oflex` extension using an isolated copy of the author's build setup; the original checkout stays untouched. A C++17 compiler and the Python development headers are also required.

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
python -m pip install setuptools wheel
python - <<'PY'
from pathlib import Path
import shutil, subprocess, sys, tempfile
source = Path('modern_study/vendor/SegMAN/kernels/selective_scan').resolve()
text = (source / 'setup.py').read_text()
old = 'MODES = ["core", "ndstate", "oflex"]'
assert old in text, 'Unexpected pinned build setup'
with tempfile.TemporaryDirectory(prefix='dpcam-scan-') as directory:
    build = Path(directory)
    (build / 'setup.py').write_text(text.replace(old, 'MODES = ["oflex"]', 1))
    shutil.copytree(source / 'csrc', build / 'csrc')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-build-isolation',
                    '--no-deps', str(build)], check=True)
PY
python -c "import selective_scan_cuda_oflex, natten; from modern_study.compat import mambavision; mambavision(); print('Native imports OK')"
```

Selecting one build target does not alter the operator implementation. The original experiment had a machine-specific symlink to another checkout; that symlink and compiled `.so` files are deliberately absent from the release. This recipe must be followed on the target machine and subsequently validated by the full-size preflight before training.

## Comparison initialization weights

All comparison pretrained weights are **classification initialization**, not pretrained segmentation models. U-Net is initialized from scratch. Fixed file sizes, SHA256 values and source URLs are in `configs/assets.lock.json`.

```bash
python scripts/fetch_weights.py --only pvt_v2_b2 mambavision_t mit_b2_source mit_b2 resnet101
```

The MiT fetcher performs the recorded lossless key renaming and key/value concatenation through `modern_study.convert_mit`; both the source and serialized output are hash checked. Serialization requires Torch 2.5.1 and the recorded `mit_b2.pth` archive name. It never replaces an existing mismatched output.

SegMAN's author classification weight must be downloaded from [this exact file](https://drive.google.com/file/d/152hZo70iJKO2mg9DWxBgG0na-wnqq4WF/view), named `SegMAN_Encoder_s.pth.tar`. This is from the author's [ImageNet classification folder](https://drive.google.com/drive/folders/1QYU7nhpe0ddH7bPxI7VH4drc__07uEHs); ADE20K/Cityscapes segmentation weights in the paper's results tables are different files. Save the download in `modern_study/assets/`, or import it from a local folder:

```bash
python scripts/fetch_weights.py --only segman_s --cache-dir /path/to/downloads --offline
python scripts/fetch_weights.py --check
```

The accepted SegMAN SHA256 is `be12d9954e49a12266913018d5565261825c53c530cf8a0ba0fdb05e4ea33059`. A browser download or `gdown` is sufficient; the fetcher does not automate Google Drive confirmation pages. A missing manual asset yields a nonzero exit with its exact URL.

The destination mapping is:

| Model / asset | Expected local path |
|---|---|
| EMCAD / PVTv2-B2 | `modern_study/assets/pvt_v2_b2.pth` |
| SegMAN-S | `modern_study/assets/SegMAN_Encoder_s.pth.tar` |
| MambaVision-T | `modern_study/assets/mambavision_tiny_1k.pth.tar` |
| SegFormer / MiT source | `modern_study/assets/mit_b2_hf.bin` |
| SegFormer / converted MiT | `modern_study/assets/mit_b2.pth` |
| DeepLabv3 / ResNet-101 | `weights/resnet101-63fe2227.pth` |

## Offline checks and cache reuse

`--check` never downloads, converts or writes files. `--root` can validate assets/checkouts in an existing experiment directory without changing it. `--cache-dir` imports only matching filenames with verified hashes. `--offline` forbids network access and permits local MiT conversion; `--list` prints the selected entries.

```bash
python scripts/fetch_sources.py --check --root /path/to/existing/experiment
python scripts/fetch_weights.py --check --root /path/to/existing/experiment
python scripts/fetch_weights.py --cache-dir /path/to/weight-cache --offline
```

After data preparation, architecture/initialization checks can be invoked per configuration with `python -m comparison_study.verify_setup --config configs/comparison_study/comparison_task1_emcad.json`. This constructs the network and writes a setup report; it is separate from training, and SegMAN/MambaVision construction alone does not validate their full CUDA forward/backward pass. Do not treat missing native operators as permission to use a simplified substitute network.
