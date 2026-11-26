# Third-party sources and attribution

The experiment runner and model adapters use the following sources. Downloaded author repositories and pretrained files retain their own terms; this repository does not relicense them. Third-party repositories, compiled operators and weight files are excluded from version control. Exact commits and source-file SHA256 values are recorded in [`configs/sources.lock.json`](../configs/sources.lock.json); initialization downloads are recorded in [`configs/assets.lock.json`](../configs/assets.lock.json).

| Model/component | Pinned source | License and use |
|---|---|---|
| EMCAD with PVTv2-B2 | [SLDGroup/EMCAD](https://github.com/SLDGroup/EMCAD/tree/26c9c31f731f749b62c5fe83f44376dac75f3aa8) | [UT Austin Research License](https://github.com/SLDGroup/EMCAD/blob/26c9c31f731f749b62c5fe83f44376dac75f3aa8/LICENSE). Research/experimental/personal use; §3.1 restricts redistribution. Users fetch from the licensor; EMCAD code is not copied into this release. |
| SegMAN-S encoder and decoder | [yunxiangfu2001/SegMAN](https://github.com/yunxiangfu2001/SegMAN/tree/9ced66ab27f23b0dbc1b0a2fe0ba42e133c55ac8) | [Apache-2.0 at repository root](https://github.com/yunxiangfu2001/SegMAN/blob/9ced66ab27f23b0dbc1b0a2fe0ba42e133c55ac8/LICENSE); embedded components retain their individual notices. |
| MambaVision-T backbone | [NVlabs/MambaVision](https://github.com/NVlabs/MambaVision/tree/7860a506b2eb844eaaae676f08461ce8c3c26f43) | [NVIDIA Source Code License-NC](https://github.com/NVlabs/MambaVision/blob/7860a506b2eb844eaaae676f08461ce8c3c26f43/LICENSE); noncommercial research/evaluation restriction applies. |
| SegFormer MiT-B2 encoder and head | NVIDIA implementation embedded in the pinned SegMAN repository, `mix_transformer.py` and `segformer_head.py`; [original NVlabs/SegFormer](https://github.com/NVlabs/SegFormer) | These files explicitly declare NVIDIA Source Code License. They are not covered solely by SegMAN's root Apache-2.0 notice. |
| MambaVision UPerNet/FCN head and shared segmentation utilities | OpenMMLab MMSegmentation code embedded in the pinned SegMAN repository | Retain OpenMMLab copyright and Apache-2.0 notices. |
| NATTEN 0.17.5 | [SHI-Labs/NATTEN](https://github.com/SHI-Labs/NATTEN/tree/v0.17.5) | Upstream BSD license and release-wheel terms. No kernels are redistributed here. |
| mamba-ssm 2.2.4 | [state-spaces/mamba](https://github.com/state-spaces/mamba/tree/v2.2.4) | Upstream Apache-2.0 and included component notices. No compiled wheel is redistributed here. |
| Selective scan `oflex` | SegMAN's pinned `kernels/selective_scan` | Preserve the embedded Mzero / Albert Gu / Tri Dao notices and relevant upstream licenses when building locally. |
| ConvNeXt-Base and ResNet-101 | [timm](https://github.com/huggingface/pytorch-image-models), [torchvision](https://github.com/pytorch/vision) | Respect package licenses and separate pretrained-weight terms. |

## Adaptations made by this experiment

`modern_study/model.py` changes class counts to the relevant SolDef task and adapts outputs to the common training/evaluation interface. EMCAD retains author multiscale supervision. MambaVision uses the author's backbone with the OpenMMLab UPerNet head and auxiliary FCN head; BatchNorm is used for the single-GPU protocol. SegMAN retains author neighborhood-attention and selective-scan operators. `modern_study/compat.py` adapts module registration/imports and renamed NATTEN symbols; it does not replace these algorithms.

The MiT classification checkpoint conversion changes tensor names and concatenates the separate key/value projection tensors in the expected order. It performs no numerical adjustment. Source and converted file hashes are checked.

`models/unet2015.py` is a local implementation of the architecture in [Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597), with valid convolutions/cropped skips and a mirror-padding adapter for common-size targets; it is not the authors' Caffe package. `models/deeplabv3_paper.py` is a local PyTorch port informed by [Chen et al. (2017)](https://arxiv.org/abs/1706.05587) and [TensorFlow DeepLab at commit 19f287775ae950e22a9b5fb82727e72cff2e3664](https://github.com/tensorflow/models/tree/19f287775ae950e22a9b5fb82727e72cff2e3664/research/deeplab); it uses torchvision ImageNet classification initialization. These should be described as paper-based implementations, not exact author training pipelines.

The legacy local experiment also contained Attention-Gated-Networks (MIT), TransUNet (Apache-2.0), and SAM2-UNet (Apache-2.0). They are not among the six modern comparison models in the released plan. If an Attention U-Net adaptation is retained in `models/attention_unet_author`, its adjacent MIT `LICENSE` and Ozan Oktay copyright notices must remain. Adding any of these methods to the active plan requires its own source/initialization documentation and validation.

Read the linked license at the pinned revision before using or redistributing a downloaded component. The downloader preserves the complete upstream checkout and license; it does not grant additional rights or change the terms of pretrained assets.
