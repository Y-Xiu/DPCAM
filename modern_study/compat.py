"""Import only used author modules; skip unrelated model-registration side effects.
All imported backbone/head operator implementations remain unmodified.
"""
import importlib,importlib.util,sys,types
from pathlib import Path
VENDOR=Path(__file__).resolve().parent/'vendor'
def package(name,path):
 if name not in sys.modules:
  m=types.ModuleType(name);m.__path__=[str(path)];m.__package__=name;sys.modules[name]=m
 return sys.modules[name]
def legacy_mmseg():
 root=VENDOR/'SegMAN/segmentation/mmseg'
 package('mmseg',root)
 for n in ['models','models.backbones','models.decode_heads','models.losses','core','utils']:
  package('mmseg.'+n,root.joinpath(*n.split('.')))
 core=sys.modules['mmseg.core']
 core.build_pixel_sampler=importlib.import_module('mmseg.core.seg.builder').build_pixel_sampler
 util=sys.modules['mmseg.utils']
 util.get_root_logger=importlib.import_module('mmseg.utils.logger').get_root_logger
 losses=sys.modules['mmseg.models.losses']
 losses.accuracy=importlib.import_module('mmseg.models.losses.accuracy').accuracy
 importlib.import_module('mmseg.models.losses.cross_entropy_loss')
 return importlib.import_module('mmseg.models.builder')
def mambavision():
 builder=legacy_mmseg()
 # Both original decorators register the same backbone; no detection code is used.
 from mmcv.utils import Registry
 registry=types.ModuleType('mmseg.registry');registry.MODELS=builder.BACKBONES
 sys.modules['mmseg.registry']=registry
 package('mmdet',VENDOR)
 registry=types.ModuleType('mmdet.registry');registry.MODELS=Registry('unused_detection_registration')
 sys.modules['mmdet.registry']=registry
 # The author only calls selective_scan_fn. Avoid importing unrelated Mamba LM/transformers code.
 spec=importlib.util.find_spec('mamba_ssm')
 if spec is None:raise ImportError('Install pinned mamba_ssm wheel')
 package('mamba_ssm',Path(spec.origin).parent)
 package('mamba_ssm.ops',Path(spec.origin).parent/'ops')
 name='modern_author_mambavision'
 if name not in sys.modules:
  spec=importlib.util.spec_from_file_location(name,VENDOR/'MambaVision/semantic_segmentation/tools/mamba_vision.py')
  module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 return sys.modules[name]

def segman_natten_compat():
 """NATTEN 0.17.5 renamed four legacy exports; model forward already uses na2d_*.
 Legacy FLOP aliases are not used by our profiler; no attention kernel is replaced.
 """
 import natten.functional as f,natten.flops as fl
 if not hasattr(f,'natten2dav'):f.natten2dav=f.na2d_av
 if not hasattr(f,'natten2dqkrpb'):f.natten2dqkrpb=f.na2d_qk
 if not hasattr(fl,'qk_2d_rpb_flop'):fl.qk_2d_rpb_flop=fl.na_qk_flop_count
 if not hasattr(fl,'av_2d_flop'):fl.av_2d_flop=fl.na_av_flop_count
