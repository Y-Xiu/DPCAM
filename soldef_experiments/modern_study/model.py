"""Author architectures, adapted only to five labels and the fixed SolDef protocol."""
import argparse,importlib,json,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from common import ROOT,digest
from modern_study.compat import VENDOR,legacy_mmseg,mambavision,segman_natten_compat
ASSETS=ROOT/'modern_study/assets'

def load_encoder(model,path,allowed_missing=(),allowed_extra=('head.',),expected_sha=None):
 if expected_sha and digest(path)!=expected_sha:raise ValueError('Pretrained checkpoint changed')
 with torch.serialization.safe_globals([argparse.Namespace]):
  value=torch.load(path,map_location='cpu',weights_only=True)
 for key in ['state_dict','model']:
  if isinstance(value,dict) and key in value:value=value[key];break
 value={k.removeprefix('module.'):v for k,v in value.items()}
 current=model.state_dict();missing=[k for k in current if k not in value]
 extra=[k for k in value if k not in current]
 if any(not k.startswith(allowed_missing) for k in missing):raise ValueError('Missing encoder keys: '+str(missing))
 if any(not k.startswith(allowed_extra) for k in extra):raise ValueError('Unexpected encoder keys: '+str(extra))
 for k,v in value.items():
  if k in current and v.shape!=current[k].shape:raise ValueError('Encoder shape mismatch '+k)
 model.load_state_dict({k:v for k,v in value.items() if k in current},strict=not missing)
 return {'path':str(path.relative_to(ROOT)),'sha256':digest(path),'missing_new_output_norms':missing,'discarded_classification_keys':extra,
         'loaded_tensors':len(current)-len(missing),'total_tensors':len(current),'source':'ImageNet-1k classification only'}

class AuthorSegmentor(nn.Module):
 def __init__(self,name,classes=5,pretrained=True,weights_sha=None):
  super().__init__();self.name=name;self.initialization_provenance={'kind':'random'}
  if name=='emcad':
   sys.path.insert(0,str(VENDOR/'EMCAD'))
   from lib.networks import EMCADNet
   self.net=EMCADNet(num_classes=classes,encoder='pvt_v2_b2',activation='relu6',pretrain=False)
   encoder=self.net.backbone;weight='pvt_v2_b2.pth';missing=();extra=('head.',)
  else:
   legacy_mmseg()
   if name=='segman_s':
    segman_natten_compat()
    from mmseg.models.backbones.segman_encoder import SegMANEncoder_s
    from mmseg.models.decode_heads.segman_decoder import SegMANDecoder
    self.encoder=SegMANEncoder_s(pretrained=None)
    self.head=SegMANDecoder(in_channels=[64,144,288,512],in_index=[0,1,2,3],channels=144,feat_proj_dim=288,
       dropout_ratio=.1,num_classes=classes,norm_cfg=dict(type='BN',requires_grad=True),align_corners=False)
    weight='SegMAN_Encoder_s.pth.tar';missing=();extra=('classifier.',)
   elif name=='mambavision_t':
    module=mambavision()
    self.encoder=module.MM_mamba_vision(dim=80,in_dim=32,depths=(1,3,8,4),num_heads=(2,4,8,16),
       window_size=(8,8,64,32),mlp_ratio=4,out_indices=(0,1,2,3),pretrained=None,norm_layer='ln2d',layer_scale=None,drop_path_rate=.3)
    from mmseg.models.decode_heads.uper_head import UPerHead
    from mmseg.models.decode_heads.fcn_head import FCNHead
    shared=dict(num_classes=classes,norm_cfg=dict(type='BN',requires_grad=True),dropout_ratio=.1,align_corners=False)
    self.head=UPerHead(in_channels=[80,160,320,640],in_index=[0,1,2,3],pool_scales=(1,2,3,6),channels=512,**shared)
    self.auxiliary_head=FCNHead(in_channels=320,in_index=2,channels=256,num_convs=1,concat_input=False,**shared)
    weight='mambavision_tiny_1k.pth.tar';missing=('outnorm',);extra=('head.','norm.')
   elif name=='segformer_b2':
    from mmseg.models.backbones.mix_transformer import mit_b2
    from mmseg.models.decode_heads.segformer_head import SegformerHead
    self.encoder=mit_b2()
    self.head=SegformerHead(in_channels=[64,128,320,512],in_index=[0,1,2,3],feature_strides=[4,8,16,32],
       channels=128,decoder_params=dict(embed_dim=768),dropout_ratio=.1,num_classes=classes,
       norm_cfg=dict(type='BN',requires_grad=True),align_corners=False)
    weight='mit_b2.pth';missing=();extra=('head.',)
   else:raise ValueError(name)
   encoder=self.encoder
   self.head.init_weights()
   if hasattr(self,'auxiliary_head'):self.auxiliary_head.init_weights()
  if pretrained:self.initialization_provenance=load_encoder(encoder,ASSETS/weight,missing,extra,weights_sha)
 def forward(self,x):
  if self.name=='emcad':
   outputs=self.net(x,mode='train' if self.training else 'test')
   return {'out':outputs[-1],'multi':outputs if self.training else []}
  features=self.encoder(x)
  resize=lambda y:F.interpolate(y,size=x.shape[-2:],mode='bilinear',align_corners=False)
  result={'out':resize(self.head(features))}
  if self.training and hasattr(self,'auxiliary_head'):result['aux']=resize(self.auxiliary_head(features))
  return result

def create_model(config,pretrained=None):
 pre=config['model'].get('pretrained',True) if pretrained is None else pretrained
 name=config['model']['name']
 if name in ('unet2015','deeplabv3_r101_paper'):
  from paper_protocol.model import create_model as build
  return build(config,pretrained=pre)
 return AuthorSegmentor(name,len(config['classes']),pre,config['model'].get('weights_sha256'))
