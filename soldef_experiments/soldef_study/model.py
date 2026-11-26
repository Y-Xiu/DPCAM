"""Matched SolDef models: common Base initialization, isolated attention RNG."""
import copy
import hashlib
import torch
from torch import nn
from paper_protocol.model import create_model as paper_model
from models.attention import DPCAM,CBAM
from innovation_study.residual_candidate import ResidualAttention

VARIANTS=('none','cbam','dpcam','no_dca','no_pa','no_dsa','equal_spatial','residual_cbam','residual_dpcam')

class EqualSpatialAttention(nn.Module):
    def __init__(self,dynamic):
        super().__init__();self.branches=dynamic.branches
    def forward(self,x):
        descriptor=torch.cat((x.mean(1,keepdim=True),x.amax(1,keepdim=True)),1)
        return torch.sigmoid(sum(branch(descriptor) for branch in self.branches)/len(self.branches))


def shared_digest(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode());h.update(str(tuple(tensor.shape)).encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def create_model(config,pretrained=None,seed=42):
    variant=config['study_variant']
    if variant not in VARIANTS:raise ValueError(variant)
    cfg=copy.deepcopy(config);cfg['model']['attention']='none'
    # Construct every shared encoder/decoder/head BEFORE inserting any attention.
    # Same seed therefore gives identical common parameters and BN buffers.
    model=paper_model(cfg,pretrained)
    signature=shared_digest(model)
    blocks=[*model.decoder,model.final_fusion]
    grids=[max(2,cfg['image_size']//r) for r in model.feature_reductions[-2::-1]]
    grids.append(max(2,cfg['image_size']*2//model.feature_reductions[0]))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+1729)
        for block,grid in zip(blocks,grids):
            channels=block.conv[0].out_channels
            if variant=='none':continue
            if variant in ('cbam','residual_cbam'):attention=CBAM(channels)
            else:
                # Build full operator before removals: common submodules also start identically.
                attention=DPCAM(channels,grid)
                if variant=='no_dca':attention.dca=None
                if variant=='no_pa':attention.pa=None
                if variant=='no_dsa':attention.dsa=None
                if variant=='equal_spatial':attention.dsa=EqualSpatialAttention(attention.dsa)
            if variant.startswith('residual_'):attention=ResidualAttention(attention)
            block.attention=attention
    model.initialization_provenance={'kind':'matched_shared_base_before_attention','shared_state_sha256':signature,
        'shared_seed':seed,'attention_seed':seed+1729,'variant':variant,
        'pretrained_requested':cfg['model']['pretrained'] if pretrained is None else pretrained}
    return model


def assert_architecture(config,model):
    v=config['study_variant'];blocks=[*model.decoder,model.final_fusion]
    assert len(blocks)==4 and model.feature_reductions==[4,8,16,32]
    for block in blocks:
        m=block.attention
        if v.startswith('residual_'):
            assert isinstance(m,ResidualAttention);m=m.attention
        if v=='none':assert isinstance(m,nn.Identity)
        elif v in ('cbam','residual_cbam'):assert isinstance(m,CBAM)
        else:
            assert isinstance(m,DPCAM)
            assert tuple(getattr(m,k) is not None for k in ('dca','pa','dsa'))==tuple(v!='no_'+k for k in ('dca','pa','dsa'))
            if v=='equal_spatial':assert isinstance(m.dsa,EqualSpatialAttention) and not hasattr(m.dsa,'weights')
