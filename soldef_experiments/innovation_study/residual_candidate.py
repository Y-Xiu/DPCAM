"""Exploratory residual DPCAM; distinct from the manuscript's multiplicative operator."""
import torch
from torch import nn
from models.attention import DPCAM,CBAM
from paper_protocol.model import create_model


class ResidualAttention(nn.Module):
    """F_out = F + alpha * Attention(F), alpha starts at zero (ReZero-style).

    Keep all original DCA/PA/DSA operations. This changes the manuscript's
    equation and must be reported as a separate candidate, not a bug fix.
    """
    def __init__(self,attention):
        super().__init__();self.attention=attention
        self.alpha=nn.Parameter(torch.zeros(()))

    def forward(self,x):
        return x+self.alpha*self.attention(x)


def build_candidate(config,pretrained=False):
    if config['model']['backbone']!='convnext_base' or config['model']['attention'] not in ('dpcam','cbam'):
        raise ValueError('This study requires ConvNeXt-Base with DPCAM or its matched CBAM control')
    model=create_model(config,pretrained=pretrained)
    for block in [*model.decoder,model.final_fusion]:
        if not isinstance(block.attention,(DPCAM,CBAM)):raise TypeError('Expected original attention operator')
        block.attention=ResidualAttention(block.attention)
    return model


def copy_shared_initialization(model,baseline):
    """Match EVERY shared encoder/decoder/head parameter and BN buffer before training."""
    source=baseline.state_dict();target=model.state_dict()
    for name,value in source.items():
        if name not in target or target[name].shape!=value.shape:raise ValueError('Unmatched shared state: '+name)
        target[name]=value.detach().clone()
    model.load_state_dict(target,strict=True)
    return len(source)
