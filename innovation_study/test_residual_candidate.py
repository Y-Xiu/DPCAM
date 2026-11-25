import sys,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import pytest
from common import ROOT
from paper_protocol.model import create_model
from innovation_study.residual_candidate import build_candidate,copy_shared_initialization,ResidualAttention


@pytest.mark.parametrize('attention',['dpcam','cbam'])
def test_identity_start_and_learnable_gate(attention):
    torch.set_num_threads(2);torch.manual_seed(42)
    cfg=json.loads((ROOT/'configs/two_task/twotask_task2_dpcam.json').read_text());cfg['image_size']=64;cfg['model']['attention']=attention
    basecfg=copy.deepcopy(cfg);basecfg['model']['attention']='none'
    baseline=create_model(basecfg,pretrained=False).eval();candidate=build_candidate(cfg).eval()
    copy_shared_initialization(candidate,baseline)
    x=torch.randn(2,3,64,64)
    with torch.no_grad():a=baseline(x)
    b=candidate(x);torch.testing.assert_close(a,b,rtol=0,atol=0)
    b.square().mean().backward()
    modules=[m for m in candidate.modules() if isinstance(m,ResidualAttention)]
    assert len(modules)==4
    assert all(m.alpha.grad is not None and torch.isfinite(m.alpha.grad) and m.alpha.grad.abs()>0 for m in modules)
    # At exact identity initialization the attention weights intentionally start with zero gradient.
    assert all(p.grad is not None and p.grad.abs().max()==0 for m in modules for p in m.attention.parameters())
    candidate.zero_grad(set_to_none=True)
    with torch.no_grad():
        for m in modules:m.alpha.fill_(.1)
    candidate(x).square().mean().backward()
    assert all(any(p.grad is not None and p.grad.abs().max()>0 for p in m.attention.parameters()) for m in modules)
    assert all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None)
